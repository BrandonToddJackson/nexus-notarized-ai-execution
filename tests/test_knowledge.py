"""Phase 2 tests: embeddings, store, context builder, and reasoning gates.

No ChromaDB on disk — KnowledgeStore tests use chromadb.EphemeralClient (in-memory).
EmbeddingService tests mock the sentence-transformers model load.
"""

import uuid
import pytest
import numpy as np

from nexus.types import (
    PersonaContract, RiskLevel, RetrievedContext, KnowledgeDocument,
    ChainPlan, ChainStatus, Seal, ActionStatus,
    IntentDeclaration, AnomalyResult, GateResult, GateVerdict,
)
from nexus.exceptions import ToolError
from nexus.knowledge.embeddings import EmbeddingService
from nexus.knowledge.store import KnowledgeStore
from nexus.knowledge.context import ContextBuilder
from nexus.reasoning.think_act import ThinkActGate
from nexus.reasoning.continue_complete import ContinueCompleteGate
from nexus.reasoning.escalate import EscalateGate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embedder: each text → unit vector based on its hash."""
    out = []
    for t in texts:
        np.random.seed(abs(hash(t)) % (2**31))
        v = np.random.randn(16).astype(float)
        v /= np.linalg.norm(v)
        out.append(v.tolist())
    return out


def _in_memory_store() -> KnowledgeStore:
    """KnowledgeStore backed by chromadb EphemeralClient (no disk)."""
    store = KnowledgeStore(persist_dir="/tmp/test_chroma", embedding_fn=_fake_embed)
    import chromadb
    store._client = chromadb.EphemeralClient()
    return store


def _tid() -> str:
    """Unique tenant ID per call — prevents EphemeralClient cross-test contamination."""
    return f"t_{uuid.uuid4().hex[:8]}"


def _doc(tenant_id="t1", namespace="docs", content="hello world", access_level="internal") -> KnowledgeDocument:
    return KnowledgeDocument(
        tenant_id=tenant_id,
        namespace=namespace,
        source="test.txt",
        content=content,
        access_level=access_level,
    )


def _researcher() -> PersonaContract:
    return PersonaContract(
        name="researcher",
        description="Searches information",
        allowed_tools=["knowledge_search"],
        resource_scopes=["kb:*"],
        intent_patterns=["find information"],
        risk_tolerance=RiskLevel.LOW,
        max_ttl_seconds=120,
    )


def _chain(steps: int = 1, seals_done: int = 0) -> ChainPlan:
    step_list = [{"action": f"step {i}", "tool": "knowledge_search"} for i in range(steps)]
    seal_ids = [f"seal-{i}" for i in range(seals_done)]
    return ChainPlan(
        tenant_id="t1",
        task="test task",
        steps=step_list,
        status=ChainStatus.EXECUTING,
        seals=seal_ids,
    )


def _seal(status: ActionStatus = ActionStatus.EXECUTED, step_index: int = 0) -> Seal:
    intent = IntentDeclaration(
        task_description="test",
        planned_action="test action",
        tool_name="knowledge_search",
        tool_params={},
        resource_targets=["kb:docs"],
        reasoning="test",
    )
    anomaly = AnomalyResult(
        gates=[GateResult(gate_name="scope", verdict=GateVerdict.PASS,
                          score=1.0, threshold=1.0, details="ok")],
        overall_verdict=GateVerdict.PASS,
        risk_level=RiskLevel.LOW,
        persona_uuid="researcher",
        action_fingerprint="abc123",
    )
    return Seal(
        chain_id="chain-1",
        step_index=step_index,
        tenant_id="t1",
        persona_id="researcher",
        intent=intent,
        anomaly_result=anomaly,
        tool_name="knowledge_search",
        tool_params={},
        status=status,
        fingerprint="abc123",
    )


# ── EmbeddingService ──────────────────────────────────────────────────────────

class TestEmbeddingService:

    def _service_with_fake_model(self) -> EmbeddingService:
        """EmbeddingService with a fake model that returns deterministic vectors."""
        svc = EmbeddingService.__new__(EmbeddingService)
        svc.model_name = "fake"

        class _FakeModel:
            def encode(self, texts):
                return np.array(_fake_embed(texts))

        svc._model = _FakeModel()
        return svc

    def test_embed_returns_vectors_for_each_text(self):
        svc = self._service_with_fake_model()
        result = svc.embed(["hello", "world", "test"])
        assert len(result) == 3
        for vec in result:
            assert isinstance(vec, list)
            assert len(vec) > 0
            assert all(isinstance(v, float) for v in vec)

    def test_similarity_same_text_is_high(self):
        svc = self._service_with_fake_model()
        score = svc.similarity("machine learning", "machine learning")
        assert score > 0.99  # identical text → cosine ~1.0

    def test_similarity_different_texts_is_valid_range(self):
        svc = self._service_with_fake_model()
        # Cosine similarity is in [-1.0, 1.0]
        score = svc.similarity("apple fruit", "quantum physics nuclear reactor")
        assert -1.0 <= score <= 1.0

    def test_similarities_returns_one_score_per_candidate(self):
        svc = self._service_with_fake_model()
        scores = svc.similarities("query text", ["candidate one", "candidate two", "candidate three"])
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert all(-1.0 <= s <= 1.0 for s in scores)

    def test_similarities_empty_candidates_returns_empty(self):
        svc = self._service_with_fake_model()
        result = svc.similarities("query", [])
        assert result == []

    def test_lazy_load_not_triggered_before_use(self):
        """Model should not load at construction time."""
        svc = EmbeddingService("all-MiniLM-L6-v2")
        assert svc._model is None  # not yet loaded


# ── KnowledgeStore._chunk_text ────────────────────────────────────────────────

class TestKnowledgeStoreChunking:

    def test_short_text_produces_single_chunk(self):
        chunks = KnowledgeStore._chunk_text("short text")
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_exactly_500_chars_produces_single_chunk(self):
        text = "x" * 500
        chunks = KnowledgeStore._chunk_text(text)
        assert len(chunks) == 1

    def test_501_chars_produces_two_chunks(self):
        text = "x" * 501
        chunks = KnowledgeStore._chunk_text(text)
        assert len(chunks) == 2

    def test_chunks_overlap_by_50(self):
        text = "a" * 200 + "b" * 200 + "c" * 200  # 600 chars
        chunks = KnowledgeStore._chunk_text(text)
        assert len(chunks) == 2
        # Second chunk starts at 500-50=450
        assert len(chunks[1]) == 150  # 600-450

    def test_empty_text_produces_no_chunks(self):
        chunks = KnowledgeStore._chunk_text("")
        assert chunks == []

    def test_all_content_preserved(self):
        """Every character appears in at least one chunk (no data loss)."""
        text = "hello " * 200  # 1200 chars
        chunks = KnowledgeStore._chunk_text(text)
        # First chunk covers [0,500), last chunk ends at 1200
        assert chunks[0][:5] == "hello"
        assert len("".join(c for c in chunks)) >= len(text)  # overlap means slightly more


# ── KnowledgeStore (in-memory ChromaDB) ───────────────────────────────────────

class TestKnowledgeStore:

    @pytest.mark.asyncio
    async def test_ingest_and_query_returns_result(self):
        tid = _tid()
        store = _in_memory_store()
        doc = _doc(tenant_id=tid, content="NEXUS is an AI agent framework with anomaly detection")
        doc_id = await store.ingest(doc)
        assert doc_id == doc.id

        result = await store.query(tid, "docs", "AI agent framework")
        assert isinstance(result, RetrievedContext)
        assert result.namespace == "docs"
        assert len(result.documents) >= 1

    @pytest.mark.asyncio
    async def test_query_empty_store_returns_empty_context(self):
        tid = _tid()
        store = _in_memory_store()
        result = await store.query(tid, "empty_ns", "anything")
        assert result.documents == []
        assert result.confidence == 0.0
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_document_scores_are_between_0_and_1(self):
        tid = _tid()
        store = _in_memory_store()
        await store.ingest(_doc(tenant_id=tid, content="The quick brown fox jumps over the lazy dog"))
        result = await store.query(tid, "docs", "quick fox")
        for doc in result.documents:
            assert 0.0 <= doc["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_is_average_of_scores(self):
        tid = _tid()
        store = _in_memory_store()
        await store.ingest(_doc(tenant_id=tid, content="sentence one about NEXUS security gates"))
        await store.ingest(_doc(tenant_id=tid, content="sentence two about anomaly detection"))
        result = await store.query(tid, "docs", "security")
        if result.documents:
            expected = sum(d["score"] for d in result.documents) / len(result.documents)
            assert abs(result.confidence - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_tenant_isolation(self):
        ta = _tid()
        tb = _tid()
        store = _in_memory_store()
        await store.ingest(_doc(tenant_id=ta, content="secret data for tenant A"))
        # Querying as tenant_b should return nothing (different collection)
        result = await store.query(tb, "docs", "secret data")
        assert result.documents == []

    @pytest.mark.asyncio
    async def test_list_namespaces_after_ingest(self):
        tid = _tid()
        store = _in_memory_store()
        await store.ingest(_doc(tenant_id=tid, namespace="docs"))
        await store.ingest(_doc(tenant_id=tid, namespace="products"))
        namespaces = store.list_namespaces(tid)
        assert "docs" in namespaces
        assert "products" in namespaces

    @pytest.mark.asyncio
    async def test_delete_removes_document(self):
        tid = _tid()
        store = _in_memory_store()
        doc = _doc(tenant_id=tid, content="unique canary phrase xyz987")
        await store.ingest(doc)
        await store.delete(tid, "docs", doc.id)
        result = await store.query(tid, "docs", "unique canary phrase xyz987")
        assert result.documents == []


# ── ContextBuilder ─────────────────────────────────────────────────────────────

class MockKnowledgeStore:
    """Controllable mock — returns canned results, no ChromaDB."""

    def __init__(self, namespaces: list[str] = None, docs: list[dict] = None):
        self._namespaces = namespaces or []
        self._docs = docs or []

    def list_namespaces(self, tenant_id: str) -> list[str]:
        return self._namespaces

    async def query(self, tenant_id, namespace, query, n_results=5, **kwargs) -> RetrievedContext:
        return RetrievedContext(
            query=query,
            documents=list(self._docs),
            confidence=sum(d["score"] for d in self._docs) / len(self._docs) if self._docs else 0.0,
            sources=[d.get("source", "mock") for d in self._docs],
            namespace=namespace,
        )


class TestContextBuilder:

    @pytest.mark.asyncio
    async def test_build_returns_retrieved_context(self):
        store = MockKnowledgeStore()
        builder = ContextBuilder(store)
        result = await builder.build(
            tenant_id="t1",
            task="search for something",
            persona=_researcher(),
        )
        assert isinstance(result, RetrievedContext)
        assert result.query == "search for something"

    @pytest.mark.asyncio
    async def test_kb_wildcard_scope_lists_all_namespaces(self):
        store = MockKnowledgeStore(namespaces=["docs", "products", "support"])
        builder = ContextBuilder(store)
        persona = _researcher()  # has resource_scopes=["kb:*"]
        result = await builder.build(tenant_id="t1", task="find info", persona=persona)
        # Should query all 3 namespaces — result is non-None
        assert isinstance(result, RetrievedContext)

    @pytest.mark.asyncio
    async def test_specific_kb_scope_uses_named_namespace(self):
        store = MockKnowledgeStore(namespaces=["docs", "private"])
        builder = ContextBuilder(store)
        persona = PersonaContract(
            name="restricted",
            description="Limited access",
            allowed_tools=["knowledge_search"],
            resource_scopes=["kb:docs"],  # specific namespace, not wildcard
            intent_patterns=["search"],
            risk_tolerance=RiskLevel.LOW,
        )
        result = await builder.build(tenant_id="t1", task="find docs", persona=persona)
        assert result.namespace == "docs"

    @pytest.mark.asyncio
    async def test_session_history_prepended_to_docs(self):
        store = MockKnowledgeStore()
        builder = ContextBuilder(store)
        history = [
            {"step_index": 0, "action": "step 0", "tool": "x", "result": "previous result text", "status": "executed"}
        ]
        result = await builder.build(
            tenant_id="t1",
            task="next step",
            persona=_researcher(),
            session_history=history,
        )
        # Session history entry should appear as a synthetic doc
        history_docs = [d for d in result.documents if d.get("source") == "session_history"]
        assert len(history_docs) >= 1
        assert "previous result text" in history_docs[0]["content"]

    @pytest.mark.asyncio
    async def test_only_last_3_history_entries_included(self):
        store = MockKnowledgeStore()
        builder = ContextBuilder(store)
        history = [
            {"step_index": i, "action": f"step {i}", "tool": "x",
             "result": f"result {i}", "status": "executed"}
            for i in range(10)
        ]
        result = await builder.build(
            tenant_id="t1", task="task", persona=_researcher(), session_history=history
        )
        history_docs = [d for d in result.documents if d.get("source") == "session_history"]
        assert len(history_docs) <= 3

    @pytest.mark.asyncio
    async def test_documents_sorted_by_score_descending(self):
        docs = [
            {"content": "low score", "score": 0.3, "source": "a", "document_id": "1", "chunk_index": 0, "access_level": "internal"},
            {"content": "high score", "score": 0.9, "source": "b", "document_id": "2", "chunk_index": 0, "access_level": "internal"},
            {"content": "mid score", "score": 0.6, "source": "c", "document_id": "3", "chunk_index": 0, "access_level": "internal"},
        ]
        store = MockKnowledgeStore(docs=docs)
        builder = ContextBuilder(store)
        result = await builder.build(tenant_id="t1", task="query", persona=_researcher())
        kb_docs = [d for d in result.documents if d.get("source") != "session_history"]
        scores = [d["score"] for d in kb_docs]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_confidence_zero_when_no_documents(self):
        store = MockKnowledgeStore(docs=[])
        builder = ContextBuilder(store)
        result = await builder.build(tenant_id="t1", task="query", persona=_researcher())
        assert result.confidence == 0.0


# ── ThinkActGate ───────────────────────────────────────────────────────────────

def _ctx(confidence: float) -> RetrievedContext:
    return RetrievedContext(query="q", documents=[], confidence=confidence, sources=[], namespace="ns")


class TestThinkActGate:

    def test_high_confidence_returns_act(self):
        gate = ThinkActGate(confidence_threshold=0.80)
        assert gate.decide(_ctx(0.90)) == "act"

    def test_exactly_at_threshold_returns_act(self):
        gate = ThinkActGate(confidence_threshold=0.80)
        assert gate.decide(_ctx(0.80)) == "act"

    def test_low_confidence_returns_think(self):
        gate = ThinkActGate(confidence_threshold=0.80)
        assert gate.decide(_ctx(0.50)) == "think"

    def test_zero_confidence_returns_think(self):
        gate = ThinkActGate(confidence_threshold=0.80)
        assert gate.decide(_ctx(0.0)) == "think"

    def test_circuit_breaker_forces_act_at_max_loops(self):
        gate = ThinkActGate(confidence_threshold=0.80, max_think_loops=3)
        # Even with low confidence, loop_count == max → ACT
        assert gate.decide(_ctx(0.0), loop_count=3) == "act"

    def test_circuit_breaker_not_triggered_before_max(self):
        gate = ThinkActGate(confidence_threshold=0.80, max_think_loops=3)
        assert gate.decide(_ctx(0.0), loop_count=2) == "think"

    def test_custom_threshold_respected(self):
        gate = ThinkActGate(confidence_threshold=0.5)
        assert gate.decide(_ctx(0.51)) == "act"
        assert gate.decide(_ctx(0.49)) == "think"


# ── ContinueCompleteGate ───────────────────────────────────────────────────────

class TestContinueCompleteGate:

    def test_more_steps_returns_continue(self):
        gate = ContinueCompleteGate()
        # 2-step chain, only 1 seal done → CONTINUE
        chain = _chain(steps=2, seals_done=1)
        seal = _seal(ActionStatus.EXECUTED)
        assert gate.decide(chain, "result", seal) == "continue"

    def test_all_steps_done_returns_complete(self):
        gate = ContinueCompleteGate()
        chain = _chain(steps=1, seals_done=1)
        seal = _seal(ActionStatus.EXECUTED)
        assert gate.decide(chain, "result", seal) == "complete"

    def test_failed_seal_returns_retry_on_first_failure(self):
        gate = ContinueCompleteGate()
        chain = _chain(steps=1, seals_done=0)
        seal = _seal(ActionStatus.FAILED)
        assert gate.decide(chain, None, seal) == "retry"

    def test_failed_seal_returns_escalate_after_max_retries(self):
        gate = ContinueCompleteGate()
        chain = _chain(steps=1, seals_done=0)
        seal = _seal(ActionStatus.FAILED)
        gate.decide(chain, None, seal)  # first → RETRY
        gate.decide(chain, None, seal)  # second → RETRY
        result = gate.decide(chain, None, seal)  # third → ESCALATE
        assert result == "escalate"

    def test_independent_chains_have_separate_retry_counts(self):
        gate = ContinueCompleteGate()
        chain_a = _chain(steps=1, seals_done=0)
        chain_b = _chain(steps=1, seals_done=0)
        seal_a = _seal(ActionStatus.FAILED, step_index=0)
        seal_b = _seal(ActionStatus.FAILED, step_index=0)
        # Exhaust retries for chain_a
        gate.decide(chain_a, None, seal_a)
        gate.decide(chain_a, None, seal_a)
        assert gate.decide(chain_a, None, seal_a) == "escalate"
        # chain_b should still have fresh retries
        assert gate.decide(chain_b, None, seal_b) == "retry"


# ── EscalateGate ──────────────────────────────────────────────────────────────

class TestEscalateGate:

    def test_timeout_error_returns_retry(self):
        gate = EscalateGate()
        chain = _chain()
        assert gate.decide(TimeoutError("connection timed out"), 0, chain) == "retry"

    def test_rate_limit_keyword_returns_retry(self):
        gate = EscalateGate()
        chain = _chain()
        assert gate.decide(Exception("rate limit exceeded"), 0, chain) == "retry"

    def test_tool_error_returns_retry(self):
        gate = EscalateGate()
        chain = _chain()
        err = ToolError("bad params", tool_name="knowledge_search")
        assert gate.decide(err, 0, chain) == "retry"

    def test_unknown_error_returns_escalate(self):
        gate = EscalateGate()
        chain = _chain()
        assert gate.decide(ValueError("unexpected error"), 0, chain) == "escalate"

    def test_max_retries_reached_returns_escalate(self):
        gate = EscalateGate()
        chain = _chain()
        # Even a transient error escalates after max retries
        assert gate.decide(TimeoutError("timeout"), 2, chain) == "escalate"

    def test_build_escalation_context_structure(self):
        gate = EscalateGate()
        chain = _chain(steps=3, seals_done=1)
        ctx = gate.build_escalation_context(chain, ValueError("something broke"))
        assert ctx["chain_id"] == chain.id
        assert ctx["task"] == chain.task
        assert ctx["progress"]["steps_completed"] == 1
        assert ctx["progress"]["steps_total"] == 3
        assert ctx["progress"]["completion_pct"] == pytest.approx(33.3, rel=0.1)
        assert ctx["error"]["type"] == "ValueError"
        assert ctx["error"]["message"] == "something broke"
        assert "recommendation" in ctx
        assert "escalated_at" in ctx

    def test_build_escalation_context_tool_error_recommendation(self):
        gate = EscalateGate()
        chain = _chain()
        err = ToolError("failed", tool_name="knowledge_search")
        ctx = gate.build_escalation_context(chain, err)
        assert "knowledge_search" in ctx["recommendation"]

    def test_build_escalation_context_transient_recommendation(self):
        gate = EscalateGate()
        chain = _chain()
        ctx = gate.build_escalation_context(chain, Exception("rate limit exceeded"))
        assert "transient" in ctx["recommendation"].lower() or "retry" in ctx["recommendation"].lower()
