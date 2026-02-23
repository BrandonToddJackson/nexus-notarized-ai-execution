"""
Comprehensive smoketest: Phase 1–9 public APIs with real assertions.

Each TestPhaseN class exercises the public surface of that phase end-to-end.
All assertions are real — no skips, no stubs.

Phase coverage:
  Phase 1  — Core Security   : personas, anomaly (4 gates), notary, ledger, chain, verifier, output_validator, cot_logger
  Phase 2  — Core Cognitive  : embeddings, store, context, think_act, continue_complete, escalate
  Phase 3  — Execution Layer : registry, plugin decorator, sandbox, executor, tool selector
  Phase 4  — Engine          : NexusEngine end-to-end (decompose → gate → seal → ledger)
  Phase 5  — Persistence     : Repository + seed via SQLite
  Phase 6  — LLM             : CostTracker + LLMClient (mocked litellm)
  Phase 7  — Cache           : FingerprintCache (mocked Redis)
  Phase 8  — Auth            : JWTManager, RateLimiter, AuthMiddleware
  Phase 9  — API             : FastAPI routes via TestClient
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Shared test data ──────────────────────────────────────────────────────────

TENANT = "smoke-tenant"
CHAIN_ID = "smoke-chain-001"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Core Security
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase1Personas:
    """PersonaManager: load, activate, validate, revoke."""

    def _manager(self):
        from nexus.core.personas import PersonaManager
        from nexus.types import PersonaContract, RiskLevel
        p = PersonaContract(
            name="researcher",
            description="Searches information",
            allowed_tools=["knowledge_search", "web_search"],
            resource_scopes=["kb:*", "web:*"],
            intent_patterns=["search for information", "look up"],
            risk_tolerance=RiskLevel.LOW,
            max_ttl_seconds=60,
        )
        return PersonaManager([p])

    def test_get_loaded_persona(self):
        pm = self._manager()
        contract = pm.get_persona("researcher")
        assert contract is not None
        assert contract.name == "researcher"

    def test_missing_persona_returns_none(self):
        pm = self._manager()
        assert pm.get_persona("nonexistent") is None

    def test_activate_sets_timestamp(self):
        pm = self._manager()
        contract = pm.activate("researcher", TENANT)
        assert contract.name == "researcher"
        ts = pm.get_activation_time("researcher")
        assert isinstance(ts, datetime)

    def test_revoke_clears_active(self):
        pm = self._manager()
        pm.activate("researcher", TENANT)
        pm.revoke("researcher")
        assert pm.get_activation_time("researcher") is None

    def test_list_personas_returns_all(self):
        pm = self._manager()
        personas = pm.list_personas()
        assert len(personas) == 1
        assert personas[0].name == "researcher"

    def test_scope_check_passes_for_allowed_resource(self):
        pm = self._manager()
        contract = pm.get_persona("researcher")
        # validate_action returns True when resource is in scope
        result = pm.validate_action(contract, "knowledge_search", ["kb:product_docs"])
        assert result is True

    def test_scope_check_fails_for_restricted_resource(self):
        from nexus.exceptions import PersonaViolation
        pm = self._manager()
        contract = pm.get_persona("researcher")
        # validate_action raises PersonaViolation when resource is out of scope
        with pytest.raises(PersonaViolation):
            pm.validate_action(contract, "knowledge_search", ["email:user@example.com"])


class TestPhase1AnomalyGates:
    """AnomalyEngine: all 4 gates checked with real assertions."""

    def _build(self):
        from nexus.core.anomaly import AnomalyEngine
        from nexus.config import NexusConfig
        return AnomalyEngine(config=NexusConfig())

    def _persona(self, tools=None, scopes=None):
        from nexus.types import PersonaContract, RiskLevel
        return PersonaContract(
            name="researcher",
            description="test",
            allowed_tools=tools or ["knowledge_search"],
            resource_scopes=scopes or ["kb:*"],
            intent_patterns=["search for information"],
            risk_tolerance=RiskLevel.LOW,
            max_ttl_seconds=60,
        )

    def _intent(self, tool="knowledge_search", targets=None):
        from nexus.types import IntentDeclaration
        return IntentDeclaration(
            task_description="search for information about NEXUS",
            planned_action="search for information",
            tool_name=tool,
            tool_params={"query": "NEXUS docs"},
            resource_targets=targets or ["kb:product_docs"],
            reasoning="found in knowledge base",
        )

    @pytest.mark.asyncio
    async def test_gate1_allowed_tool_passes(self):
        engine = self._build()
        result = await engine.check(
            persona=self._persona(),
            intent=self._intent(),
            activation_time=datetime.now(timezone.utc),
        )
        scope_gate = next(g for g in result.gates if g.gate_name == "scope")
        assert scope_gate.verdict.value == "pass"

    @pytest.mark.asyncio
    async def test_gate1_blocked_tool_fails(self):
        engine = self._build()
        result = await engine.check(
            persona=self._persona(tools=["knowledge_search"]),
            intent=self._intent(tool="send_email"),
            activation_time=datetime.now(timezone.utc),
        )
        scope_gate = next(g for g in result.gates if g.gate_name == "scope")
        assert scope_gate.verdict.value == "fail"

    @pytest.mark.asyncio
    async def test_gate3_fresh_activation_passes(self):
        engine = self._build()
        result = await engine.check(
            persona=self._persona(),
            intent=self._intent(),
            activation_time=datetime.now(timezone.utc),
        )
        ttl_gate = next(g for g in result.gates if g.gate_name == "ttl")
        assert ttl_gate.verdict.value == "pass"

    @pytest.mark.asyncio
    async def test_overall_verdict_fail_if_any_gate_fails(self):
        engine = self._build()
        result = await engine.check(
            persona=self._persona(tools=["knowledge_search"]),
            intent=self._intent(tool="send_email"),
            activation_time=datetime.now(timezone.utc),
        )
        from nexus.types import GateVerdict
        assert result.overall_verdict == GateVerdict.FAIL

    @pytest.mark.asyncio
    async def test_all_gates_present_in_result(self):
        engine = self._build()
        result = await engine.check(
            persona=self._persona(),
            intent=self._intent(),
            activation_time=datetime.now(timezone.utc),
        )
        gate_names = {g.gate_name for g in result.gates}
        assert "scope" in gate_names
        assert "intent" in gate_names
        assert "ttl" in gate_names
        assert "drift" in gate_names


class TestPhase1Notary:
    """Notary: seal creation, finalization, Merkle chain integrity."""

    def _notary(self):
        from nexus.core.notary import Notary
        return Notary()

    def _intent(self):
        from nexus.types import IntentDeclaration
        return IntentDeclaration(
            task_description="search",
            planned_action="search for information",
            tool_name="knowledge_search",
            tool_params={"query": "test"},
            resource_targets=["kb:docs"],
            reasoning="user requested search",
        )

    def _anomaly_result(self):
        from nexus.types import AnomalyResult, GateResult, GateVerdict, RiskLevel
        gate = GateResult(
            gate_name="scope", verdict=GateVerdict.PASS, score=1.0, threshold=1.0, details="ok"
        )
        return AnomalyResult(
            gates=[gate, gate, gate, gate],
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_id="researcher",
            action_fingerprint="fp123",
        )

    def test_create_seal_returns_pending_seal(self):
        from nexus.types import ActionStatus
        notary = self._notary()
        seal = notary.create_seal(
            chain_id=CHAIN_ID, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=self._intent(),
            anomaly_result=self._anomaly_result(),
        )
        assert seal.status == ActionStatus.PENDING
        assert seal.fingerprint != ""
        assert seal.chain_id == CHAIN_ID

    def test_finalize_seal_sets_executed_status(self):
        from nexus.types import ActionStatus
        notary = self._notary()
        seal = notary.create_seal(
            chain_id=CHAIN_ID, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=self._intent(),
            anomaly_result=self._anomaly_result(),
        )
        final = notary.finalize_seal(seal, {"result": "data"}, ActionStatus.EXECUTED)
        assert final.status == ActionStatus.EXECUTED
        assert final.tool_result == {"result": "data"}
        assert final.completed_at is not None

    def test_merkle_fingerprints_chain(self):
        notary = self._notary()
        seal0 = notary.create_seal(
            chain_id=CHAIN_ID, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=self._intent(),
            anomaly_result=self._anomaly_result(),
        )
        seal1 = notary.create_seal(
            chain_id=CHAIN_ID, step_index=1, tenant_id=TENANT,
            persona_id="researcher", intent=self._intent(),
            anomaly_result=self._anomaly_result(),
        )
        # seal1's parent_fingerprint must equal seal0's fingerprint
        assert seal1.parent_fingerprint == seal0.fingerprint
        assert seal0.fingerprint != seal1.fingerprint

    def test_verify_chain_passes_for_valid_seals(self):
        from nexus.types import ActionStatus
        notary = self._notary()
        seal0 = notary.create_seal(
            chain_id=CHAIN_ID, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=self._intent(),
            anomaly_result=self._anomaly_result(),
        )
        seal0 = notary.finalize_seal(seal0, "result0", ActionStatus.EXECUTED)
        seal1 = notary.create_seal(
            chain_id=CHAIN_ID, step_index=1, tenant_id=TENANT,
            persona_id="researcher", intent=self._intent(),
            anomaly_result=self._anomaly_result(),
        )
        seal1 = notary.finalize_seal(seal1, "result1", ActionStatus.EXECUTED)
        assert notary.verify_chain([seal0, seal1]) is True

    def test_verify_chain_fails_after_tampering(self):
        from nexus.types import ActionStatus
        from nexus.exceptions import SealIntegrityError
        notary = self._notary()
        seal0 = notary.create_seal(
            chain_id=CHAIN_ID, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=self._intent(),
            anomaly_result=self._anomaly_result(),
        )
        seal0 = notary.finalize_seal(seal0, "result0", ActionStatus.EXECUTED)
        tampered = seal0.model_copy(update={"fingerprint": "TAMPERED"})
        # verify_chain raises SealIntegrityError when chain is broken
        with pytest.raises(SealIntegrityError):
            notary.verify_chain([tampered])


class TestPhase1Ledger:
    """Ledger: append, get_chain, get_by_tenant, in-memory."""

    def _make_seal(self, chain_id=CHAIN_ID, step_index=0, tenant_id=TENANT):
        from nexus.types import (
            Seal, IntentDeclaration, AnomalyResult, GateResult, GateVerdict,
            RiskLevel, ActionStatus,
        )
        intent = IntentDeclaration(
            task_description="test",
            planned_action="search for information",
            tool_name="knowledge_search",
            tool_params={},
            resource_targets=[],
            reasoning="",
        )
        gate = GateResult(
            gate_name="scope", verdict=GateVerdict.PASS, score=1.0, threshold=1.0, details=""
        )
        anomaly = AnomalyResult(
            gates=[gate, gate, gate, gate],
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_id="researcher",
            action_fingerprint="fp",
        )
        return Seal(
            chain_id=chain_id, step_index=step_index, tenant_id=tenant_id,
            persona_id="researcher", intent=intent, anomaly_result=anomaly,
            tool_name="knowledge_search", tool_params={},
            status=ActionStatus.EXECUTED,
        )

    @pytest.mark.asyncio
    async def test_append_and_get_chain(self):
        from nexus.core.ledger import Ledger
        ledger = Ledger()
        seal = self._make_seal()
        await ledger.append(seal)
        chain_seals = await ledger.get_chain(CHAIN_ID)
        assert len(chain_seals) == 1
        assert chain_seals[0].id == seal.id

    @pytest.mark.asyncio
    async def test_get_by_tenant_filters_correctly(self):
        from nexus.core.ledger import Ledger
        ledger = Ledger()
        s1 = self._make_seal(tenant_id="tenant-A")
        s2 = self._make_seal(tenant_id="tenant-B")
        await ledger.append(s1)
        await ledger.append(s2)
        results = await ledger.get_by_tenant("tenant-A")
        assert len(results) == 1
        assert results[0].id == s1.id

    @pytest.mark.asyncio
    async def test_get_chain_ordered_by_step(self):
        from nexus.core.ledger import Ledger
        ledger = Ledger()
        s0 = self._make_seal(step_index=0)
        s1 = self._make_seal(step_index=1)
        await ledger.append(s1)  # append out of order
        await ledger.append(s0)
        chain = await ledger.get_chain(CHAIN_ID)
        assert chain[0].step_index == 0
        assert chain[1].step_index == 1


class TestPhase1ChainManager:
    """ChainManager: create, advance, fail, escalate."""

    def _cm(self):
        from nexus.core.chain import ChainManager
        return ChainManager()

    def test_create_chain_has_planning_status(self):
        from nexus.types import ChainStatus
        cm = self._cm()
        chain = cm.create_chain(TENANT, "search for info", [{"action": "step1"}])
        assert chain.status == ChainStatus.PLANNING
        assert chain.tenant_id == TENANT
        assert len(chain.steps) == 1

    def test_advance_appends_seal_id(self):
        cm = self._cm()
        chain = cm.create_chain(TENANT, "task", [{"action": "step1"}, {"action": "step2"}])
        chain = cm.advance(chain, "seal-001")
        assert "seal-001" in chain.seals

    def test_fail_sets_failed_status(self):
        from nexus.types import ChainStatus
        cm = self._cm()
        chain = cm.create_chain(TENANT, "task", [{"action": "s"}])
        chain = cm.fail(chain, "something went wrong")
        assert chain.status == ChainStatus.FAILED
        assert "something went wrong" in chain.error

    def test_escalate_sets_escalated_status(self):
        from nexus.types import ChainStatus
        cm = self._cm()
        chain = cm.create_chain(TENANT, "task", [{"action": "s"}])
        chain = cm.escalate(chain, "human needed")
        assert chain.status == ChainStatus.ESCALATED

    def test_is_complete_true_when_all_steps_done(self):
        cm = self._cm()
        chain = cm.create_chain(TENANT, "task", [{"action": "s1"}])
        chain = cm.advance(chain, "seal-001")
        assert cm.is_complete(chain) is True


class TestPhase1IntentVerifier:
    """IntentVerifier: cross-check intent against actual execution."""

    def _intent(self, tool="knowledge_search"):
        from nexus.types import IntentDeclaration
        return IntentDeclaration(
            task_description="find info",
            planned_action="search for information",
            tool_name=tool,
            tool_params={"query": "test"},
            resource_targets=["kb:docs"],
            reasoning="test",
        )

    def test_matching_tool_verifies_true(self):
        from nexus.core.verifier import IntentVerifier
        v = IntentVerifier()
        assert v.verify(self._intent(), "knowledge_search", {"query": "test"}) is True

    def test_mismatched_tool_verifies_false(self):
        from nexus.core.verifier import IntentVerifier
        from nexus.exceptions import PersonaViolation
        v = IntentVerifier()
        # verify raises PersonaViolation when tool names mismatch
        with pytest.raises(PersonaViolation):
            v.verify(self._intent(), "send_email", {"query": "test"})


class TestPhase1OutputValidator:
    """OutputValidator: validates tool results."""

    def _intent(self):
        from nexus.types import IntentDeclaration
        return IntentDeclaration(
            task_description="search",
            planned_action="search for information",
            tool_name="knowledge_search",
            tool_params={},
            resource_targets=[],
            reasoning="",
        )

    @pytest.mark.asyncio
    async def test_non_none_result_is_valid(self):
        from nexus.core.output_validator import OutputValidator
        ov = OutputValidator()
        is_valid, reason = await ov.validate(self._intent(), "some result text")
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_none_result_is_invalid(self):
        from nexus.core.output_validator import OutputValidator
        ov = OutputValidator()
        is_valid, reason = await ov.validate(self._intent(), None)
        assert is_valid is False
        assert reason  # non-empty explanation


class TestPhase1CotLogger:
    """CoTLogger: log, retrieve, clear."""

    def test_log_and_get_trace(self):
        from nexus.core.cot_logger import CoTLogger
        logger = CoTLogger()
        logger.log("seal-001", "Context built")
        logger.log("seal-001", "Tool selected: knowledge_search")
        trace = logger.get_trace("seal-001")
        assert len(trace) == 2
        assert "Context built" in trace[0]
        assert "knowledge_search" in trace[1]

    def test_clear_removes_entries(self):
        from nexus.core.cot_logger import CoTLogger
        logger = CoTLogger()
        logger.log("seal-001", "step A")
        logger.clear("seal-001")
        assert logger.get_trace("seal-001") == []

    def test_different_keys_are_isolated(self):
        from nexus.core.cot_logger import CoTLogger
        logger = CoTLogger()
        logger.log("key-A", "for A")
        logger.log("key-B", "for B")
        assert logger.get_trace("key-A") == ["for A"]
        assert logger.get_trace("key-B") == ["for B"]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Core Cognitive
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2EmbeddingService:
    """EmbeddingService: embed, similarities."""

    def test_embed_returns_correct_shape(self):
        from nexus.knowledge.embeddings import EmbeddingService
        svc = EmbeddingService()
        vecs = svc.embed(["hello world", "test sentence"])
        assert len(vecs) == 2
        assert all(isinstance(v, list) for v in vecs)
        assert len(vecs[0]) == 384  # all-MiniLM-L6-v2

    def test_same_text_has_high_similarity(self):
        from nexus.knowledge.embeddings import EmbeddingService
        svc = EmbeddingService()
        scores = svc.similarities("the quick brown fox", ["the quick brown fox", "something else"])
        assert scores[0] > 0.99
        assert scores[1] < scores[0]

    def test_similarities_empty_candidates(self):
        from nexus.knowledge.embeddings import EmbeddingService
        svc = EmbeddingService()
        assert svc.similarities("query", []) == []


class TestPhase2KnowledgeStore:
    """KnowledgeStore: ingest + query with real ChromaDB."""

    def _store(self, tmp_path):
        from nexus.knowledge.store import KnowledgeStore
        from nexus.knowledge.embeddings import EmbeddingService
        svc = EmbeddingService()
        return KnowledgeStore(persist_dir=str(tmp_path), embedding_fn=svc.embed)

    @pytest.mark.asyncio
    async def test_ingest_and_query_returns_results(self, tmp_path):
        from nexus.types import KnowledgeDocument
        store = self._store(tmp_path)
        doc = KnowledgeDocument(
            tenant_id=TENANT, namespace="default", source="test.txt",
            content="NEXUS is an AI agent framework with notarized actions.",
        )
        await store.ingest(doc)
        ctx = await store.query(TENANT, "default", "notarized AI agent", n_results=1)
        assert len(ctx.documents) > 0
        assert ctx.confidence > 0.0
        assert ctx.documents[0]["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_tenant_isolation_prevents_cross_access(self, tmp_path):
        from nexus.types import KnowledgeDocument
        store = self._store(tmp_path)
        doc = KnowledgeDocument(
            tenant_id="tenant-secret", namespace="default", source="secret.txt",
            content="Top secret data that only tenant-secret can see.",
        )
        await store.ingest(doc)
        ctx = await store.query("other-tenant", "default", "secret data", n_results=5)
        assert len(ctx.documents) == 0


class TestPhase2ThinkActGate:
    """ThinkActGate: confidence-based routing."""

    def _ctx(self, confidence):
        from nexus.types import RetrievedContext
        return RetrievedContext(
            query="test", documents=[], confidence=confidence,
            sources=[], namespace="default",
        )

    def test_high_confidence_returns_act(self):
        from nexus.reasoning.think_act import ThinkActGate
        from nexus.types import ReasoningDecision
        gate = ThinkActGate(confidence_threshold=0.80)
        assert gate.decide(self._ctx(0.95), 0) == ReasoningDecision.ACT

    def test_low_confidence_returns_think(self):
        from nexus.reasoning.think_act import ThinkActGate
        from nexus.types import ReasoningDecision
        gate = ThinkActGate(confidence_threshold=0.80)
        assert gate.decide(self._ctx(0.4), 0) == ReasoningDecision.THINK

    def test_max_loops_forces_act(self):
        from nexus.reasoning.think_act import ThinkActGate
        from nexus.types import ReasoningDecision
        gate = ThinkActGate(confidence_threshold=0.80, max_think_loops=3)
        assert gate.decide(self._ctx(0.0), 3) == ReasoningDecision.ACT


class TestPhase2ContinueCompleteGate:
    """ContinueCompleteGate: decides whether to continue chain or stop."""

    def _chain(self, n_steps=2, n_seals=0):
        from nexus.types import ChainPlan, ChainStatus
        return ChainPlan(
            tenant_id=TENANT, task="task",
            steps=[{"action": f"step{i}"} for i in range(n_steps)],
            seals=[f"seal-{i}" for i in range(n_seals)],
            status=ChainStatus.EXECUTING,
        )

    def _seal(self, status="executed"):
        from nexus.types import (
            Seal, IntentDeclaration, AnomalyResult, GateResult, GateVerdict,
            RiskLevel, ActionStatus,
        )
        intent = IntentDeclaration(
            task_description="t", planned_action="p", tool_name="t",
            tool_params={}, resource_targets=[], reasoning="",
        )
        gate = GateResult(
            gate_name="scope", verdict=GateVerdict.PASS, score=1.0, threshold=1.0, details=""
        )
        anomaly = AnomalyResult(
            gates=[gate, gate, gate, gate],
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_id="researcher",
            action_fingerprint="fp",
        )
        s = ActionStatus(status)
        return Seal(
            chain_id=CHAIN_ID, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=intent, anomaly_result=anomaly,
            tool_name="knowledge_search", tool_params={}, status=s,
        )

    def test_more_steps_returns_continue(self):
        from nexus.reasoning.continue_complete import ContinueCompleteGate
        from nexus.types import ReasoningDecision
        gate = ContinueCompleteGate()
        chain = self._chain(n_steps=2, n_seals=0)
        decision = gate.decide(chain, "some result", self._seal("executed"))
        assert decision == ReasoningDecision.CONTINUE

    def test_all_steps_done_returns_complete(self):
        from nexus.reasoning.continue_complete import ContinueCompleteGate
        from nexus.types import ReasoningDecision
        gate = ContinueCompleteGate()
        chain = self._chain(n_steps=1, n_seals=1)
        decision = gate.decide(chain, "result", self._seal("executed"))
        assert decision == ReasoningDecision.COMPLETE


class TestPhase2EscalateGate:
    """EscalateGate: retry vs escalate logic."""

    def _chain(self):
        from nexus.types import ChainPlan, ChainStatus
        return ChainPlan(
            tenant_id=TENANT, task="task",
            steps=[{"action": "s"}], seals=[], status=ChainStatus.EXECUTING,
        )

    def test_timeout_error_returns_retry(self):
        from nexus.reasoning.escalate import EscalateGate
        from nexus.types import ReasoningDecision
        gate = EscalateGate()
        decision = gate.decide(TimeoutError("timed out"), 0, self._chain())
        assert decision == ReasoningDecision.RETRY

    def test_unknown_error_returns_escalate(self):
        from nexus.reasoning.escalate import EscalateGate
        from nexus.types import ReasoningDecision
        gate = EscalateGate()
        decision = gate.decide(ValueError("bad input"), 0, self._chain())
        assert decision == ReasoningDecision.ESCALATE

    def test_max_retries_exceeded_returns_escalate(self):
        from nexus.reasoning.escalate import EscalateGate
        from nexus.types import ReasoningDecision
        gate = EscalateGate()
        # default max_retries is 2; retry_count >= max_retries → escalate
        decision = gate.decide(TimeoutError("retry again"), 2, self._chain())
        assert decision == ReasoningDecision.ESCALATE

    def test_build_escalation_context_structure(self):
        from nexus.reasoning.escalate import EscalateGate
        gate = EscalateGate()
        ctx = gate.build_escalation_context(self._chain(), RuntimeError("boom"))
        assert "chain_id" in ctx
        assert "error" in ctx
        assert "recommendation" in ctx


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Execution Layer
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase3ToolRegistry:
    """ToolRegistry: register, retrieve, filter by persona."""

    def _registry(self):
        from nexus.tools.registry import ToolRegistry
        from nexus.types import ToolDefinition, RiskLevel
        reg = ToolRegistry()
        defn = ToolDefinition(
            name="knowledge_search",
            description="Search the knowledge base",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            risk_level=RiskLevel.LOW,
            resource_pattern="kb:*",
        )

        async def impl(query: str):
            return f"Results for: {query}"

        reg.register(defn, impl)
        return reg

    def test_register_and_list(self):
        reg = self._registry()
        tools = reg.list_tools()
        assert any(t.name == "knowledge_search" for t in tools)

    def test_get_known_tool(self):
        reg = self._registry()
        defn, impl = reg.get("knowledge_search")
        assert defn.name == "knowledge_search"
        assert callable(impl)

    def test_get_unknown_raises_tool_error(self):
        from nexus.exceptions import ToolError
        reg = self._registry()
        with pytest.raises(ToolError):
            reg.get("nonexistent_tool")

    def test_filter_by_persona(self):
        from nexus.types import PersonaContract, RiskLevel
        reg = self._registry()
        persona = PersonaContract(
            name="researcher",
            description="",
            allowed_tools=["knowledge_search"],
            resource_scopes=["kb:*"],
            intent_patterns=[],
            risk_tolerance=RiskLevel.LOW,
        )
        filtered = reg.list_for_persona(persona)
        assert len(filtered) == 1
        assert filtered[0].name == "knowledge_search"


class TestPhase3PluginDecorator:
    """@tool decorator: auto-registers ToolDefinition from signature."""

    def test_decorated_fn_has_nexus_tool_attr(self):
        from nexus.tools.plugin import tool, _registered_tools
        from nexus.types import RiskLevel

        @tool(name="smoke_test_tool", description="Smoke test tool", risk_level=RiskLevel.LOW)
        async def my_tool(query: str) -> str:
            """Search something."""
            return f"result: {query}"

        assert hasattr(my_tool, "_nexus_tool")
        assert my_tool._nexus_tool.name == "smoke_test_tool"
        assert "smoke_test_tool" in _registered_tools

    @pytest.mark.asyncio
    async def test_decorated_fn_still_callable(self):
        from nexus.tools.plugin import tool
        from nexus.types import RiskLevel

        @tool(name="callable_test", description="callable", risk_level=RiskLevel.LOW)
        async def my_fn(x: str) -> str:
            return f"ok:{x}"

        result = await my_fn("hello")
        assert result == "ok:hello"


class TestPhase3Sandbox:
    """Sandbox: runs tools with timeout enforcement."""

    @pytest.mark.asyncio
    async def test_successful_execution_returns_result(self):
        from nexus.tools.sandbox import Sandbox

        async def echo(query: str):
            return f"echo:{query}"

        sandbox = Sandbox()
        result = await sandbox.execute(echo, {"query": "hello"})
        assert result == "echo:hello"

    @pytest.mark.asyncio
    async def test_exception_is_wrapped_as_tool_error(self):
        from nexus.tools.sandbox import Sandbox
        from nexus.exceptions import ToolError

        async def boom(**kwargs):
            raise ValueError("something broke")

        sandbox = Sandbox()
        with pytest.raises(ToolError):
            await sandbox.execute(boom, {})

    @pytest.mark.asyncio
    async def test_timeout_raises_tool_error(self):
        from nexus.tools.sandbox import Sandbox
        from nexus.exceptions import ToolError

        async def slow(**kwargs):
            await asyncio.sleep(10)
            return "done"

        sandbox = Sandbox()
        with pytest.raises(ToolError):
            await sandbox.execute(slow, {}, timeout=0.01)


class TestPhase3ToolExecutor:
    """ToolExecutor: intent verify → sandbox → result."""

    def _setup(self):
        from nexus.tools.registry import ToolRegistry
        from nexus.tools.sandbox import Sandbox
        from nexus.core.verifier import IntentVerifier
        from nexus.tools.executor import ToolExecutor
        from nexus.types import ToolDefinition, RiskLevel

        reg = ToolRegistry()

        async def search(query: str):
            return f"found:{query}"

        defn = ToolDefinition(
            name="knowledge_search",
            description="Search knowledge",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            risk_level=RiskLevel.LOW,
            resource_pattern="kb:*",
        )
        reg.register(defn, search)
        sandbox = Sandbox()
        verifier = IntentVerifier()
        executor = ToolExecutor(registry=reg, sandbox=sandbox, verifier=verifier)
        return executor

    def _intent(self, tool="knowledge_search"):
        from nexus.types import IntentDeclaration
        return IntentDeclaration(
            task_description="search docs",
            planned_action="search for information",
            tool_name=tool,
            tool_params={"query": "NEXUS"},
            resource_targets=["kb:docs"],
            reasoning="test",
        )

    @pytest.mark.asyncio
    async def test_execute_returns_result(self):
        executor = self._setup()
        result, error = await executor.execute(self._intent())
        assert result == "found:NEXUS"
        assert error is None

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        executor = self._setup()
        result, error = await executor.execute(self._intent(tool="nonexistent"))
        assert error is not None
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase4Engine:
    """NexusEngine: end-to-end task execution without LLM (rule-based fallback)."""

    def _build_engine(self, personas=None):
        from nexus.config import NexusConfig
        from nexus.core.engine import NexusEngine
        from nexus.core.personas import PersonaManager
        from nexus.core.anomaly import AnomalyEngine
        from nexus.core.notary import Notary
        from nexus.core.ledger import Ledger
        from nexus.core.chain import ChainManager
        from nexus.core.verifier import IntentVerifier
        from nexus.core.output_validator import OutputValidator
        from nexus.core.cot_logger import CoTLogger
        from nexus.knowledge.store import KnowledgeStore
        from nexus.knowledge.context import ContextBuilder
        from nexus.tools.registry import ToolRegistry
        from nexus.tools.selector import ToolSelector
        from nexus.tools.sandbox import Sandbox
        from nexus.tools.executor import ToolExecutor
        from nexus.reasoning.think_act import ThinkActGate
        from nexus.reasoning.continue_complete import ContinueCompleteGate
        from nexus.reasoning.escalate import EscalateGate
        from nexus.types import PersonaContract, RiskLevel, ToolDefinition

        if personas is None:
            personas = [PersonaContract(
                name="researcher",
                description="Searches information",
                allowed_tools=["knowledge_search"],
                resource_scopes=["kb:*"],
                intent_patterns=["search for information"],
                risk_tolerance=RiskLevel.LOW,
                max_ttl_seconds=60,
            )]

        reg = ToolRegistry()

        async def knowledge_search(query: str):
            return f"Knowledge result for: {query}"

        async def send_email(to: str, subject: str, body: str):
            return f"Email sent to {to}"

        reg.register(ToolDefinition(
            name="knowledge_search", description="Search KB",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            risk_level=RiskLevel.LOW, resource_pattern="kb:*",
        ), knowledge_search)
        reg.register(ToolDefinition(
            name="send_email", description="Send email",
            parameters={"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to"]},
            risk_level=RiskLevel.HIGH, resource_pattern="email:*",
        ), send_email)

        cfg = NexusConfig(
            database_url="sqlite+aiosqlite:///test.db",
            redis_url="redis://localhost:6379/15",
        )
        pm = PersonaManager(personas)
        anomaly = AnomalyEngine(config=cfg)
        store = KnowledgeStore(persist_dir="/tmp/nexus_smoke_chroma")
        context_builder = ContextBuilder(knowledge_store=store)
        selector = ToolSelector(registry=reg)
        sandbox = Sandbox()
        verifier = IntentVerifier()
        executor = ToolExecutor(registry=reg, sandbox=sandbox, verifier=verifier)

        return NexusEngine(
            persona_manager=pm,
            anomaly_engine=anomaly,
            notary=Notary(),
            ledger=Ledger(),
            chain_manager=ChainManager(),
            context_builder=context_builder,
            tool_registry=reg,
            tool_selector=selector,
            tool_executor=executor,
            output_validator=OutputValidator(),
            cot_logger=CoTLogger(),
            think_act_gate=ThinkActGate(),
            continue_complete_gate=ContinueCompleteGate(),
            escalate_gate=EscalateGate(),
            config=cfg,
        )

    @pytest.mark.asyncio
    async def test_single_step_produces_one_seal(self):
        from nexus.types import ChainStatus
        engine = self._build_engine()
        chain = await engine.run("search for information about NEXUS", TENANT)
        assert len(chain.seals) == 1
        assert chain.status == ChainStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_seal_has_all_4_gate_results(self):
        engine = self._build_engine()
        chain = await engine.run("search for information about NEXUS", TENANT)
        ledger = engine.ledger
        seals = await ledger.get_chain(chain.id)
        assert len(seals) == 1
        assert len(seals[0].anomaly_result.gates) == 4

    @pytest.mark.asyncio
    async def test_seal_has_merkle_fingerprint(self):
        engine = self._build_engine()
        chain = await engine.run("search for information about NEXUS", TENANT)
        seals = await engine.ledger.get_chain(chain.id)
        assert seals[0].fingerprint != ""

    @pytest.mark.asyncio
    async def test_seal_has_cot_trace(self):
        engine = self._build_engine()
        chain = await engine.run("search for information about NEXUS", TENANT)
        seals = await engine.ledger.get_chain(chain.id)
        assert len(seals[0].cot_trace) > 0

    @pytest.mark.asyncio
    async def test_out_of_scope_tool_produces_blocked_seal(self):
        from nexus.types import PersonaContract, RiskLevel
        personas = [PersonaContract(
            name="researcher",
            description="Researcher — can only search",
            allowed_tools=["knowledge_search"],  # NOT send_email
            resource_scopes=["kb:*"],
            intent_patterns=["search for information"],
            risk_tolerance=RiskLevel.LOW,
            max_ttl_seconds=60,
        )]
        engine = self._build_engine(personas=personas)
        # The ToolSelector only picks tools allowed by the persona.
        # Running a search task should NEVER result in send_email being executed.
        chain = await engine.run("search for information about NEXUS", TENANT)
        seals = await engine.ledger.get_chain(chain.id)
        assert all(s.tool_name != "send_email" for s in seals)

    @pytest.mark.asyncio
    async def test_callbacks_are_fired(self):
        events = []

        async def cb(event, data):
            events.append(event)

        engine = self._build_engine()
        engine.callbacks = [cb]
        await engine.run("search for information about NEXUS", TENANT)
        assert "chain_created" in events
        assert "step_started" in events
        assert "step_completed" in events
        assert "chain_completed" in events

    @pytest.mark.asyncio
    async def test_chain_integrity_is_verifiable(self):
        from nexus.core.notary import Notary
        engine = self._build_engine()
        chain = await engine.run("search for information about NEXUS", TENANT)
        seals = await engine.ledger.get_chain(chain.id)
        assert Notary().verify_chain(seals) is True


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Persistence
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
async def db_session():
    """In-memory SQLite session for repository tests."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from nexus.db.models import Base
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


class TestPhase5Repository:
    """Repository: tenant, persona, seal, chain, cost operations."""

    @pytest.mark.asyncio
    async def test_create_and_get_tenant(self, db_session):
        from nexus.db.repository import Repository
        repo = Repository(db_session)
        tenant = await repo.create_tenant("Test Corp", "hash-abc123")
        fetched = await repo.get_tenant(tenant.id)
        assert fetched is not None
        assert fetched.name == "Test Corp"

    @pytest.mark.asyncio
    async def test_get_tenant_by_api_key_hash(self, db_session):
        from nexus.db.repository import Repository
        repo = Repository(db_session)
        tenant = await repo.create_tenant("Key Corp", "hash-key-xyz")
        found = await repo.get_tenant_by_api_key_hash("hash-key-xyz")
        assert found is not None
        assert found.id == tenant.id

    @pytest.mark.asyncio
    async def test_upsert_and_list_personas(self, db_session):
        from nexus.db.repository import Repository
        repo = Repository(db_session)
        await repo.create_tenant("T", "hh")
        t = await repo.get_tenant_by_api_key_hash("hh")
        await repo.upsert_persona(t.id, {
            "name": "researcher",
            "description": "Searches",
            "allowed_tools": ["knowledge_search"],
            "resource_scopes": ["kb:*"],
            "intent_patterns": ["search"],
            "risk_tolerance": "low",
            "max_ttl_seconds": 60,
        })
        personas = await repo.list_personas(t.id)
        assert len(personas) == 1
        assert personas[0].name == "researcher"

    @pytest.mark.asyncio
    async def test_create_and_list_seals(self, db_session):
        from nexus.db.repository import Repository
        from nexus.types import (
            Seal, IntentDeclaration, AnomalyResult, GateResult,
            GateVerdict, RiskLevel, ActionStatus,
        )
        repo = Repository(db_session)
        await repo.create_tenant("T2", "hh2")
        t = await repo.get_tenant_by_api_key_hash("hh2")

        intent = IntentDeclaration(
            task_description="task", planned_action="p",
            tool_name="knowledge_search", tool_params={},
            resource_targets=[], reasoning="",
        )
        gate = GateResult(
            gate_name="scope", verdict=GateVerdict.PASS,
            score=1.0, threshold=1.0, details="",
        )
        anomaly = AnomalyResult(
            gates=[gate, gate, gate, gate],
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_id="researcher",
            action_fingerprint="fp",
        )
        seal = Seal(
            chain_id="chain-1", step_index=0, tenant_id=t.id,
            persona_id="researcher", intent=intent, anomaly_result=anomaly,
            tool_name="knowledge_search", tool_params={},
            status=ActionStatus.EXECUTED, fingerprint="fp1",
        )
        await repo.create_seal(seal)
        seals = await repo.list_seals(t.id)
        assert len(seals) == 1
        assert seals[0].id == seal.id

    @pytest.mark.asyncio
    async def test_seed_database_creates_demo_tenant_and_5_personas(self, db_session):
        from nexus.db.seed import seed_database
        await seed_database(db_session)
        from nexus.db.repository import Repository
        repo = Repository(db_session)
        tenant = await repo.get_tenant("demo")
        assert tenant is not None
        personas = await repo.list_personas("demo")
        assert len(personas) == 5
        names = {p.name for p in personas}
        assert "researcher" in names
        assert "analyst" in names

    @pytest.mark.asyncio
    async def test_seed_is_idempotent(self, db_session):
        from nexus.db.seed import seed_database
        await seed_database(db_session)
        await seed_database(db_session)  # second call must not duplicate
        from nexus.db.repository import Repository
        repo = Repository(db_session)
        personas = await repo.list_personas("demo")
        assert len(personas) == 5


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — LLM
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase6CostTracker:
    """CostTracker: cost calculation, budget enforcement, repository persistence."""

    @pytest.mark.asyncio
    async def test_returns_cost_record(self):
        from nexus.llm.cost_tracker import CostTracker
        from nexus.types import CostRecord
        tracker = CostTracker()
        record = await tracker.record(TENANT, CHAIN_ID, "seal-1", "gpt-4", {"input_tokens": 100, "output_tokens": 50})
        assert isinstance(record, CostRecord)
        assert record.input_tokens == 100
        assert record.output_tokens == 50

    @pytest.mark.asyncio
    async def test_known_model_yields_nonzero_cost(self):
        from nexus.llm.cost_tracker import CostTracker
        tracker = CostTracker()
        record = await tracker.record(TENANT, CHAIN_ID, None, "gpt-4", {"input_tokens": 200, "output_tokens": 100})
        assert record.cost_usd > 0.0

    @pytest.mark.asyncio
    async def test_unknown_model_falls_back_to_zero(self):
        from nexus.llm.cost_tracker import CostTracker
        tracker = CostTracker()
        record = await tracker.record(TENANT, CHAIN_ID, None, "mock/unknown", {"input_tokens": 100, "output_tokens": 50})
        assert record.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_cumulative_per_tenant(self):
        from nexus.llm.cost_tracker import CostTracker
        tracker = CostTracker()
        await tracker.record("t-A", CHAIN_ID, None, "gpt-4", {"input_tokens": 100, "output_tokens": 50})
        await tracker.record("t-B", CHAIN_ID, None, "gpt-4", {"input_tokens": 100, "output_tokens": 50})
        assert "t-A" in tracker._tenant_costs
        assert "t-B" in tracker._tenant_costs

    @pytest.mark.asyncio
    async def test_budget_exceeded_raises(self):
        from nexus.llm.cost_tracker import CostTracker
        from nexus.exceptions import BudgetExceeded
        tracker = CostTracker()
        tracker._tenant_costs[TENANT] = 49.999  # just below $50 budget
        with patch("nexus.llm.cost_tracker.litellm.completion_cost", return_value=0.01):
            with pytest.raises(BudgetExceeded) as exc_info:
                await tracker.record(TENANT, CHAIN_ID, None, "gpt-4", {"input_tokens": 1, "output_tokens": 1})
        assert exc_info.value.details["tenant_id"] == TENANT

    @pytest.mark.asyncio
    async def test_repository_called_on_record(self):
        from nexus.llm.cost_tracker import CostTracker
        mock_repo = MagicMock()
        mock_repo.add_cost = AsyncMock()
        tracker = CostTracker(repository=mock_repo)
        await tracker.record(TENANT, CHAIN_ID, "s1", "gpt-4", {"input_tokens": 10, "output_tokens": 5})
        mock_repo.add_cost.assert_awaited_once()


class TestPhase6LLMClient:
    """LLMClient: wraps litellm, returns structured response."""

    def _mock_response(self, content="answer", prompt_tokens=10, completion_tokens=5):
        message = MagicMock()
        message.content = content
        message.tool_calls = None
        choice = MagicMock()
        choice.message = message
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        return response

    @pytest.mark.asyncio
    async def test_complete_returns_structured_dict(self):
        from nexus.llm.client import LLMClient
        mock_resp = self._mock_response("hello world")
        with patch("nexus.llm.client.litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
            client = LLMClient(model="openai/gpt-4o")
            result = await client.complete([{"role": "user", "content": "hi"}])
        assert result["content"] == "hello world"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_provider_error_raises_nexus_error(self):
        from nexus.llm.client import LLMClient
        from nexus.exceptions import NexusError
        with patch("nexus.llm.client.litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("api down"))):
            client = LLMClient(model="openai/gpt-4o")
            with pytest.raises(NexusError):
                await client.complete([{"role": "user", "content": "hi"}])

    def test_default_model_from_config(self):
        from nexus.llm.client import LLMClient, select_model
        client = LLMClient()
        assert client.model == select_model()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Cache
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase7FingerprintCache:
    """FingerprintCache: store fingerprints, retrieve baseline statistics."""

    def _cache(self, lrange_return=None):
        from nexus.cache.fingerprints import FingerprintCache
        mock_redis = MagicMock()
        mock_redis._key = lambda t, k: f"nexus:{t}:{k}"
        mock_redis.client = MagicMock()
        mock_redis.client.rpush = AsyncMock(return_value=1)
        mock_redis.client.ltrim = AsyncMock(return_value=True)
        mock_redis.client.lrange = AsyncMock(return_value=lrange_return or [])
        return FingerprintCache(redis_client=mock_redis), mock_redis

    @pytest.mark.asyncio
    async def test_store_calls_rpush_then_ltrim(self):
        cache, redis = self._cache()
        await cache.store(TENANT, "researcher", "fp-abc")
        redis.client.rpush.assert_awaited_once()
        redis.client.ltrim.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_uses_tenant_namespaced_key(self):
        cache, redis = self._cache()
        await cache.store(TENANT, "researcher", "fp-abc")
        key_arg = redis.client.rpush.call_args[0][0]
        assert TENANT in key_arg
        assert "researcher" in key_arg

    @pytest.mark.asyncio
    async def test_get_baseline_empty_returns_zero_count(self):
        cache, _ = self._cache(lrange_return=[])
        result = await cache.get_baseline(TENANT, "researcher")
        assert result["sample_count"] == 0
        assert result["fingerprints"] == []
        assert result["frequency_map"] == {}

    @pytest.mark.asyncio
    async def test_get_baseline_counts_frequencies(self):
        fps = [b"fp-a", b"fp-b", b"fp-a", b"fp-a"]
        cache, _ = self._cache(lrange_return=fps)
        result = await cache.get_baseline(TENANT, "researcher")
        assert result["sample_count"] == 4
        assert result["frequency_map"]["fp-a"] == 3
        assert result["frequency_map"]["fp-b"] == 1

    @pytest.mark.asyncio
    async def test_get_baseline_decodes_bytes(self):
        cache, _ = self._cache(lrange_return=[b"fp-x", b"fp-y"])
        result = await cache.get_baseline(TENANT, "researcher")
        assert all(isinstance(f, str) for f in result["fingerprints"])

    @pytest.mark.asyncio
    async def test_store_limits_to_1000_entries(self):
        cache, redis = self._cache()
        await cache.store(TENANT, "researcher", "fp-1")
        redis.client.ltrim.assert_awaited_with(
            f"nexus:{TENANT}:fingerprints:researcher", -1000, -1
        )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8 — Auth
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase8JWT:
    """JWTManager: create and verify tokens."""

    @pytest.mark.asyncio
    async def test_create_token_returns_string(self):
        from nexus.auth.jwt import JWTManager
        mgr = JWTManager()
        token = await mgr.create_token("tenant-123", "admin")
        assert isinstance(token, str)
        assert len(token) > 10

    @pytest.mark.asyncio
    async def test_verify_token_returns_payload(self):
        from nexus.auth.jwt import JWTManager
        mgr = JWTManager()
        token = await mgr.create_token("tenant-abc", "user")
        payload = await mgr.verify_token(token)
        assert payload["tenant_id"] == "tenant-abc"
        assert payload["role"] == "user"

    @pytest.mark.asyncio
    async def test_invalid_token_raises_nexus_error(self):
        from nexus.auth.jwt import JWTManager
        from nexus.exceptions import NexusError
        mgr = JWTManager()
        with pytest.raises(NexusError):
            await mgr.verify_token("not.a.valid.token")

    @pytest.mark.asyncio
    async def test_round_trip_preserves_tenant_and_role(self):
        from nexus.auth.jwt import JWTManager
        mgr = JWTManager()
        token = await mgr.create_token("my-tenant", "admin")
        payload = await mgr.verify_token(token)
        assert payload["tenant_id"] == "my-tenant"
        assert payload["role"] == "admin"


class TestPhase8RateLimiter:
    """RateLimiter: Redis-backed fixed-window counting."""

    def _limiter(self, current_count=1):
        from nexus.auth.rate_limiter import RateLimiter
        mock_redis = MagicMock()
        mock_redis.client = MagicMock()
        mock_redis.client.incr = AsyncMock(return_value=current_count)
        mock_redis.client.expire = AsyncMock(return_value=True)
        return RateLimiter(redis_client=mock_redis)

    @pytest.mark.asyncio
    async def test_first_request_is_allowed(self):
        limiter = self._limiter(current_count=1)
        result = await limiter.check(TENANT, "api")
        assert result is True

    @pytest.mark.asyncio
    async def test_at_limit_is_allowed(self):
        from nexus.config import config
        limiter = self._limiter(current_count=config.rate_limit_requests_per_minute)
        result = await limiter.check(TENANT, "api")
        assert result is True

    @pytest.mark.asyncio
    async def test_over_limit_raises_nexus_error(self):
        from nexus.config import config
        from nexus.exceptions import NexusError
        limiter = self._limiter(current_count=config.rate_limit_requests_per_minute + 1)
        with pytest.raises(NexusError, match="Rate limit exceeded"):
            await limiter.check(TENANT, "api")

    @pytest.mark.asyncio
    async def test_chain_action_uses_hourly_limit(self):
        from nexus.config import config
        from nexus.exceptions import NexusError
        limiter = self._limiter(current_count=config.rate_limit_chains_per_hour + 1)
        with pytest.raises(NexusError):
            await limiter.check(TENANT, "chain")

    @pytest.mark.asyncio
    async def test_expire_called_on_first_request(self):
        from nexus.auth.rate_limiter import RateLimiter
        mock_redis = MagicMock()
        mock_redis.client = MagicMock()
        mock_redis.client.incr = AsyncMock(return_value=1)
        mock_redis.client.expire = AsyncMock(return_value=True)
        limiter = RateLimiter(redis_client=mock_redis)
        await limiter.check(TENANT, "api")
        mock_redis.client.expire.assert_awaited_once()


class TestPhase8AuthMiddleware:
    """AuthMiddleware: Bearer JWT + API key routing."""

    @pytest.mark.asyncio
    async def test_bearer_jwt_sets_tenant_on_state(self):
        from nexus.auth.middleware import AuthMiddleware
        from nexus.auth.jwt import JWTManager
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.testclient import TestClient as SC

        jwt_mgr = JWTManager()
        token = await jwt_mgr.create_token("tenant-jwt", "user")

        captured = {}

        async def endpoint(request: Request):
            captured["tenant_id"] = getattr(request.state, "tenant_id", None)
            return JSONResponse({"ok": True})

        from starlette.routing import Route
        app = Starlette(routes=[Route("/protected", endpoint)])
        app.add_middleware(AuthMiddleware, jwt_manager=jwt_mgr)

        client = SC(app)
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert captured["tenant_id"] == "tenant-jwt"

    @pytest.mark.asyncio
    async def test_missing_header_returns_401(self):
        from nexus.auth.middleware import AuthMiddleware
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient as SC

        async def endpoint(request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/protected", endpoint)])
        app.add_middleware(AuthMiddleware)
        client = SC(app)
        resp = client.get("/protected")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_public_paths_skip_auth(self):
        from nexus.auth.middleware import AuthMiddleware
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient as SC

        async def health(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/v1/health", health)])
        app.add_middleware(AuthMiddleware)
        client = SC(app)
        resp = client.get("/v1/health")
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 9 — API Routes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def api_client():
    """TestClient with full NEXUS app (no live DB/Redis — lifespan not run)."""
    from nexus.api.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers(api_client):
    """Get valid JWT headers via /v1/auth/token."""
    resp = api_client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
    assert resp.status_code == 200
    token = resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}


class TestPhase9Health:
    """GET /v1/health: returns status + version."""

    def test_health_200(self, api_client):
        resp = api_client.get("/v1/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, api_client):
        data = api_client.get("/v1/health").json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_has_version_field(self, api_client):
        data = api_client.get("/v1/health").json()
        assert "version" in data
        assert data["version"]  # non-empty

    def test_health_has_services_dict(self, api_client):
        data = api_client.get("/v1/health").json()
        assert "services" in data
        assert isinstance(data["services"], dict)
        assert "api" in data["services"]


class TestPhase9Auth:
    """POST /v1/auth/token: API key exchange."""

    def test_valid_demo_key_returns_200(self, api_client):
        resp = api_client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
        assert resp.status_code == 200

    def test_valid_demo_key_returns_token(self, api_client):
        resp = api_client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
        data = resp.json()
        assert "token" in data
        assert isinstance(data["token"], str)
        assert len(data["token"]) > 10

    def test_valid_demo_key_returns_tenant_id(self, api_client):
        resp = api_client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
        data = resp.json()
        assert "tenant_id" in data
        assert data["tenant_id"] == "demo"

    def test_valid_demo_key_returns_expires_in(self, api_client):
        resp = api_client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
        data = resp.json()
        assert "expires_in" in data
        assert data["expires_in"] > 0

    def test_invalid_key_returns_401(self, api_client):
        resp = api_client.post("/v1/auth/token", json={"api_key": "wrong_key"})
        assert resp.status_code == 401

    def test_missing_api_key_returns_422(self, api_client):
        resp = api_client.post("/v1/auth/token", json={})
        assert resp.status_code == 422


class TestPhase9AuthProtection:
    """Protected routes require valid auth."""

    def test_execute_without_auth_returns_401(self, api_client):
        resp = api_client.post("/v1/execute", json={"task": "test"})
        assert resp.status_code == 401

    def test_ledger_without_auth_returns_401(self, api_client):
        resp = api_client.get("/v1/ledger")
        assert resp.status_code == 401

    def test_personas_without_auth_returns_401(self, api_client):
        resp = api_client.get("/v1/personas")
        assert resp.status_code == 401

    def test_tools_without_auth_returns_401(self, api_client):
        resp = api_client.get("/v1/tools")
        assert resp.status_code == 401

    def test_knowledge_namespaces_without_auth_returns_401(self, api_client):
        resp = api_client.get("/v1/knowledge/namespaces")
        assert resp.status_code == 401

    def test_jwt_token_is_accepted_for_protected_routes(self, api_client, auth_headers):
        """A valid JWT reaches the route handler — initialize minimal state and assert 200."""
        from nexus.core.ledger import Ledger
        api_client.app.state.ledger = Ledger()
        resp = api_client.get("/v1/ledger", headers=auth_headers)
        assert resp.status_code == 200

    def test_tools_route_accessible_with_auth(self, api_client, auth_headers):
        from nexus.tools.registry import ToolRegistry
        api_client.app.state.tool_registry = ToolRegistry()
        resp = api_client.get("/v1/tools", headers=auth_headers)
        assert resp.status_code == 200

    def test_personas_route_accessible_with_auth(self, api_client, auth_headers):
        from nexus.core.personas import PersonaManager
        api_client.app.state.persona_manager = PersonaManager([])
        resp = api_client.get("/v1/personas", headers=auth_headers)
        assert resp.status_code == 200

    def test_ledger_chain_detail_unknown_id_returns_404(self, api_client, auth_headers):
        """GET /v1/ledger/{chain_id} with an unknown (or wrong-tenant) chain returns 404.

        This enforces the tenant-isolation guarantee: existence of another tenant's
        chain is never confirmed — the response is indistinguishable from not found.
        """
        from nexus.core.ledger import Ledger
        api_client.app.state.ledger = Ledger()
        resp = api_client.get("/v1/ledger/nonexistent-chain-id-xyz", headers=auth_headers)
        assert resp.status_code == 404

    def test_ledger_collection_with_no_seals_returns_200_empty(self, api_client, auth_headers):
        """GET /v1/ledger (collection) returns 200 with empty list when tenant has no seals."""
        from nexus.core.ledger import Ledger
        api_client.app.state.ledger = Ledger()
        resp = api_client.get("/v1/ledger", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["seals"] == []


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 19 — MCP Integration
# ─────────────────────────────────────────────────────────────────────────────

import sys as _sys  # noqa: E402
import threading as _threading  # noqa: E402
from http.server import BaseHTTPRequestHandler as _Handler, HTTPServer as _HTTPServer  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_FIXTURE_DIR = _Path(__file__).parent / "fixtures"
_ECHO_SERVER  = str(_FIXTURE_DIR / "mcp_echo_server.py")
_FETCH_SERVER = str(_FIXTURE_DIR / "mcp_fetch_server.py")


class TestPhase19MCPIntegration:
    """Smoke-level MCP integration — real subprocesses, zero mocks.

    Tests that NEXUS can:
      1. Connect to a FastMCP server (mcp SDK) and call its tools
      2. Connect to mcp-server-fetch (modelcontextprotocol/servers) and use it
      3. Register MCP tools via MCPToolAdapter alongside local tools
      4. Call an MCP tool through the ToolRegistry (end-to-end pipe)

    All assertions are concrete — if the MCP stack is broken, these fail.
    """

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _local_http_server(self):
        """Start a local HTTP server; return (server, url)."""
        body = b"<html><body><h1>Smoketest MCP Fetch</h1></body></html>"

        class _H(_Handler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *a): pass

        srv = _HTTPServer(("127.0.0.1", 0), _H)
        port = srv.server_address[1]
        t = _threading.Thread(target=srv.serve_forever)
        t.daemon = True
        t.start()
        return srv, f"http://127.0.0.1:{port}/"

    # ── Tests ─────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_connect_echo_server_and_call_tool(self):
        """Full stdio transport: spawn FastMCP server, echo tool returns input."""
        from nexus.mcp.client import MCPClient
        from nexus.types import MCPServerConfig
        cfg = MCPServerConfig(
            id="smoke-echo", tenant_id=TENANT, name="nexus-test-echo-server",
            url="", transport="stdio", command=_sys.executable,
            args=[_ECHO_SERVER],
        )
        client = MCPClient()
        try:
            defs = await client.connect(cfg)
            assert any(d.name == "mcp_nexus_test_echo_server_echo" for d in defs), (
                f"echo tool not found in {[d.name for d in defs]}"
            )
            result = await client.call_tool(cfg.id, "mcp_nexus_test_echo_server_echo", {"text": "smoke"})
            assert result == {"result": "smoke"}
        finally:
            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_connect_fetch_server_and_fetch_local_url(self):
        """mcp-server-fetch (modelcontextprotocol): fetches content from local HTTP server."""
        from nexus.mcp.client import MCPClient
        from nexus.types import MCPServerConfig
        srv, url = self._local_http_server()
        try:
            cfg = MCPServerConfig(
                id="smoke-fetch", tenant_id=TENANT, name="mcp-server-fetch",
                url="", transport="stdio", command=_sys.executable,
                args=[_FETCH_SERVER, "--ignore-robots-txt"],
            )
            client = MCPClient()
            try:
                defs = await client.connect(cfg)
                assert any(d.name == "mcp_mcp_server_fetch_fetch" for d in defs), (
                    f"fetch tool not found in {[d.name for d in defs]}"
                )
                result = await client.call_tool(
                    cfg.id, "mcp_mcp_server_fetch_fetch",
                    {"url": url, "max_length": 200},
                )
                assert "Smoketest MCP Fetch" in str(result), (
                    f"Expected page content in result: {str(result)[:200]}"
                )
            finally:
                await client.disconnect_all()
        finally:
            srv.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_registers_mcp_tools_with_correct_source(self):
        """MCPToolAdapter.register_server() stores tools with source='mcp' in registry."""
        from nexus.mcp.adapter import MCPToolAdapter
        from nexus.tools.registry import ToolRegistry
        from nexus.types import MCPServerConfig
        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry)
        cfg = MCPServerConfig(
            id="smoke-adapter", tenant_id=TENANT, name="nexus-test-echo-server",
            url="", transport="stdio", command=_sys.executable,
            args=[_ECHO_SERVER],
        )
        try:
            await adapter.register_server(TENANT, cfg)
            mcp_tools = registry.get_by_source("mcp")
            assert len(mcp_tools) == 3
            assert all(t.name.startswith("mcp_nexus_test_echo_server_") for t in mcp_tools)
        finally:
            await adapter._client.disconnect_all()

    @pytest.mark.asyncio
    async def test_registry_mcp_tool_callable_end_to_end(self):
        """Tool retrieved from ToolRegistry calls the real MCP subprocess and returns result."""
        from nexus.mcp.adapter import MCPToolAdapter
        from nexus.tools.registry import ToolRegistry
        from nexus.types import MCPServerConfig
        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry)
        cfg = MCPServerConfig(
            id="smoke-e2e", tenant_id=TENANT, name="nexus-test-echo-server",
            url="", transport="stdio", command=_sys.executable,
            args=[_ECHO_SERVER],
        )
        try:
            await adapter.register_server(TENANT, cfg)
            _, impl = registry.get("mcp_nexus_test_echo_server_add")
            result = await impl(a=21, b=21)
            assert result == {"result": 42}, f"Expected 42, got: {result}"
        finally:
            await adapter._client.disconnect_all()
