"""Tests for the core NEXUS engine — Phase 4.

All tests use a fully wired engine with real components and no LLM
(rule-based tool selection + fallback decomposition).

A MockKnowledgeStore is used so no ChromaDB connection is needed.
"""

import json
import pytest

from nexus.types import (
    PersonaContract, RiskLevel, ToolDefinition, RetrievedContext,
    ChainStatus, ActionStatus, TrustTier,
)
from nexus.exceptions import AnomalyDetected
from nexus.config import NexusConfig
from nexus.core.personas import PersonaManager
from nexus.core.anomaly import AnomalyEngine
from nexus.core.notary import Notary
from nexus.core.ledger import Ledger
from nexus.core.chain import ChainManager
from nexus.core.verifier import IntentVerifier
from nexus.core.output_validator import OutputValidator
from nexus.core.cot_logger import CoTLogger
from nexus.core.engine import NexusEngine
from nexus.knowledge.context import ContextBuilder
from nexus.tools.registry import ToolRegistry
from nexus.tools.selector import ToolSelector
from nexus.tools.executor import ToolExecutor
from nexus.tools.sandbox import Sandbox
from nexus.reasoning.think_act import ThinkActGate
from nexus.reasoning.continue_complete import ContinueCompleteGate
from nexus.reasoning.escalate import EscalateGate


# ── Stubs ─────────────────────────────────────────────────────────────────────

class MockKnowledgeStore:
    """No-op store — returns empty context. No ChromaDB needed."""
    def list_namespaces(self, tenant_id: str) -> list[str]:
        return []

    async def query(self, tenant_id, namespace, query, n_results=5, **kwargs) -> RetrievedContext:
        return RetrievedContext(
            query=query,
            documents=[],
            confidence=0.0,
            sources=[],
            namespace=namespace,
        )


class MockLLMClient:
    """Controllable mock: first call returns canned decompose JSON, rest fall back."""
    def __init__(self, decompose_steps: list[dict]):
        self._decompose_json = json.dumps(decompose_steps)
        self._call_count = 0

    async def complete(self, messages, **kwargs) -> dict:
        resp = self._decompose_json if self._call_count == 0 else "INVALID_JSON"
        self._call_count += 1
        return {"content": resp, "tool_calls": [], "usage": {"input_tokens": 5, "output_tokens": 5}}


# ── Engine factory ─────────────────────────────────────────────────────────────

def _make_engine(
    personas: list[PersonaContract],
    tools: dict[str, tuple[ToolDefinition, callable]] = None,
    llm_client=None,
) -> NexusEngine:
    config = NexusConfig(
        database_url="sqlite+aiosqlite:///test.db",
        redis_url="redis://localhost:6379/15",
        secret_key="test-secret",
    )

    registry = ToolRegistry()
    if tools:
        for defn, fn in tools.values():
            registry.register(defn, fn)

    persona_manager = PersonaManager(personas)
    anomaly_engine = AnomalyEngine(config)  # no embeddings → Gate 2 SKIP, no history → Gate 4 SKIP
    notary = Notary()
    ledger = Ledger()
    chain_manager = ChainManager()
    context_builder = ContextBuilder(MockKnowledgeStore())
    verifier = IntentVerifier()
    output_validator = OutputValidator()
    cot_logger = CoTLogger()
    selector = ToolSelector(registry, llm_client=llm_client)
    sandbox = Sandbox()
    executor = ToolExecutor(registry, sandbox, verifier)
    think_act_gate = ThinkActGate()
    cc_gate = ContinueCompleteGate()
    escalate_gate = EscalateGate()

    return NexusEngine(
        persona_manager=persona_manager,
        anomaly_engine=anomaly_engine,
        notary=notary,
        ledger=ledger,
        chain_manager=chain_manager,
        context_builder=context_builder,
        tool_registry=registry,
        tool_selector=selector,
        tool_executor=executor,
        output_validator=output_validator,
        cot_logger=cot_logger,
        think_act_gate=think_act_gate,
        continue_complete_gate=cc_gate,
        escalate_gate=escalate_gate,
        llm_client=llm_client,
        config=config,
    )


def _researcher() -> PersonaContract:
    return PersonaContract(
        name="researcher",
        description="Searches information",
        allowed_tools=["knowledge_search", "web_search"],
        resource_scopes=["kb:*", "web:*"],
        intent_patterns=["search for information", "find data"],
        risk_tolerance=RiskLevel.LOW,
        max_ttl_seconds=120,
    )


def _knowledge_search_tool() -> tuple[ToolDefinition, callable]:
    defn = ToolDefinition(
        name="knowledge_search",
        description="Search the knowledge base",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        risk_level=RiskLevel.LOW,
        resource_pattern="kb:*",
    )
    async def knowledge_search(query: str) -> str:
        return f"Results for: {query}"
    return defn, knowledge_search


def _send_email_tool() -> tuple[ToolDefinition, callable]:
    defn = ToolDefinition(
        name="send_email",
        description="Send an email",
        parameters={"type": "object", "properties": {"to": {"type": "string"}}, "required": ["to"]},
        risk_level=RiskLevel.HIGH,
        resource_pattern="email:*",
    )
    async def send_email(to: str) -> str:
        return f"Email sent to {to}"
    return defn, send_email


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestNexusEngine:

    @pytest.mark.asyncio
    async def test_single_action_produces_seal(self, test_tenant_id):
        """A simple task produces at least 1 seal with status EXECUTED."""
        defn, fn = _knowledge_search_tool()
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
        )
        chain = await engine.run("search for NEXUS documentation", test_tenant_id)

        assert chain.status == ChainStatus.COMPLETED
        assert len(chain.seals) >= 1

        # Verify the seal is in the ledger and has the right status
        seals = await engine.ledger.get_chain(chain.id)
        assert len(seals) >= 1
        assert seals[0].status == ActionStatus.EXECUTED
        assert seals[0].tenant_id == test_tenant_id

    @pytest.mark.asyncio
    async def test_multi_step_produces_multiple_seals(self, test_tenant_id):
        """A 2-step decomposition produces 2 seals."""
        defn, fn = _knowledge_search_tool()
        two_steps = [
            {"action": "search for basics", "tool": "knowledge_search",
             "params": {"query": "what is NEXUS"}, "persona": "researcher"},
            {"action": "search for details", "tool": "knowledge_search",
             "params": {"query": "NEXUS architecture"}, "persona": "researcher"},
        ]
        mock_llm = MockLLMClient(two_steps)
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
            llm_client=mock_llm,
        )
        chain = await engine.run("research NEXUS basics and architecture", test_tenant_id)

        assert chain.status == ChainStatus.COMPLETED
        assert len(chain.seals) == 2

        seals = await engine.ledger.get_chain(chain.id)
        assert len(seals) == 2
        assert all(s.status == ActionStatus.EXECUTED for s in seals)
        assert seals[0].step_index == 0
        assert seals[1].step_index == 1

    @pytest.mark.asyncio
    async def test_out_of_scope_blocked(self, test_tenant_id):
        """Researcher persona trying to use send_email → Gate 1 FAIL → AnomalyDetected."""
        email_defn, email_fn = _send_email_tool()
        # Only register send_email — no knowledge_search
        # Decompose fallback picks send_email as only tool
        # researcher doesn't allow send_email → Gate 1 FAIL
        engine = _make_engine(
            personas=[_researcher()],
            tools={"send_email": (email_defn, email_fn)},
        )

        with pytest.raises(AnomalyDetected) as exc_info:
            await engine.run("send email to user@example.com", test_tenant_id, persona_name="researcher")

        # Exception must carry gate results
        gate_results = exc_info.value.gate_results
        assert len(gate_results) > 0, "AnomalyDetected must carry at least one GateResult"

        # Gate 1 (scope) must have failed — send_email is not in researcher's allowed_tools
        scope_results = [g for g in gate_results if g.gate_name == "scope"]
        assert len(scope_results) == 1, "Expected exactly one scope gate result"
        assert scope_results[0].verdict.value == "fail", (
            f"Expected scope gate to FAIL but got {scope_results[0].verdict!r}"
        )

    @pytest.mark.asyncio
    async def test_chain_has_merkle_fingerprints(self, test_tenant_id):
        """Every seal must have a non-empty Merkle fingerprint."""
        defn, fn = _knowledge_search_tool()
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
        )
        chain = await engine.run("search for something", test_tenant_id)

        seals = await engine.ledger.get_chain(chain.id)
        for seal in seals:
            assert seal.fingerprint, f"Seal {seal.id} at step {seal.step_index} has no fingerprint"
            assert len(seal.fingerprint) == 64  # SHA256 hex digest

    @pytest.mark.asyncio
    async def test_cot_trace_captured(self, test_tenant_id):
        """Each executed seal must carry at least one CoT reasoning entry."""
        defn, fn = _knowledge_search_tool()
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
        )
        chain = await engine.run("search for AI research", test_tenant_id)

        seals = await engine.ledger.get_chain(chain.id)
        for seal in seals:
            assert seal.cot_trace, f"Seal {seal.id} has no CoT trace"
            assert len(seal.cot_trace) >= 1

    @pytest.mark.asyncio
    async def test_persona_revoked_after_each_step(self, test_tenant_id):
        """After each step, the persona's activation is revoked (TTL clock cleared)."""
        defn, fn = _knowledge_search_tool()
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
        )
        await engine.run("search for something", test_tenant_id)

        # After completion, researcher should not have an active TTL
        remaining = engine.persona_manager.get_ttl_remaining("researcher")
        assert remaining == 0  # revoked = 0 remaining

    @pytest.mark.asyncio
    async def test_merkle_chain_integrity_verifiable(self, test_tenant_id):
        """Ledger.verify_integrity must pass on a completed chain."""
        defn, fn = _knowledge_search_tool()
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
        )
        chain = await engine.run("search for integrity test", test_tenant_id)

        integrity_ok = await engine.ledger.verify_integrity(chain.id)
        assert integrity_ok is True

    @pytest.mark.asyncio
    async def test_callbacks_fired_with_correct_events(self, test_tenant_id):
        """Verify that lifecycle callbacks are called with the correct event names and keys."""

        events = []

        async def collect(event, data):
            events.append((event, data))

        defn, fn = _knowledge_search_tool()
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
        )
        await engine.run(
            "search for NEXUS documentation", test_tenant_id, callbacks=[collect]
        )

        event_names = [e[0] for e in events]
        assert "task_started" in event_names
        assert "chain_created" in event_names
        assert "step_started" in event_names
        assert "anomaly_checked" in event_names
        assert "step_completed" in event_names
        assert "chain_completed" in event_names

        # Verify anomaly_checked payload structure (used by SSE route)
        anomaly_event = next(d for n, d in events if n == "anomaly_checked")
        assert "chain_id" in anomaly_event
        assert "gates" in anomaly_event
        assert "step_index" in anomaly_event
        assert isinstance(anomaly_event["gates"], list)
        assert len(anomaly_event["gates"]) == 4

        for gate in anomaly_event["gates"]:
            assert "gate_name" in gate
            assert "verdict" in gate
            assert "score" in gate

    @pytest.mark.asyncio
    async def test_fingerprint_stored_after_successful_step(self, test_tenant_id):
        """After a successful execution, fingerprint store.store() must be awaited."""
        from unittest.mock import AsyncMock

        mock_fp_store = AsyncMock()
        mock_fp_store.store = AsyncMock()
        # AsyncMock auto-creates all attributes; explicitly remove get_baseline so the
        # engine falls back to the dict-check path (no async pre-fetch), step executes,
        # and store() is called.
        del mock_fp_store.get_baseline

        defn, fn = _knowledge_search_tool()
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
        )
        # Replace dict store with non-dict mock so the store() call path is taken
        engine.anomaly_engine.fingerprint_store = mock_fp_store

        await engine.run("search for NEXUS documentation", test_tenant_id)

        mock_fp_store.store.assert_awaited()
        call_args = mock_fp_store.store.call_args
        assert call_args.args[0] == test_tenant_id  # tenant_id
        assert isinstance(call_args.args[2], str) and len(call_args.args[2]) > 0  # fingerprint

    @pytest.mark.asyncio
    async def test_trust_degraded_after_gate_failure(self, test_tenant_id):
        """ESTABLISHED persona that fails Gate 1 must be degraded to COLD_START."""
        from nexus.exceptions import AnomalyDetected

        # Create ESTABLISHED researcher persona
        persona = PersonaContract(
            name="researcher",
            description="Searches information",
            allowed_tools=["knowledge_search"],
            resource_scopes=["kb:*"],
            intent_patterns=["search for information"],
            risk_tolerance=RiskLevel.LOW,
            max_ttl_seconds=120,
            trust_tier=TrustTier.ESTABLISHED,
        )
        # Only register send_email — researcher can't use it → Gate 1 FAIL
        email_defn, email_fn = _send_email_tool()
        engine = _make_engine(
            personas=[persona],
            tools={"send_email": (email_defn, email_fn)},
        )

        with pytest.raises(AnomalyDetected):
            await engine.run(
                "send email to user@example.com", test_tenant_id, persona_name="researcher"
            )

        # Trust tier must be degraded: ESTABLISHED → COLD_START
        degraded = engine.persona_manager._contracts.get("researcher")
        assert degraded is not None
        assert degraded.trust_tier == TrustTier.COLD_START

    @pytest.mark.asyncio
    async def test_cost_tracker_called_with_correct_args(self, test_tenant_id):
        """cost_tracker.record() is awaited with correct tenant_id and chain_id."""
        from unittest.mock import AsyncMock

        mock_cost_tracker = AsyncMock()
        mock_cost_tracker.record = AsyncMock()

        defn, fn = _knowledge_search_tool()
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
        )
        engine.cost_tracker = mock_cost_tracker

        await engine.run("search for NEXUS docs", test_tenant_id)

        mock_cost_tracker.record.assert_awaited()
        call_kwargs = mock_cost_tracker.record.call_args.kwargs
        assert call_kwargs["tenant_id"] == test_tenant_id
        assert isinstance(call_kwargs["chain_id"], str)
        assert call_kwargs["model"] is not None

    @pytest.mark.asyncio
    async def test_retry_decision_reruns_same_step(self, test_tenant_id):
        """When escalate_gate returns RETRY, the same step runs again."""
        from nexus.types import ReasoningDecision
        from nexus.exceptions import EscalationRequired, ToolError

        defn, fn = _knowledge_search_tool()
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
        )

        # Make tool_executor always raise ToolError
        execute_call_count = [0]

        async def always_fail(intent, **kwargs):
            execute_call_count[0] += 1
            raise ToolError("transient failure")

        engine.tool_executor.execute = always_fail

        # escalate_gate: RETRY on first exception, ESCALATE on second
        decide_call_count = [0]

        def controlled_decide(exc, retry_count, chain):
            decide_call_count[0] += 1
            if decide_call_count[0] == 1:
                return ReasoningDecision.RETRY
            return ReasoningDecision.ESCALATE

        engine.escalate_gate.decide = controlled_decide

        with pytest.raises(EscalationRequired):
            await engine.run("test task", test_tenant_id)

        # Step was attempted at least twice (initial + 1 retry)
        assert execute_call_count[0] >= 2
        assert decide_call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_human_approval_timeout_raises_escalation(self, test_tenant_id):
        """HUMAN_APPROVAL step with timeout_seconds=0 immediately raises EscalationRequired."""
        from nexus.types import WorkflowStep, StepType, ChainPlan, ChainStatus
        from nexus.exceptions import EscalationRequired

        defn, fn = _knowledge_search_tool()
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
        )

        chain = ChainPlan(
            tenant_id=test_tenant_id,
            task="approval test",
            steps=[],
            status=ChainStatus.EXECUTING,
        )
        step = WorkflowStep(
            id="approval-step-1",
            workflow_id=chain.id,
            name="needs_approval",
            step_type=StepType.HUMAN_APPROVAL,
            config={"timeout_seconds": 0, "instructions": "Please approve"},
        )
        context = {"steps": {}}

        with pytest.raises(EscalationRequired, match="timed out"):
            await engine._execute_approval_step(
                step, chain, context, step_index=0, tenant_id=test_tenant_id
            )
        # Context must record the timeout status
        assert context["steps"]["needs_approval"]["status"] == "timeout"

    @pytest.mark.asyncio
    async def test_human_approval_denied_raises_escalation(self, test_tenant_id):
        """HUMAN_APPROVAL step with 'denied' status raises EscalationRequired."""
        from unittest.mock import patch
        from nexus.types import WorkflowStep, StepType, ChainPlan, ChainStatus
        from nexus.exceptions import EscalationRequired

        defn, fn = _knowledge_search_tool()
        engine = _make_engine(
            personas=[_researcher()],
            tools={"knowledge_search": (defn, fn)},
        )

        chain = ChainPlan(
            tenant_id=test_tenant_id,
            task="approval test",
            steps=[],
            status=ChainStatus.EXECUTING,
        )
        step = WorkflowStep(
            id="approval-step-2",
            workflow_id=chain.id,
            name="needs_approval",
            step_type=StepType.HUMAN_APPROVAL,
            config={"timeout_seconds": 10},
        )
        context = {"steps": {}}
        approval_id = f"approval:{chain.id}:0"

        # Patch asyncio.sleep to set denied status on first call (no actual wait)
        call_count = [0]

        async def patched_sleep(seconds):
            call_count[0] += 1
            if (
                call_count[0] == 1
                and "_approvals" in context
                and approval_id in context["_approvals"]
            ):
                context["_approvals"][approval_id]["status"] = "denied"
            # Don't actually sleep

        with patch("nexus.core.engine.asyncio.sleep", patched_sleep):
            with pytest.raises(EscalationRequired):
                await engine._execute_approval_step(
                    step, chain, context, step_index=0, tenant_id=test_tenant_id
                )

        assert context["steps"]["needs_approval"]["status"] == "denied"
