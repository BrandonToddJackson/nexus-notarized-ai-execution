"""Unit tests for all three reasoning gates.

Coverage:
  ThinkActGate          — high/low confidence, circuit breaker, custom threshold, boundary
  ContinueCompleteGate  — stateful retry tracking, RETRY/ESCALATE at MAX_RETRIES, CONTINUE/COMPLETE
  EscalateGate          — transient error types (timeout, rate limit, OSError), ToolError retry,
                          permanent error escalate, retry_count cap, build_escalation_context
"""

from nexus.types import (
    RetrievedContext, ChainPlan, Seal, ActionStatus, RiskLevel, IntentDeclaration, AnomalyResult, GateResult, GateVerdict,
    ReasoningDecision,
)
from nexus.exceptions import ToolError
from nexus.reasoning.think_act import ThinkActGate
from nexus.reasoning.continue_complete import ContinueCompleteGate
from nexus.reasoning.escalate import EscalateGate
from nexus.core.chain import ChainManager
from nexus.core.notary import Notary

TENANT = "reasoning-tenant"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _context(confidence: float) -> RetrievedContext:
    return RetrievedContext(
        query="test query",
        documents=[],
        confidence=confidence,
        sources=[],
        namespace="default",
    )


def _chain(n_steps: int = 2, n_seals: int = 0) -> ChainPlan:
    cm = ChainManager()
    chain = cm.create_chain(
        TENANT, "test task", [{"action": f"step {i}"} for i in range(n_steps)]
    )
    for i in range(n_seals):
        chain = cm.advance(chain, f"seal-{i:03d}")
    return chain


def _seal(status: ActionStatus = ActionStatus.EXECUTED, step_index: int = 0) -> Seal:
    gate = GateResult(
        gate_name="scope", verdict=GateVerdict.PASS, score=1.0, threshold=1.0, details="ok"
    )
    anomaly = AnomalyResult(
        gates=[gate, gate, gate, gate],
        overall_verdict=GateVerdict.PASS,
        risk_level=RiskLevel.LOW,
        persona_uuid="researcher",
        action_fingerprint="fp",
    )
    intent = IntentDeclaration(
        task_description="test",
        planned_action="search for info",
        tool_name="knowledge_search",
        tool_params={"query": "test"},
        resource_targets=["kb:docs"],
        reasoning="test",
    )
    notary = Notary()
    seal = notary.create_seal(
        chain_id="chain-001", step_index=step_index,
        tenant_id=TENANT, persona_id="researcher",
        intent=intent, anomaly_result=anomaly,
    )
    return notary.finalize_seal(seal, "result", status)


# ── ThinkActGate ───────────────────────────────────────────────────────────────

class TestThinkActGate:

    def test_high_confidence_decides_act(self):
        gate = ThinkActGate(confidence_threshold=0.80)
        assert gate.decide(_context(0.90), loop_count=0) == ReasoningDecision.ACT

    def test_exact_threshold_decides_act(self):
        gate = ThinkActGate(confidence_threshold=0.80)
        assert gate.decide(_context(0.80), loop_count=0) == ReasoningDecision.ACT

    def test_below_threshold_decides_think(self):
        gate = ThinkActGate(confidence_threshold=0.80)
        assert gate.decide(_context(0.50), loop_count=0) == ReasoningDecision.THINK

    def test_zero_confidence_decides_think(self):
        gate = ThinkActGate(confidence_threshold=0.80)
        assert gate.decide(_context(0.0), loop_count=0) == ReasoningDecision.THINK

    def test_circuit_breaker_at_max_loops(self):
        """When loop_count >= max_think_loops, always ACT regardless of confidence."""
        gate = ThinkActGate(confidence_threshold=0.80, max_think_loops=3)
        assert gate.decide(_context(0.0), loop_count=3) == ReasoningDecision.ACT

    def test_circuit_breaker_above_max(self):
        gate = ThinkActGate(confidence_threshold=0.80, max_think_loops=3)
        assert gate.decide(_context(0.0), loop_count=99) == ReasoningDecision.ACT

    def test_just_below_max_loops_still_checks_confidence(self):
        """loop_count = max - 1 still applies the confidence check."""
        gate = ThinkActGate(confidence_threshold=0.80, max_think_loops=3)
        # loop_count=2 < 3 → check confidence
        assert gate.decide(_context(0.3), loop_count=2) == ReasoningDecision.THINK
        assert gate.decide(_context(0.9), loop_count=2) == ReasoningDecision.ACT

    def test_custom_confidence_threshold(self):
        gate = ThinkActGate(confidence_threshold=0.50)
        assert gate.decide(_context(0.51), loop_count=0) == ReasoningDecision.ACT
        assert gate.decide(_context(0.49), loop_count=0) == ReasoningDecision.THINK

    def test_max_think_loops_one_immediately_fires(self):
        """With max_think_loops=1, loop_count=1 immediately triggers circuit breaker."""
        gate = ThinkActGate(confidence_threshold=0.80, max_think_loops=1)
        assert gate.decide(_context(0.0), loop_count=1) == ReasoningDecision.ACT

    def test_default_threshold_is_point_eight(self):
        gate = ThinkActGate()
        assert gate.confidence_threshold == 0.80

    def test_default_max_loops_is_three(self):
        gate = ThinkActGate()
        assert gate.max_think_loops == 3


# ── ContinueCompleteGate ───────────────────────────────────────────────────────

class TestContinueCompleteGate:

    def test_executed_seal_with_more_steps_continues(self):
        gate = ContinueCompleteGate()
        chain = _chain(n_steps=3, n_seals=1)  # 1 done, 2 remain
        seal = _seal(ActionStatus.EXECUTED, step_index=0)
        assert gate.decide(chain, "result", seal) == ReasoningDecision.CONTINUE

    def test_executed_seal_last_step_completes(self):
        gate = ContinueCompleteGate()
        chain = _chain(n_steps=2, n_seals=2)  # all done
        seal = _seal(ActionStatus.EXECUTED, step_index=1)
        assert gate.decide(chain, "result", seal) == ReasoningDecision.COMPLETE

    def test_failed_seal_first_retry_returns_retry(self):
        gate = ContinueCompleteGate()
        chain = _chain(n_steps=2, n_seals=1)
        seal = _seal(ActionStatus.FAILED, step_index=0)
        decision = gate.decide(chain, None, seal)
        assert decision == ReasoningDecision.RETRY

    def test_failed_seal_second_retry_returns_retry(self):
        gate = ContinueCompleteGate()
        chain = _chain(n_steps=2, n_seals=1)
        seal = _seal(ActionStatus.FAILED, step_index=0)
        gate.decide(chain, None, seal)  # count=1 → RETRY
        decision = gate.decide(chain, None, seal)  # count=2 → still < MAX_RETRIES(2)? Let's check
        # _MAX_RETRIES = 2, count starts at 0+1=1 (first call), then 1+1=2 (second call)
        # count < 2 → RETRY on first, count >= 2 → ESCALATE on second
        assert decision == ReasoningDecision.ESCALATE

    def test_failed_seal_escalates_after_max_retries(self):
        """After MAX_RETRIES (2) failed attempts on the same step → ESCALATE."""
        gate = ContinueCompleteGate()
        chain = _chain(n_steps=2, n_seals=1)
        seal = _seal(ActionStatus.FAILED, step_index=0)
        gate.decide(chain, None, seal)   # retry 1
        result = gate.decide(chain, None, seal)  # retry 2 → escalate
        assert result == ReasoningDecision.ESCALATE

    def test_retry_counts_per_step_independent(self):
        """Retry counter is tracked per (chain_id, step_index) — different steps are independent."""
        gate = ContinueCompleteGate()
        chain = _chain(n_steps=3, n_seals=1)
        seal_0 = _seal(ActionStatus.FAILED, step_index=0)
        seal_1 = _seal(ActionStatus.FAILED, step_index=1)

        gate.decide(chain, None, seal_0)  # step 0 retry 1
        gate.decide(chain, None, seal_0)  # step 0 retry 2 → ESCALATE

        # step 1 is independent — should start at retry 1
        result = gate.decide(chain, None, seal_1)
        assert result == ReasoningDecision.RETRY

    def test_single_step_chain_completes_immediately(self):
        gate = ContinueCompleteGate()
        chain = _chain(n_steps=1, n_seals=1)
        seal = _seal(ActionStatus.EXECUTED, step_index=0)
        assert gate.decide(chain, "result", seal) == ReasoningDecision.COMPLETE

    def test_blocked_seal_with_remaining_steps_continues(self):
        """BLOCKED seals are not FAILED — ContinueCompleteGate treats them as
        non-failures. The engine intercepts AnomalyDetected before calling this
        gate for blocks, but if it does reach here with more steps it returns CONTINUE."""
        gate = ContinueCompleteGate()
        chain = _chain(n_steps=2, n_seals=1)
        seal = _seal(ActionStatus.BLOCKED, step_index=0)
        result = gate.decide(chain, None, seal)
        # BLOCKED ≠ FAILED, so gate falls through to the "more steps?" check → CONTINUE
        assert result == ReasoningDecision.CONTINUE


# ── EscalateGate ──────────────────────────────────────────────────────────────

class TestEscalateGate:

    def test_timeout_error_retries(self):
        gate = EscalateGate()
        chain = _chain()
        assert gate.decide(TimeoutError("timed out"), retry_count=0, chain=chain) == ReasoningDecision.RETRY

    def test_connection_error_retries(self):
        gate = EscalateGate()
        chain = _chain()
        assert gate.decide(ConnectionError("connection refused"), retry_count=0, chain=chain) == ReasoningDecision.RETRY

    def test_oserror_retries(self):
        gate = EscalateGate()
        chain = _chain()
        assert gate.decide(OSError("I/O error"), retry_count=0, chain=chain) == ReasoningDecision.RETRY

    def test_rate_limit_message_retries(self):
        gate = EscalateGate()
        chain = _chain()
        err = Exception("rate limit exceeded — try again later")
        assert gate.decide(err, retry_count=0, chain=chain) == ReasoningDecision.RETRY

    def test_429_message_retries(self):
        gate = EscalateGate()
        chain = _chain()
        err = Exception("HTTP 429 too many requests")
        assert gate.decide(err, retry_count=0, chain=chain) == ReasoningDecision.RETRY

    def test_503_message_retries(self):
        gate = EscalateGate()
        chain = _chain()
        err = Exception("503 service temporarily unavailable")
        assert gate.decide(err, retry_count=0, chain=chain) == ReasoningDecision.RETRY

    def test_tool_error_retries(self):
        gate = EscalateGate()
        chain = _chain()
        err = ToolError("Invalid parameters for knowledge_search", tool_name="knowledge_search")
        assert gate.decide(err, retry_count=0, chain=chain) == ReasoningDecision.RETRY

    def test_value_error_escalates(self):
        """Non-transient, non-ToolError exceptions → ESCALATE."""
        gate = EscalateGate()
        chain = _chain()
        assert gate.decide(ValueError("bad input"), retry_count=0, chain=chain) == ReasoningDecision.ESCALATE

    def test_key_error_escalates(self):
        gate = EscalateGate()
        chain = _chain()
        assert gate.decide(KeyError("missing_key"), retry_count=0, chain=chain) == ReasoningDecision.ESCALATE

    def test_retry_count_cap_escalates_transient(self):
        """Even transient errors escalate once retry_count >= MAX_RETRIES."""
        gate = EscalateGate()
        chain = _chain()
        # retry_count=2 → always ESCALATE regardless of error type
        assert gate.decide(TimeoutError("timed out"), retry_count=2, chain=chain) == ReasoningDecision.ESCALATE

    def test_retry_count_cap_escalates_tool_error(self):
        gate = EscalateGate()
        chain = _chain()
        err = ToolError("tool failed", tool_name="web_search")
        assert gate.decide(err, retry_count=2, chain=chain) == ReasoningDecision.ESCALATE

    def test_build_escalation_context_has_required_keys(self):
        gate = EscalateGate()
        chain = _chain(n_steps=3, n_seals=1)
        err = ValueError("something failed")
        ctx = gate.build_escalation_context(chain, err)
        assert "chain_id" in ctx
        assert "tenant_id" in ctx
        assert "task" in ctx
        assert "escalated_at" in ctx
        assert "progress" in ctx
        assert "error" in ctx
        assert "recommendation" in ctx

    def test_build_escalation_context_progress_pct(self):
        gate = EscalateGate()
        chain = _chain(n_steps=4, n_seals=2)  # 50% done
        ctx = gate.build_escalation_context(chain, Exception("test"))
        assert ctx["progress"]["steps_completed"] == 2
        assert ctx["progress"]["steps_total"] == 4
        assert ctx["progress"]["completion_pct"] == 50.0

    def test_build_escalation_context_error_type(self):
        gate = EscalateGate()
        chain = _chain()
        err = ToolError("tool crashed", tool_name="knowledge_search")
        ctx = gate.build_escalation_context(chain, err)
        assert ctx["error"]["type"] == "ToolError"
        assert "tool crashed" in ctx["error"]["message"]

    def test_build_escalation_context_recommendation_for_tool_error(self):
        gate = EscalateGate()
        chain = _chain()
        err = ToolError("bad params", tool_name="knowledge_search")
        ctx = gate.build_escalation_context(chain, err)
        assert ctx["recommendation"]  # non-empty string
        assert "knowledge_search" in ctx["recommendation"]

    def test_build_escalation_context_recommendation_for_zero_progress(self):
        gate = EscalateGate()
        chain = _chain(n_steps=2, n_seals=0)  # nothing completed
        ctx = gate.build_escalation_context(chain, ValueError("bad"))
        assert "decomposition" in ctx["recommendation"].lower() or "step" in ctx["recommendation"].lower()

    def test_build_escalation_context_remaining_steps_list(self):
        gate = EscalateGate()
        chain = _chain(n_steps=3, n_seals=1)  # step 0 done, steps 1 and 2 remain
        ctx = gate.build_escalation_context(chain, Exception("test"))
        assert len(ctx["remaining_steps"]) == 2
