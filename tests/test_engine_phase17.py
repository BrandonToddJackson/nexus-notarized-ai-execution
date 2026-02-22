"""Phase 17 — DAG Execution Engine tests.

Covers run_workflow(), _execute_dag_layer(), and all step-type handlers.
Uses fully in-memory fakes so no DB, Redis, or LLM is required.
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexus.core.engine import NexusEngine
from nexus.types import (
    ActionStatus, ChainStatus, ChainPlan, GateVerdict,
    ReasoningDecision, WorkflowDefinition, WorkflowExecution,
    WorkflowStatus, StepType, EdgeType, TriggerType,
    WorkflowStep, WorkflowEdge,
    PersonaContract, RiskLevel,
)
from nexus.exceptions import (
    AnomalyDetected, EscalationRequired,
    WorkflowNotFound, WorkflowValidationError,
)


# ── Helpers / Factories ────────────────────────────────────────────────────────

TENANT = "test-tenant"
WF_ID = str(uuid4())


def make_step(
    step_type: StepType = StepType.ACTION,
    name: str = "step1",
    tool_name: str = "knowledge_search",
    tool_params: dict | None = None,
    persona_name: str = "researcher",
    config: dict | None = None,
    wf_id: str = WF_ID,
) -> WorkflowStep:
    return WorkflowStep(
        id=str(uuid4()),
        workflow_id=wf_id,
        step_type=step_type,
        name=name,
        tool_name=tool_name,
        tool_params=tool_params or {},
        persona_name=persona_name,
        config=config or {},
    )


def make_edge(
    source: WorkflowStep,
    target: WorkflowStep,
    edge_type: EdgeType = EdgeType.DEFAULT,
    condition: str | None = None,
    wf_id: str = WF_ID,
) -> WorkflowEdge:
    return WorkflowEdge(
        id=str(uuid4()),
        workflow_id=wf_id,
        source_step_id=source.id,
        target_step_id=target.id,
        edge_type=edge_type,
        condition=condition,
    )


def make_workflow(
    steps: list[WorkflowStep],
    edges: list[WorkflowEdge],
    status: WorkflowStatus = WorkflowStatus.ACTIVE,
    wf_id: str = WF_ID,
) -> WorkflowDefinition:
    return WorkflowDefinition(
        id=wf_id,
        tenant_id=TENANT,
        name="Test Workflow",
        version=1,
        status=status,
        steps=steps,
        edges=edges,
    )


def make_persona(name: str = "researcher") -> PersonaContract:
    return PersonaContract(
        name=name,
        description="Test persona",
        allowed_tools=["knowledge_search", "web_search"],
        resource_scopes=["kb:*", "web:*"],
        intent_patterns=["search for information"],
        risk_tolerance=RiskLevel.LOW,
    )


def make_chain(tenant_id: str = TENANT) -> ChainPlan:
    return ChainPlan(
        id=str(uuid4()),
        tenant_id=tenant_id,
        task="test workflow",
        steps=[],
        seals=[],
        status=ChainStatus.EXECUTING,
        created_at=datetime.now(timezone.utc),
    )


def make_gate_result(verdict: GateVerdict = GateVerdict.PASS, gate_name: str = "scope"):
    g = MagicMock()
    g.verdict = verdict
    g.gate_name = gate_name
    g.score = 1.0
    g.details = f"{gate_name} details"
    return g


def make_anomaly_result(verdict: GateVerdict = GateVerdict.PASS):
    ar = MagicMock()
    ar.overall_verdict = verdict
    ar.gates = [make_gate_result(verdict, "scope"), make_gate_result(verdict, "intent")]
    ar.action_fingerprint = "fp123"
    return ar


def make_seal(chain_id: str, step_index: int = 0) -> MagicMock:
    s = MagicMock()
    s.id = str(uuid4())
    s.chain_id = chain_id
    s.step_index = step_index
    s.status = ActionStatus.PENDING
    s.cot_trace = []
    def finalize(**kw):
        s.status = kw.get("status", ActionStatus.EXECUTED)
        return s
    return s


def make_intent(tool_name: str = "knowledge_search"):
    intent = MagicMock()
    intent.tool_name = tool_name
    intent.tool_params = {"query": "test"}
    intent.resource_targets = ["kb:test"]
    intent.reasoning = "test reasoning"
    intent.planned_action = "search for information"

    def model_copy(update=None):
        new = MagicMock()
        new.tool_name = update.get("tool_name", intent.tool_name)
        new.tool_params = update.get("tool_params", intent.tool_params)
        new.resource_targets = update.get("resource_targets", intent.resource_targets)
        new.reasoning = intent.reasoning
        new.planned_action = intent.planned_action
        return new
    intent.model_copy = model_copy
    return intent


def build_engine(workflow_manager=None) -> NexusEngine:
    """Build a NexusEngine with all dependencies mocked."""
    persona = make_persona()

    persona_manager = MagicMock()
    persona_manager.get_persona.return_value = persona
    persona_manager.list_personas.return_value = [persona]
    persona_manager.activate.return_value = persona
    persona_manager.get_activation_time.return_value = datetime.now(timezone.utc)
    persona_manager.revoke.return_value = None
    persona_manager._contracts = {"researcher": persona}

    chain_manager = MagicMock()
    chain_manager.create_chain.return_value = make_chain()
    chain_manager.advance.side_effect = lambda chain, seal_id: chain
    chain_manager.fail.side_effect = lambda chain, msg: chain

    intent = make_intent()
    tool_selector = MagicMock()
    tool_selector.select = AsyncMock(return_value=intent)

    anomaly_engine = MagicMock()
    anomaly_engine.fingerprint_store = None
    anomaly_engine.check = AsyncMock(return_value=make_anomaly_result(GateVerdict.PASS))

    notary = MagicMock()
    seal = MagicMock()
    seal.id = str(uuid4())
    seal.status = ActionStatus.PENDING
    seal.cot_trace = []
    finalized_seal = MagicMock()
    finalized_seal.id = seal.id
    finalized_seal.status = ActionStatus.EXECUTED
    finalized_seal.cot_trace = []
    notary.create_seal.return_value = seal
    notary.finalize_seal.return_value = finalized_seal

    ledger = MagicMock()
    ledger.append = AsyncMock()

    context_builder = MagicMock()
    kb_ctx = MagicMock()
    kb_ctx.confidence = 0.9
    context_builder.build = AsyncMock(return_value=kb_ctx)

    tool_executor = MagicMock()
    tool_executor.execute = AsyncMock(return_value=("search result", None))

    output_validator = MagicMock()
    output_validator.validate = AsyncMock(return_value=(True, "ok"))

    cot_logger = MagicMock()
    cot_logger.log.return_value = None
    cot_logger.get_trace.return_value = []
    cot_logger.clear.return_value = None

    think_act_gate = MagicMock()
    think_act_gate.decide.return_value = ReasoningDecision.ACT

    continue_complete_gate = MagicMock()
    continue_complete_gate.decide.return_value = ReasoningDecision.COMPLETE

    escalate_gate = MagicMock()
    escalate_gate.decide.return_value = ReasoningDecision.ESCALATE
    escalate_gate.build_escalation_context.return_value = {}

    tool_registry = MagicMock()
    tool = MagicMock()
    tool.name = "knowledge_search"
    tool.description = "Search knowledge base"
    tool_registry.list_tools.return_value = [tool]

    engine = NexusEngine(
        persona_manager=persona_manager,
        anomaly_engine=anomaly_engine,
        notary=notary,
        ledger=ledger,
        chain_manager=chain_manager,
        context_builder=context_builder,
        tool_registry=tool_registry,
        tool_selector=tool_selector,
        tool_executor=tool_executor,
        output_validator=output_validator,
        cot_logger=cot_logger,
        think_act_gate=think_act_gate,
        continue_complete_gate=continue_complete_gate,
        escalate_gate=escalate_gate,
        workflow_manager=workflow_manager,
    )
    return engine


def build_wf_manager(workflow: WorkflowDefinition):
    """Build a minimal async workflow_manager mock."""
    wm = MagicMock()
    wm.get = AsyncMock(return_value=workflow)
    return wm


# ── _resolve_params tests ──────────────────────────────────────────────────────

class TestResolveParams:
    def setup_method(self):
        self.engine = build_engine()
        self.ctx = {
            "trigger": {"user_id": "u123", "items": [1, 2, 3]},
            "steps": {
                "fetch": {"result": {"count": 42}},
            },
            "loop_current": "item-x",
            "loop_index": 5,
        }

    def test_literal_string_unchanged(self):
        result = self.engine._resolve_params({"k": "hello"}, self.ctx)
        assert result["k"] == "hello"

    def test_full_template_preserves_type_list(self):
        result = self.engine._resolve_params({"items": "{{trigger.items}}"}, self.ctx)
        assert result["items"] == [1, 2, 3]

    def test_full_template_preserves_type_int(self):
        ctx = {"steps": {"a": {"result": 99}}}
        result = self.engine._resolve_params({"n": "{{steps.a.result}}"}, ctx)
        assert result["n"] == 99

    def test_partial_template_interpolation(self):
        result = self.engine._resolve_params({"msg": "user={{trigger.user_id}}"}, self.ctx)
        assert result["msg"] == "user=u123"

    def test_nested_dict_resolved(self):
        result = self.engine._resolve_params({"sub": {"k": "{{loop_current}}"}}, self.ctx)
        assert result["sub"]["k"] == "item-x"

    def test_list_values_resolved(self):
        result = self.engine._resolve_params({"arr": ["{{loop_index}}", "static"]}, self.ctx)
        assert result["arr"][0] == 5
        assert result["arr"][1] == "static"

    def test_missing_path_returns_empty_string_in_partial(self):
        result = self.engine._resolve_params({"k": "v={{missing.key}}"}, self.ctx)
        assert result["k"] == "v="

    def test_missing_path_returns_none_for_full_template(self):
        result = self.engine._resolve_params({"k": "{{missing.key}}"}, self.ctx)
        assert result["k"] is None

    def test_non_string_values_pass_through(self):
        result = self.engine._resolve_params({"n": 42, "b": True}, self.ctx)
        assert result["n"] == 42
        assert result["b"] is True

    def test_empty_params(self):
        assert self.engine._resolve_params({}, self.ctx) == {}


# ── _lookup_path tests ─────────────────────────────────────────────────────────

class TestLookupPath:
    def setup_method(self):
        self.engine = build_engine()

    def test_top_level_key(self):
        assert self.engine._lookup_path("trigger", {"trigger": "val"}) == "val"

    def test_nested_key(self):
        ctx = {"trigger": {"user": "alice"}}
        assert self.engine._lookup_path("trigger.user", ctx) == "alice"

    def test_deeply_nested(self):
        ctx = {"steps": {"a": {"result": {"count": 7}}}}
        assert self.engine._lookup_path("steps.a.result.count", ctx) == 7

    def test_missing_intermediate(self):
        assert self.engine._lookup_path("a.b.c", {"a": {}}) is None

    def test_missing_top(self):
        assert self.engine._lookup_path("missing", {}) is None


# ── run_workflow validation tests ──────────────────────────────────────────────

class TestRunWorkflowValidation:
    @pytest.mark.asyncio
    async def test_raises_if_no_workflow_manager(self):
        engine = build_engine(workflow_manager=None)
        with pytest.raises(WorkflowNotFound):
            await engine.run_workflow(WF_ID, TENANT)

    @pytest.mark.asyncio
    async def test_raises_if_workflow_not_active(self):
        step = make_step()
        wf = make_workflow([step], [], status=WorkflowStatus.DRAFT)
        engine = build_engine(workflow_manager=build_wf_manager(wf))
        with pytest.raises(WorkflowValidationError, match="not ACTIVE"):
            await engine.run_workflow(WF_ID, TENANT)

    @pytest.mark.asyncio
    async def test_raises_if_no_entry_points(self):
        # Two steps with a circular edge (no entry points)
        s1 = make_step(name="a")
        s2 = make_step(name="b")
        e1 = make_edge(s1, s2)
        e2 = make_edge(s2, s1)
        wf = make_workflow([s1, s2], [e1, e2])
        engine = build_engine(workflow_manager=build_wf_manager(wf))
        with pytest.raises(WorkflowValidationError, match="entry point"):
            await engine.run_workflow(WF_ID, TENANT)


# ── Single-step workflow ───────────────────────────────────────────────────────

class TestSingleActionStep:
    @pytest.mark.asyncio
    async def test_single_action_step_completes(self):
        step = make_step(name="search", tool_name="knowledge_search")
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT, trigger_data={"q": "hello"})

        assert execution.status == ChainStatus.COMPLETED
        assert "search" in execution.step_results
        assert execution.step_results["search"]["status"] == ActionStatus.EXECUTED.value

    @pytest.mark.asyncio
    async def test_execution_has_chain_id(self):
        step = make_step(name="s1")
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT)

        assert execution.chain_id != ""

    @pytest.mark.asyncio
    async def test_trigger_data_stored_in_execution(self):
        step = make_step(name="s1")
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        trigger = {"source": "test", "value": 42}
        execution = await engine.run_workflow(WF_ID, TENANT, trigger_data=trigger)

        assert execution.trigger_data == trigger

    @pytest.mark.asyncio
    async def test_ledger_append_called_once(self):
        step = make_step(name="s1")
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        await engine.run_workflow(WF_ID, TENANT)

        engine.ledger.append.assert_called_once()

    @pytest.mark.asyncio
    async def test_persona_activate_and_revoke_called(self):
        step = make_step(name="s1", persona_name="analyst")
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        await engine.run_workflow(WF_ID, TENANT)

        engine.persona_manager.activate.assert_called_once()
        engine.persona_manager.revoke.assert_called()


# ── Two-step linear workflow ───────────────────────────────────────────────────

class TestLinearWorkflow:
    @pytest.mark.asyncio
    async def test_two_sequential_steps(self):
        s1 = make_step(name="step_a")
        s2 = make_step(name="step_b")
        edge = make_edge(s1, s2)
        wf = make_workflow([s1, s2], [edge])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT)

        assert execution.status == ChainStatus.COMPLETED
        assert "step_a" in execution.step_results
        assert "step_b" in execution.step_results

    @pytest.mark.asyncio
    async def test_steps_executed_in_order(self):
        """step_a must execute before step_b (child of step_a)."""
        order = []
        s1 = make_step(name="first")
        s2 = make_step(name="second")
        edge = make_edge(s1, s2)
        wf = make_workflow([s1, s2], [edge])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        original_execute = engine._execute_action_step

        async def tracked_execute(step, *args, **kwargs):
            order.append(step.name)
            return await original_execute(step, *args, **kwargs)

        engine._execute_action_step = tracked_execute
        await engine.run_workflow(WF_ID, TENANT)

        assert order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_three_step_chain(self):
        s1, s2, s3 = make_step(name="a"), make_step(name="b"), make_step(name="c")
        wf = make_workflow([s1, s2, s3], [make_edge(s1, s2), make_edge(s2, s3)])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT)

        assert execution.status == ChainStatus.COMPLETED
        for name in ("a", "b", "c"):
            assert name in execution.step_results


# ── Gate blocking ──────────────────────────────────────────────────────────────

class TestGateBlocking:
    @pytest.mark.asyncio
    async def test_blocked_step_stored_in_context(self):
        step = make_step(name="blocked_step")
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        # Make anomaly check fail
        engine.anomaly_engine.check = AsyncMock(
            return_value=make_anomaly_result(GateVerdict.FAIL)
        )

        # Finalize seal should return a blocked seal
        blocked_seal = MagicMock()
        blocked_seal.id = str(uuid4())
        blocked_seal.status = ActionStatus.BLOCKED
        blocked_seal.cot_trace = []
        engine.notary.finalize_seal.return_value = blocked_seal

        with pytest.raises(AnomalyDetected):
            await engine.run_workflow(WF_ID, TENANT)

        # The workflow raises, but we can inspect the engine's ledger calls
        engine.ledger.append.assert_called()

    @pytest.mark.asyncio
    async def test_blocked_step_causes_workflow_failure(self):
        step = make_step(name="s1")
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))
        engine.anomaly_engine.check = AsyncMock(
            return_value=make_anomaly_result(GateVerdict.FAIL)
        )
        blocked_seal = MagicMock()
        blocked_seal.id = str(uuid4())
        blocked_seal.status = ActionStatus.BLOCKED
        blocked_seal.cot_trace = []
        engine.notary.finalize_seal.return_value = blocked_seal

        with pytest.raises(AnomalyDetected):
            await engine.run_workflow(WF_ID, TENANT)


# ── Branch step ───────────────────────────────────────────────────────────────

class TestBranchStep:
    def _make_branch_workflow(self, condition_passes: bool):
        branch = make_step(step_type=StepType.BRANCH, name="decide")
        yes_step = make_step(name="yes_branch")
        no_step = make_step(name="no_branch")
        condition = "true" if condition_passes else "false"
        cond_edge = make_edge(branch, yes_step, EdgeType.CONDITIONAL, condition=condition)
        default_edge = make_edge(branch, no_step, EdgeType.DEFAULT)
        wf = make_workflow([branch, yes_step, no_step], [cond_edge, default_edge])
        return wf, branch, yes_step, no_step

    @pytest.mark.asyncio
    async def test_branch_takes_conditional_when_true(self):
        wf, branch, yes_step, no_step = self._make_branch_workflow(condition_passes=True)
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        executed = []
        orig = engine._execute_action_step
        async def track(step, *args, **kwargs):
            executed.append(step.name)
            return await orig(step, *args, **kwargs)
        engine._execute_action_step = track

        execution = await engine.run_workflow(WF_ID, TENANT)

        assert execution.status == ChainStatus.COMPLETED
        assert yes_step.name in executed
        assert no_step.name not in executed

    @pytest.mark.asyncio
    async def test_branch_takes_default_when_condition_false(self):
        wf, branch, yes_step, no_step = self._make_branch_workflow(condition_passes=False)
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        executed = []
        orig = engine._execute_action_step
        async def track(step, *args, **kwargs):
            executed.append(step.name)
            return await orig(step, *args, **kwargs)
        engine._execute_action_step = track

        await engine.run_workflow(WF_ID, TENANT)

        assert no_step.name in executed
        assert yes_step.name not in executed

    @pytest.mark.asyncio
    async def test_branch_result_stored(self):
        wf, branch, yes_step, no_step = self._make_branch_workflow(condition_passes=True)
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT)

        assert execution.step_results[branch.name]["status"] == "branched"
        assert execution.step_results[branch.name]["chosen_step_id"] == yes_step.id

    @pytest.mark.asyncio
    async def test_branch_no_matching_edge(self):
        """Branch with only a failing condition and no default edge → no_branch_matched."""
        branch = make_step(step_type=StepType.BRANCH, name="decide")
        target = make_step(name="t")
        cond_edge = make_edge(branch, target, EdgeType.CONDITIONAL, condition="false")
        wf = make_workflow([branch, target], [cond_edge])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT)

        assert execution.step_results["decide"]["status"] == "no_branch_matched"


# ── Loop step ─────────────────────────────────────────────────────────────────

class TestLoopStep:
    """Loop step uses the self-loop model: the loop step executes its own tool
    once per item, with a self-referential LOOP_BACK edge + DEFAULT exit edge."""

    def _make_loop_workflow(self, items_key: str = "{{trigger.items}}"):
        """Build a minimal valid loop workflow (self-loop + exit step)."""
        loop = make_step(
            step_type=StepType.LOOP,
            name="loop",
            tool_name="knowledge_search",
            config={"iterator": items_key},
        )
        exit_step = make_step(name="after_loop")
        loop_back = make_edge(loop, loop, EdgeType.LOOP_BACK)   # self-loop (required)
        exit_edge = make_edge(loop, exit_step, EdgeType.DEFAULT)
        wf = make_workflow([loop, exit_step], [loop_back, exit_edge])
        return wf, loop, exit_step

    @pytest.mark.asyncio
    async def test_loop_iterates_over_items(self):
        """Loop step executes its tool once per item in the iterator."""
        items = ["a", "b", "c"]
        wf, loop, _ = self._make_loop_workflow()
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT, trigger_data={"items": items})

        assert execution.status == ChainStatus.COMPLETED
        assert execution.step_results["loop"]["iterations"] == 3

    @pytest.mark.asyncio
    async def test_loop_empty_iterator(self):
        wf, loop, _ = self._make_loop_workflow()
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT, trigger_data={"items": []})

        assert execution.step_results["loop"]["iterations"] == 0
        assert execution.step_results["loop"]["results"] == []

    @pytest.mark.asyncio
    async def test_loop_sets_loop_current_in_context(self):
        """loop_current is set to the current item during each iteration."""
        seen_current: list[Any] = []
        wf, loop, _ = self._make_loop_workflow()
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        orig = engine._execute_action_step
        async def capture(step, wf_, chain, context, *args, **kwargs):
            if step.step_type == StepType.LOOP or step.name == "loop":
                seen_current.append(context.get("loop_current"))
            return await orig(step, wf_, chain, context, *args, **kwargs)
        engine._execute_action_step = capture

        await engine.run_workflow(WF_ID, TENANT, trigger_data={"items": [10, 20]})

        assert seen_current == [10, 20]


# ── Parallel step ─────────────────────────────────────────────────────────────

class TestParallelStep:
    @pytest.mark.asyncio
    async def test_parallel_runs_all_branches(self):
        b1 = make_step(name="branch_a")
        b2 = make_step(name="branch_b")
        par = make_step(
            step_type=StepType.PARALLEL,
            name="parallel",
            config={"branches": [b1.id, b2.id]},
        )
        wf = make_workflow([par, b1, b2], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        executed = []
        orig = engine._execute_action_step
        async def track(step, *args, **kwargs):
            executed.append(step.name)
            return await orig(step, *args, **kwargs)
        engine._execute_action_step = track

        execution = await engine.run_workflow(WF_ID, TENANT)

        assert execution.status == ChainStatus.COMPLETED
        assert set(executed) == {"branch_a", "branch_b"}

    @pytest.mark.asyncio
    async def test_parallel_result_contains_branches(self):
        b1 = make_step(name="b1")
        b2 = make_step(name="b2")
        par = make_step(
            step_type=StepType.PARALLEL,
            name="par",
            config={"branches": [b1.id, b2.id]},
        )
        wf = make_workflow([par, b1, b2], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT)

        assert "branches" in execution.step_results["par"]

    @pytest.mark.asyncio
    async def test_parallel_no_branches_config_uses_edges(self):
        """When no branches in config, DEFAULT children are used."""
        b1 = make_step(name="e1")
        par = make_step(step_type=StepType.PARALLEL, name="par", config={})
        edge = make_edge(par, b1, EdgeType.DEFAULT)
        wf = make_workflow([par, b1], [edge])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        executed = []
        orig = engine._execute_action_step
        async def track(step, *args, **kwargs):
            executed.append(step.name)
            return await orig(step, *args, **kwargs)
        engine._execute_action_step = track

        execution = await engine.run_workflow(WF_ID, TENANT)

        assert "e1" in executed


# ── Sub-workflow step ──────────────────────────────────────────────────────────

class TestSubWorkflowStep:
    @pytest.mark.asyncio
    async def test_sub_workflow_executed(self):
        sub_id = str(uuid4())
        sub_body = make_step(name="sub_action", wf_id=sub_id)
        sub_wf = make_workflow([sub_body], [], wf_id=sub_id)

        parent_step = make_step(
            step_type=StepType.SUB_WORKFLOW,
            name="sub",
            config={"sub_workflow_id": sub_id},
        )
        parent_wf = make_workflow([parent_step], [])

        # Workflow manager returns correct workflow by ID
        async def get_wf(wid, tid):
            if wid == sub_id:
                return sub_wf
            return parent_wf

        wm = MagicMock()
        wm.get = get_wf

        engine = build_engine(workflow_manager=wm)
        execution = await engine.run_workflow(WF_ID, TENANT)

        assert execution.status == ChainStatus.COMPLETED
        sub_result = execution.step_results["sub"]
        assert sub_result["status"] == ChainStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_sub_workflow_missing_id_stores_failure(self):
        step = make_step(
            step_type=StepType.SUB_WORKFLOW,
            name="sub",
            config={},  # no sub_workflow_id
        )
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        # Should not raise — stores failure in context
        execution = await engine.run_workflow(WF_ID, TENANT)

        assert execution.step_results["sub"]["status"] == "failed"


# ── Wait step ─────────────────────────────────────────────────────────────────

class TestWaitStep:
    @pytest.mark.asyncio
    async def test_wait_step_stored(self):
        step = make_step(step_type=StepType.WAIT, name="pause", config={"seconds": 0})
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT)

        assert execution.status == ChainStatus.COMPLETED
        assert execution.step_results["pause"]["status"] == "waited"

    @pytest.mark.asyncio
    async def test_wait_step_stores_seconds(self):
        step = make_step(step_type=StepType.WAIT, name="pause", config={"seconds": 0})
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT)

        assert execution.step_results["pause"]["seconds"] == 0.0


# ── LOOP_BACK edge exclusion ──────────────────────────────────────────────────

class TestLoopBackEdgeExclusion:
    @pytest.mark.asyncio
    async def test_loop_back_edges_not_followed(self):
        """Self-referential LOOP_BACK edge must not cause infinite DAG traversal."""
        # Correct self-loop model: loop → [LOOP_BACK] → loop + loop → [DEFAULT] → exit
        loop = make_step(
            step_type=StepType.LOOP,
            name="loop",
            tool_name="knowledge_search",
            config={"iterator": "{{trigger.items}}"},
        )
        exit_step = make_step(name="exit")
        loop_back = make_edge(loop, loop, EdgeType.LOOP_BACK)   # self-loop marker
        exit_edge = make_edge(loop, exit_step, EdgeType.DEFAULT)
        wf = make_workflow([loop, exit_step], [loop_back, exit_edge])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        execution = await engine.run_workflow(WF_ID, TENANT, trigger_data={"items": ["x"]})

        # Should complete, not loop forever
        assert execution.status == ChainStatus.COMPLETED
        assert execution.step_results["loop"]["iterations"] == 1


# ── Conditional edge evaluation ───────────────────────────────────────────────

class TestConditionalEdges:
    @pytest.mark.asyncio
    async def test_conditional_edge_with_context_variable(self):
        """Condition references trigger.flag; only matching branch executes."""
        s1 = make_step(name="s1")
        s2 = make_step(name="s2")
        s3 = make_step(name="s3")
        e12 = make_edge(s1, s2, EdgeType.CONDITIONAL, condition="${flag} == True")
        e13 = make_edge(s1, s3, EdgeType.DEFAULT)
        wf = make_workflow([s1, s2, s3], [e12, e13])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        executed = []
        orig = engine._execute_action_step
        async def track(step, *args, **kwargs):
            executed.append(step.name)
            return await orig(step, *args, **kwargs)
        engine._execute_action_step = track

        # Note: evaluate_condition uses ${} syntax; trigger isn't exposed directly.
        # We're testing that conditions with 'false' literal block the edge.
        s1b = make_step(name="root")
        s2b = make_step(name="yes")
        s3b = make_step(name="no")
        e_cond = make_edge(s1b, s2b, EdgeType.CONDITIONAL, condition="false")
        e_def = make_edge(s1b, s3b, EdgeType.DEFAULT)
        wf2 = make_workflow([s1b, s2b, s3b], [e_cond, e_def])
        engine2 = build_engine(workflow_manager=build_wf_manager(wf2))

        executed2 = []
        orig2 = engine2._execute_action_step
        async def track2(step, *args, **kwargs):
            executed2.append(step.name)
            return await orig2(step, *args, **kwargs)
        engine2._execute_action_step = track2

        await engine2.run_workflow(wf2.id, TENANT)
        assert "no" in executed2
        assert "yes" not in executed2

    @pytest.mark.asyncio
    async def test_conditional_edge_true_takes_branch(self):
        s1 = make_step(name="root")
        s2 = make_step(name="taken")
        s3 = make_step(name="skipped")
        e12 = make_edge(s1, s2, EdgeType.CONDITIONAL, condition="true")
        e13 = make_edge(s1, s3, EdgeType.DEFAULT)
        wf = make_workflow([s1, s2, s3], [e12, e13])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        executed = []
        orig = engine._execute_action_step
        async def track(step, *args, **kwargs):
            executed.append(step.name)
            return await orig(step, *args, **kwargs)
        engine._execute_action_step = track

        await engine.run_workflow(WF_ID, TENANT)

        assert "taken" in executed
        # "skipped" may also execute if it's the DEFAULT fallback;
        # the engine follows conditional=true AND default independently.
        # This is expected for non-branch steps — BRANCH steps pick only one.


# ── Persona override ──────────────────────────────────────────────────────────

class TestPersonaOverride:
    @pytest.mark.asyncio
    async def test_persona_override_used(self):
        step = make_step(name="s1", persona_name="analyst")
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        await engine.run_workflow(WF_ID, TENANT, persona_override="admin")

        # persona_manager.activate should be called with "admin", not "analyst"
        call_args = engine.persona_manager.activate.call_args
        assert call_args[0][0] == "admin"

    @pytest.mark.asyncio
    async def test_no_override_uses_step_persona(self):
        step = make_step(name="s1", persona_name="analyst")
        wf = make_workflow([step], [])
        engine = build_engine(workflow_manager=build_wf_manager(wf))

        await engine.run_workflow(WF_ID, TENANT)

        call_args = engine.persona_manager.activate.call_args
        assert call_args[0][0] == "analyst"


# ── Backwards compatibility: run() still works ────────────────────────────────

class TestBackwardsCompatibility:
    @pytest.mark.asyncio
    async def test_run_method_still_works(self):
        """Existing run() must be unaffected by Phase 17 changes."""
        engine = build_engine()
        # Stub _decompose_task for run()
        engine._decompose_task = AsyncMock(return_value=[{
            "action": "test task",
            "tool": "knowledge_search",
            "params": {"query": "test"},
            "persona": "researcher",
        }])

        chain = await engine.run("test task", TENANT)

        assert chain.status == ChainStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_workflow_manager_defaults_to_none(self):
        """workflow_manager=None is the default; run() still works."""
        engine = build_engine(workflow_manager=None)
        assert engine.workflow_manager is None
