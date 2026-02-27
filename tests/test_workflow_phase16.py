"""
Phase 16 — Workflow Definition Layer tests.

Coverage:
  dag.py        — topological_sort, entry/exit points, children/parents,
                  parallel groups, evaluate_condition (including membership,
                  parentheses, real context data)
  validator.py  — all 9 structural checks + duplicate-warning deduplication
  manager.py    — full lifecycle: create, get, update, activate, pause,
                  archive, list (pagination), version history, rollback,
                  duplicate (edge integrity), export/import JSON
                  + repository integration + silent NotImplementedError
  smoke tests   — end-to-end realistic scenarios with real tool/persona names
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.exceptions import WorkflowNotFound, WorkflowValidationError
from nexus.types import (
    EdgeType,
    StepType,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowStatus,
    WorkflowStep,
)
from nexus.workflows import WorkflowManager, WorkflowValidator
from nexus.workflows.dag import (
    evaluate_condition,
    get_children,
    get_entry_points,
    get_exit_points,
    get_parallel_group,
    get_parents,
    topological_sort,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

TENANT = "tenant-abc"


def _step(sid: str, name: str = "", step_type: StepType = StepType.ACTION,
          tool_name: str | None = None) -> WorkflowStep:
    return WorkflowStep(
        id=sid,
        workflow_id="wf-test",
        step_type=step_type,
        name=name or sid,
        tool_name=tool_name,
    )


def _edge(
    eid: str,
    src: str,
    tgt: str,
    edge_type: EdgeType = EdgeType.DEFAULT,
    condition: str | None = None,
) -> WorkflowEdge:
    return WorkflowEdge(
        id=eid,
        workflow_id="wf-test",
        source_step_id=src,
        target_step_id=tgt,
        edge_type=edge_type,
        condition=condition,
    )


def _wf(
    steps: list[WorkflowStep],
    edges: list[WorkflowEdge],
    wid: str = "wf-1",
    tenant: str = TENANT,
) -> WorkflowDefinition:
    now = datetime.now(tz=timezone.utc)
    return WorkflowDefinition(
        id=wid,
        tenant_id=tenant,
        name="Test WF",
        steps=steps,
        edges=edges,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def validator() -> WorkflowValidator:
    return WorkflowValidator()


@pytest.fixture
def manager(validator: WorkflowValidator) -> WorkflowManager:
    from nexus.config import NexusConfig
    return WorkflowManager(repository=None, validator=validator, config=NexusConfig())


# ─────────────────────────────────────────────────────────────────────────────
# dag.py — topological_sort
# ─────────────────────────────────────────────────────────────────────────────


class TestTopologicalSort:
    def test_empty_graph(self):
        assert topological_sort([], []) == []

    def test_single_step(self):
        steps = [_step("s1")]
        assert topological_sort(steps, []) == ["s1"]

    def test_linear_chain(self):
        steps = [_step("s1"), _step("s2"), _step("s3")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s2", "s3")]
        order = topological_sort(steps, edges)
        assert order.index("s1") < order.index("s2")
        assert order.index("s2") < order.index("s3")

    def test_branching(self):
        # s1 → s2, s1 → s3 (both branch from s1)
        steps = [_step("s1"), _step("s2"), _step("s3")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s1", "s3")]
        order = topological_sort(steps, edges)
        assert order.index("s1") < order.index("s2")
        assert order.index("s1") < order.index("s3")
        assert len(order) == 3

    def test_diamond_shape(self):
        # s1 → s2, s1 → s3, s2 → s4, s3 → s4
        steps = [_step("s1"), _step("s2"), _step("s3"), _step("s4")]
        edges = [
            _edge("e1", "s1", "s2"), _edge("e2", "s1", "s3"),
            _edge("e3", "s2", "s4"), _edge("e4", "s3", "s4"),
        ]
        order = topological_sort(steps, edges)
        assert order.index("s1") < order.index("s2")
        assert order.index("s1") < order.index("s3")
        assert order.index("s2") < order.index("s4")
        assert order.index("s3") < order.index("s4")

    def test_cycle_raises(self):
        steps = [_step("s1"), _step("s2")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s2", "s1")]
        with pytest.raises(WorkflowValidationError, match="[Cc]ycle"):
            topological_sort(steps, edges)

    def test_self_loop_raises(self):
        steps = [_step("s1")]
        edges = [_edge("e1", "s1", "s1")]
        with pytest.raises(WorkflowValidationError, match="[Cc]ycle"):
            topological_sort(steps, edges)

    def test_three_node_cycle_raises(self):
        steps = [_step("s1"), _step("s2"), _step("s3")]
        edges = [
            _edge("e1", "s1", "s2"),
            _edge("e2", "s2", "s3"),
            _edge("e3", "s3", "s1"),
        ]
        with pytest.raises(WorkflowValidationError):
            topological_sort(steps, edges)

    def test_cycle_error_names_involved_nodes(self):
        steps = [_step("s1"), _step("s2")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s2", "s1")]
        exc = pytest.raises(WorkflowValidationError, topological_sort, steps, edges)
        # violations list should contain the cycle node IDs
        assert exc.value.violations
        assert any("s1" in v or "s2" in v for v in exc.value.violations)


# ─────────────────────────────────────────────────────────────────────────────
# dag.py — entry / exit points
# ─────────────────────────────────────────────────────────────────────────────


class TestEntryExitPoints:
    def test_entry_linear(self):
        steps = [_step("s1"), _step("s2")]
        edges = [_edge("e1", "s1", "s2")]
        assert get_entry_points(steps, edges) == ["s1"]

    def test_exit_linear(self):
        steps = [_step("s1"), _step("s2")]
        edges = [_edge("e1", "s1", "s2")]
        assert get_exit_points(steps, edges) == ["s2"]

    def test_single_step_is_both_entry_and_exit(self):
        steps = [_step("s1")]
        assert get_entry_points(steps, []) == ["s1"]
        assert get_exit_points(steps, []) == ["s1"]

    def test_multiple_entries(self):
        steps = [_step("s1"), _step("s2"), _step("s3")]
        edges = [_edge("e1", "s1", "s3"), _edge("e2", "s2", "s3")]
        entries = get_entry_points(steps, edges)
        assert set(entries) == {"s1", "s2"}

    def test_multiple_exits(self):
        steps = [_step("s1"), _step("s2"), _step("s3")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s1", "s3")]
        exits = get_exit_points(steps, edges)
        assert set(exits) == {"s2", "s3"}


# ─────────────────────────────────────────────────────────────────────────────
# dag.py — children / parents / parallel group
# ─────────────────────────────────────────────────────────────────────────────


class TestGraphTraversal:
    def test_get_children(self):
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s1", "s3")]
        children = get_children("s1", edges)
        assert {child_id for child_id, _ in children} == {"s2", "s3"}

    def test_get_children_returns_edge_object(self):
        edge = _edge("e1", "s1", "s2", EdgeType.CONDITIONAL)
        children = get_children("s1", [edge])
        assert len(children) == 1
        assert children[0][1].edge_type == EdgeType.CONDITIONAL

    def test_get_children_empty(self):
        assert get_children("s1", []) == []

    def test_get_parents(self):
        edges = [_edge("e1", "s1", "s3"), _edge("e2", "s2", "s3")]
        parents = get_parents("s3", edges)
        assert {parent_id for parent_id, _ in parents} == {"s1", "s2"}

    def test_get_parents_returns_edge_object(self):
        edge = _edge("e1", "s1", "s2", EdgeType.LOOP_BACK)
        parents = get_parents("s2", [edge])
        assert parents[0][1].edge_type == EdgeType.LOOP_BACK

    def test_get_parallel_group_no_parents(self):
        steps = [_step("s1"), _step("s2")]
        result = get_parallel_group("s1", steps, [])
        assert result == ["s1"]

    def test_get_parallel_group_same_parent(self):
        steps = [_step("s1"), _step("s2"), _step("s3")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s1", "s3")]
        group = get_parallel_group("s2", steps, edges)
        assert set(group) == {"s2", "s3"}

    def test_get_parallel_group_different_parents(self):
        # s2 has parent s1; s3 has parent s4 — not in same group
        steps = [_step("s1"), _step("s2"), _step("s3"), _step("s4")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s4", "s3")]
        group = get_parallel_group("s2", steps, edges)
        assert group == ["s2"]

    def test_get_parallel_group_symmetric(self):
        # Asking from s2's perspective vs s3's perspective gives same group
        steps = [_step("s1"), _step("s2"), _step("s3")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s1", "s3")]
        group_from_s2 = set(get_parallel_group("s2", steps, edges))
        group_from_s3 = set(get_parallel_group("s3", steps, edges))
        assert group_from_s2 == group_from_s3 == {"s2", "s3"}


# ─────────────────────────────────────────────────────────────────────────────
# dag.py — evaluate_condition
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluateCondition:
    def test_empty_condition_is_true(self):
        assert evaluate_condition("", {}) is True
        assert evaluate_condition("  ", {}) is True

    def test_none_condition_is_true(self):
        # Defensive: test with None-like empty string
        assert evaluate_condition("", {"key": "val"}) is True

    def test_literal_true(self):
        assert evaluate_condition("true", {}) is True
        assert evaluate_condition("True", {}) is True
        assert evaluate_condition("yes", {}) is True
        assert evaluate_condition("1", {}) is True

    def test_literal_false(self):
        assert evaluate_condition("false", {}) is False
        assert evaluate_condition("False", {}) is False
        assert evaluate_condition("no", {}) is False
        assert evaluate_condition("0", {}) is False

    def test_equality_with_context(self):
        ctx = {"status": "completed"}
        assert evaluate_condition("${status} == 'completed'", ctx) is True
        assert evaluate_condition("${status} == 'failed'", ctx) is False

    def test_inequality_with_context(self):
        ctx = {"status": "running"}
        assert evaluate_condition("${status} != 'completed'", ctx) is True
        assert evaluate_condition("${status} != 'running'", ctx) is False

    def test_numeric_comparison(self):
        ctx = {"count": 10}
        assert evaluate_condition("${count} > 5", ctx) is True
        assert evaluate_condition("${count} < 5", ctx) is False
        assert evaluate_condition("${count} >= 10", ctx) is True
        assert evaluate_condition("${count} <= 9", ctx) is False
        assert evaluate_condition("${count} == 10", ctx) is True

    def test_boolean_and(self):
        ctx = {"a": 1, "b": 2}
        assert evaluate_condition("${a} == 1 and ${b} == 2", ctx) is True
        assert evaluate_condition("${a} == 1 and ${b} == 99", ctx) is False

    def test_boolean_or(self):
        ctx = {"x": "hello"}
        assert evaluate_condition("${x} == 'hello' or ${x} == 'world'", ctx) is True
        assert evaluate_condition("${x} == 'foo' or ${x} == 'bar'", ctx) is False

    def test_not_operator(self):
        ctx = {"flag": False}
        assert evaluate_condition("not ${flag}", ctx) is True
        ctx2 = {"flag": True}
        assert evaluate_condition("not ${flag}", ctx2) is False

    def test_unresolved_variable_is_falsy(self):
        assert evaluate_condition("${missing}", {}) is False

    def test_nested_key(self):
        ctx = {"result": {"status": "ok"}}
        assert evaluate_condition("${result.status} == 'ok'", ctx) is True
        assert evaluate_condition("${result.status} == 'fail'", ctx) is False

    def test_membership_in_tuple(self):
        ctx = {"role": "admin"}
        assert evaluate_condition("${role} in ('admin', 'superuser')", ctx) is True
        assert evaluate_condition("${role} in ('viewer', 'editor')", ctx) is False

    def test_not_in_operator(self):
        ctx = {"status": "pending"}
        assert evaluate_condition("${status} not in ('completed', 'failed')", ctx) is True
        assert evaluate_condition("${status} not in ('pending', 'running')", ctx) is False

    def test_chained_comparisons(self):
        # Python supports chained: 1 < x < 10
        ctx = {"score": 5}
        assert evaluate_condition("1 < ${score} < 10", ctx) is True
        assert evaluate_condition("6 < ${score} < 10", ctx) is False

    def test_syntax_error_raises(self):
        with pytest.raises(WorkflowValidationError):
            evaluate_condition("${x} ===== 'bad'", {"x": 1})

    def test_disallowed_function_call_raises(self):
        with pytest.raises(WorkflowValidationError):
            evaluate_condition("len('hello') == 5", {})

    def test_disallowed_attribute_access_raises(self):
        with pytest.raises(WorkflowValidationError):
            evaluate_condition("'hello'.__class__ == str", {})

    def test_disallowed_subscript_raises(self):
        with pytest.raises(WorkflowValidationError):
            evaluate_condition("x[0] == 1", {"x": [1, 2, 3]})

    def test_string_context_value(self):
        ctx = {"env": "production"}
        assert evaluate_condition("${env} == 'production'", ctx) is True
        assert evaluate_condition("${env} == 'staging'", ctx) is False

    def test_bool_context_value(self):
        ctx = {"success": True}
        assert evaluate_condition("${success}", ctx) is True
        ctx2 = {"success": False}
        assert evaluate_condition("${success}", ctx2) is False


# ─────────────────────────────────────────────────────────────────────────────
# validator.py
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkflowValidator:
    def test_valid_linear_workflow(self, validator):
        steps = [_step("s1"), _step("s2")]
        edges = [_edge("e1", "s1", "s2")]
        errors = validator.validate(_wf(steps, edges))
        assert errors == []

    def test_empty_workflow_is_valid(self, validator):
        errors = validator.validate(_wf([], []))
        assert errors == []

    def test_single_step_is_valid(self, validator):
        errors = validator.validate(_wf([_step("s1")], []))
        assert errors == []

    def test_returns_list_of_strings_on_error(self, validator):
        steps = [_step("s1"), _step("s2")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s2", "s1")]
        errors = validator.validate(_wf(steps, edges))
        assert isinstance(errors, list)
        assert len(errors) >= 1
        assert all(isinstance(e, str) for e in errors)

    # Check 1: cycle
    def test_cycle_detected(self, validator):
        steps = [_step("s1"), _step("s2")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s2", "s1")]
        errors = validator.validate(_wf(steps, edges))
        assert any("cycle" in e.lower() for e in errors)

    def test_self_loop_detected(self, validator):
        steps = [_step("s1")]
        edges = [_edge("e1", "s1", "s1")]
        errors = validator.validate(_wf(steps, edges))
        assert any("cycle" in e.lower() for e in errors)

    # Check 2: connectivity
    def test_disconnected_step(self, validator):
        # s1 → s2, s3 is isolated
        steps = [_step("s1"), _step("s2"), _step("s3")]
        edges = [_edge("e1", "s1", "s2")]
        errors = validator.validate(_wf(steps, edges))
        assert any("s3" in e and "disconnected" in e for e in errors)

    def test_two_isolated_components(self, validator):
        # s1→s2 and s3→s4 are separate — both s3 and s4 disconnected from s1/s2
        steps = [_step("s1"), _step("s2"), _step("s3"), _step("s4")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s3", "s4")]
        errors = validator.validate(_wf(steps, edges))
        disconnected = [e for e in errors if "disconnected" in e]
        # s3 and s4 are disconnected from s1
        assert len(disconnected) >= 1

    # Check 3: edge validity
    def test_invalid_edge_source(self, validator):
        steps = [_step("s1")]
        edges = [_edge("e1", "ghost", "s1")]
        errors = validator.validate(_wf(steps, edges))
        assert any("ghost" in e for e in errors)

    def test_invalid_edge_target(self, validator):
        steps = [_step("s1")]
        edges = [_edge("e1", "s1", "ghost")]
        errors = validator.validate(_wf(steps, edges))
        assert any("ghost" in e for e in errors)

    def test_both_endpoints_invalid(self, validator):
        steps = [_step("s1")]
        edges = [_edge("e1", "ghost_src", "ghost_tgt")]
        errors = validator.validate(_wf(steps, edges))
        assert any("ghost_src" in e for e in errors)
        assert any("ghost_tgt" in e for e in errors)

    # Check 4: branch completeness
    def test_branch_too_few_edges(self, validator):
        steps = [_step("s1", step_type=StepType.BRANCH), _step("s2")]
        edges = [_edge("e1", "s1", "s2", EdgeType.DEFAULT)]
        errors = validator.validate(_wf(steps, edges))
        assert any("at least 2" in e for e in errors)

    def test_branch_missing_default_edge(self, validator):
        steps = [_step("s1", step_type=StepType.BRANCH), _step("s2"), _step("s3")]
        edges = [
            _edge("e1", "s1", "s2", EdgeType.CONDITIONAL),
            _edge("e2", "s1", "s3", EdgeType.CONDITIONAL),
        ]
        errors = validator.validate(_wf(steps, edges))
        assert any("default" in e for e in errors)

    def test_valid_branch(self, validator):
        # branch with conditional + default + convergence step
        steps = [
            _step("start"), _step("b", step_type=StepType.BRANCH),
            _step("yes"), _step("no"), _step("end"),
        ]
        edges = [
            _edge("e1", "start", "b"),
            _edge("e2", "b", "yes", EdgeType.CONDITIONAL),
            _edge("e3", "b", "no", EdgeType.DEFAULT),
            _edge("e4", "yes", "end"),
            _edge("e5", "no", "end"),
        ]
        errors = validator.validate(_wf(steps, edges))
        assert errors == []

    # Check 5: loop safety
    def test_loop_missing_loop_back(self, validator):
        steps = [_step("s1", step_type=StepType.LOOP), _step("s2")]
        edges = [_edge("e1", "s1", "s2", EdgeType.DEFAULT)]
        errors = validator.validate(_wf(steps, edges))
        assert any("loop_back" in e for e in errors)

    def test_loop_missing_default_exit(self, validator):
        steps = [_step("s1", step_type=StepType.LOOP), _step("s2")]
        edges = [_edge("e1", "s1", "s2", EdgeType.LOOP_BACK)]
        errors = validator.validate(_wf(steps, edges))
        assert any("default" in e for e in errors)

    def test_valid_loop(self, validator):
        # loop_back → same step, default → exit step
        steps = [_step("s1", step_type=StepType.LOOP), _step("s2")]
        edges = [
            _edge("e1", "s1", "s1", EdgeType.LOOP_BACK),
            _edge("e2", "s1", "s2", EdgeType.DEFAULT),
        ]
        errors = validator.validate(_wf(steps, edges))
        assert errors == []

    # Check 6: parallel convergence (warning only)
    def test_parallel_without_convergence_emits_warning(self, validator):
        # s0 → s1(PARALLEL), s1 has no outgoing edges
        steps = [_step("s0"), _step("s1", step_type=StepType.PARALLEL)]
        edges = [_edge("e1", "s0", "s1")]
        errors = validator.validate(_wf(steps, edges))
        warnings = [e for e in errors if e.startswith("WARNING:")]
        assert len(warnings) >= 1
        assert any("parallel" in w.lower() or "converge" in w.lower() for w in warnings)

    def test_parallel_warning_is_not_hard_error(self, validator):
        steps = [_step("s0"), _step("s1", step_type=StepType.PARALLEL)]
        edges = [_edge("e1", "s0", "s1")]
        errors = validator.validate(_wf(steps, edges))
        hard = [e for e in errors if not e.startswith("WARNING:")]
        assert hard == []

    def test_parallel_duplicate_warning_deduplicated(self, validator):
        # Two PARALLEL steps sharing same parent — should produce only ONE warning,
        # not two (one per PARALLEL step iteration).
        steps = [
            _step("s0"),
            _step("s1", step_type=StepType.PARALLEL),
            _step("s2", step_type=StepType.PARALLEL),
        ]
        edges = [
            _edge("e1", "s0", "s1"),
            _edge("e2", "s0", "s2"),
        ]
        errors = validator.validate(_wf(steps, edges))
        warnings = [e for e in errors if e.startswith("WARNING:")]
        assert len(warnings) == 1  # deduplicated — same group, one warning

    def test_parallel_with_convergence_no_warning(self, validator):
        # s0 → s1(PAR), s0 → s2(PAR), both → s3 (convergence)
        steps = [
            _step("s0"), _step("s1", step_type=StepType.PARALLEL),
            _step("s2", step_type=StepType.PARALLEL), _step("s3"),
        ]
        edges = [
            _edge("e1", "s0", "s1"), _edge("e2", "s0", "s2"),
            _edge("e3", "s1", "s3"), _edge("e4", "s2", "s3"),
        ]
        errors = validator.validate(_wf(steps, edges))
        warnings = [e for e in errors if e.startswith("WARNING:")]
        assert warnings == []

    # Check 7: step limit
    def test_step_limit_exceeded(self, validator):
        steps = [_step(f"s{i}") for i in range(5)]
        errors = validator.validate(_wf(steps, []), max_steps=3)
        assert any("maximum" in e for e in errors)

    def test_step_limit_at_boundary_is_valid(self, validator):
        steps = [_step(f"s{i}") for i in range(3)]
        errors = validator.validate(_wf(steps, []), max_steps=3)
        assert not any("maximum" in e for e in errors)

    def test_step_limit_error_contains_counts(self, validator):
        steps = [_step(f"s{i}") for i in range(10)]
        errors = validator.validate(_wf(steps, []), max_steps=5)
        limit_errors = [e for e in errors if "maximum" in e]
        assert len(limit_errors) == 1
        assert "10" in limit_errors[0]
        assert "5" in limit_errors[0]

    # Check 8: tool references (with mock registry)
    def test_unknown_tool_is_error(self, validator):
        registry = MagicMock()
        registry.list_tools.return_value = []

        steps = [_step("s1", tool_name="ghost_tool")]
        errors = validator.validate(_wf(steps, []), registry=registry)
        assert any("ghost_tool" in e for e in errors)

    def test_known_tool_passes(self, validator):
        tool = MagicMock()
        tool.name = "knowledge_search"
        registry = MagicMock()
        registry.list_tools.return_value = [tool]

        steps = [_step("s1", tool_name="knowledge_search")]
        errors = validator.validate(_wf(steps, []), registry=registry)
        assert not any("knowledge_search" in e for e in errors)

    def test_step_without_tool_skips_check(self, validator):
        # tool_name=None → no tool check needed
        registry = MagicMock()
        registry.list_tools.return_value = []

        steps = [_step("s1")]  # tool_name=None by default
        errors = validator.validate(_wf(steps, []), registry=registry)
        assert not any("tool" in e.lower() for e in errors)

    # Check 9: persona references (with mock persona_manager)
    def test_unknown_persona_is_error(self, validator):
        pm = MagicMock()
        pm.get_persona.return_value = None

        steps = [_step("s1")]
        errors = validator.validate(_wf(steps, []), persona_manager=pm)
        assert any("persona" in e.lower() for e in errors)

    def test_known_persona_passes(self, validator):
        pm = MagicMock()
        pm.get_persona.return_value = MagicMock()  # non-None → found

        steps = [_step("s1")]
        errors = validator.validate(_wf(steps, []), persona_manager=pm)
        assert not any("persona" in e.lower() and "not found" in e.lower() for e in errors)

    def test_all_checks_run_even_after_first_failure(self, validator):
        # Cycle + disconnected + edge validity + step limit — all should appear
        steps = [_step("s1"), _step("s2"), _step("s3")]
        edges = [
            _edge("e1", "s1", "s2"),
            _edge("e2", "s2", "s1"),         # cycle
            _edge("e3", "s1", "ghost_tgt"),  # bad edge target
        ]
        errors = validator.validate(_wf(steps, edges), max_steps=2)
        # Should have: cycle error, step limit error, edge validity error
        assert any("cycle" in e.lower() for e in errors)
        assert any("maximum" in e for e in errors)
        assert any("ghost_tgt" in e for e in errors)


# ─────────────────────────────────────────────────────────────────────────────
# manager.py
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkflowManager:
    @pytest.mark.asyncio
    async def test_create_returns_draft_version_1(self, manager):
        wf = await manager.create(
            tenant_id=TENANT,
            name="My Workflow",
            steps=[_step("s1")],
            edges=[],
        )
        assert wf.status == WorkflowStatus.DRAFT
        assert wf.version == 1
        assert wf.name == "My Workflow"
        assert wf.tenant_id == TENANT
        assert wf.id  # has an ID

    @pytest.mark.asyncio
    async def test_create_stores_all_fields(self, manager):
        wf = await manager.create(
            tenant_id=TENANT,
            name="Full WF",
            description="Test description",
            steps=[_step("s1")],
            edges=[],
            created_by="user-1",
            tags=["tag1", "tag2"],
            settings={"timeout_seconds": 120},
            trigger_config={"type": "manual"},
        )
        assert wf.description == "Test description"
        assert wf.created_by == "user-1"
        assert wf.tags == ["tag1", "tag2"]
        assert wf.settings == {"timeout_seconds": 120}
        assert wf.trigger_config == {"type": "manual"}

    @pytest.mark.asyncio
    async def test_create_invalid_raises(self, manager):
        # cycle → validation error
        steps = [_step("s1"), _step("s2")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s2", "s1")]
        with pytest.raises(WorkflowValidationError):
            await manager.create(tenant_id=TENANT, name="Bad WF", steps=steps, edges=edges)

    @pytest.mark.asyncio
    async def test_create_calls_repository(self):
        """Repository.create_workflow is called when a repo is provided."""
        repo = MagicMock()
        repo.create_workflow = AsyncMock(return_value=None)
        mgr = WorkflowManager(repository=repo)

        await mgr.create(tenant_id=TENANT, name="WF")
        repo.create_workflow.assert_called_once()
        # First positional arg should be the WorkflowDefinition
        wf_arg = repo.create_workflow.call_args[0][0]
        assert isinstance(wf_arg, WorkflowDefinition)
        assert wf_arg.tenant_id == TENANT

    @pytest.mark.asyncio
    async def test_repository_not_implemented_silently_ignored(self):
        """NotImplementedError from repository is swallowed — workflow still created."""
        repo = MagicMock()
        repo.create_workflow = AsyncMock(side_effect=NotImplementedError("stub"))
        mgr = WorkflowManager(repository=repo)

        wf = await mgr.create(tenant_id=TENANT, name="WF")
        assert wf.id  # created in-memory successfully
        repo.create_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_existing(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="Fetch Me")
        fetched = await manager.get(wf.id, TENANT)
        assert fetched.id == wf.id
        assert fetched.name == "Fetch Me"

    @pytest.mark.asyncio
    async def test_get_not_found(self, manager):
        with pytest.raises(WorkflowNotFound) as exc:
            await manager.get("does-not-exist", TENANT)
        assert exc.value.workflow_id == "does-not-exist"

    @pytest.mark.asyncio
    async def test_get_wrong_tenant_raises(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="Private WF")
        with pytest.raises(WorkflowNotFound):
            await manager.get(wf.id, "other-tenant")

    @pytest.mark.asyncio
    async def test_update_metadata_no_version_bump(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="Original")
        updated = await manager.update(wf.id, TENANT, name="Renamed")
        assert updated.version == 1
        assert updated.name == "Renamed"

    @pytest.mark.asyncio
    async def test_update_steps_increments_version(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="WF")
        new_steps = [_step("s1"), _step("s2")]
        new_edges = [_edge("e1", "s1", "s2")]
        updated = await manager.update(wf.id, TENANT, steps=new_steps, edges=new_edges)
        assert updated.version == 2

    @pytest.mark.asyncio
    async def test_update_edges_only_increments_version(self, manager):
        # Adding an edge to an existing two-step workflow
        wf = await manager.create(
            tenant_id=TENANT, name="WF",
            steps=[_step("s1"), _step("s2")],
            edges=[_edge("e1", "s1", "s2")],
        )
        # Add another edge type (still valid — s1→s2 exists, add e.g. same but different type)
        updated = await manager.update(
            wf.id, TENANT,
            edges=[_edge("e1", "s1", "s2", EdgeType.DEFAULT)],
        )
        assert updated.version == 2

    @pytest.mark.asyncio
    async def test_update_invalid_steps_raises(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="WF")
        steps = [_step("s1"), _step("s2")]
        edges = [_edge("e1", "s1", "s2"), _edge("e2", "s2", "s1")]  # cycle
        with pytest.raises(WorkflowValidationError):
            await manager.update(wf.id, TENANT, steps=steps, edges=edges)

    @pytest.mark.asyncio
    async def test_update_preserves_non_updated_fields(self, manager):
        wf = await manager.create(
            tenant_id=TENANT, name="WF",
            description="original desc",
            tags=["keep-me"],
        )
        updated = await manager.update(wf.id, TENANT, name="New Name")
        assert updated.description == "original desc"
        assert updated.tags == ["keep-me"]

    @pytest.mark.asyncio
    async def test_activate_sets_active_status(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="WF")
        active = await manager.activate(wf.id, TENANT)
        assert active.status == WorkflowStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_activate_revalidates(self, manager):
        """activate() re-validates the workflow even if no steps changed."""
        wf = await manager.create(tenant_id=TENANT, name="WF", steps=[_step("s1")])
        # Manually corrupt the in-memory store (bypassing validation)
        corrupted = wf.model_copy(update={
            "steps": [_step("s1"), _step("s2")],  # disconnected s2
        })
        manager._store[wf.id] = corrupted
        with pytest.raises(WorkflowValidationError):
            await manager.activate(wf.id, TENANT)

    @pytest.mark.asyncio
    async def test_pause_sets_paused_status(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="WF")
        paused = await manager.pause(wf.id, TENANT)
        assert paused.status == WorkflowStatus.PAUSED

    @pytest.mark.asyncio
    async def test_archive_sets_archived_status(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="WF")
        archived = await manager.archive(wf.id, TENANT)
        assert archived.status == WorkflowStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_list_returns_tenant_workflows(self, manager):
        await manager.create(tenant_id=TENANT, name="WF A")
        await manager.create(tenant_id=TENANT, name="WF B")
        await manager.create(tenant_id="other", name="Other")
        results = await manager.list(TENANT)
        assert len(results) == 2
        assert all(wf.tenant_id == TENANT for wf in results)

    @pytest.mark.asyncio
    async def test_list_empty(self, manager):
        results = await manager.list("no-workflows-tenant")
        assert results == []

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="WF")
        await manager.activate(wf.id, TENANT)
        await manager.create(tenant_id=TENANT, name="Draft WF")
        active_list = await manager.list(TENANT, status=WorkflowStatus.ACTIVE)
        assert len(active_list) >= 1
        assert all(w.status == WorkflowStatus.ACTIVE for w in active_list)

    @pytest.mark.asyncio
    async def test_list_pagination_limit(self, manager):
        for i in range(5):
            await manager.create(tenant_id=TENANT, name=f"WF-{i}")
        page = await manager.list(TENANT, limit=2)
        assert len(page) == 2

    @pytest.mark.asyncio
    async def test_list_pagination_offset(self, manager):
        for i in range(5):
            await manager.create(tenant_id=TENANT, name=f"WF-{i}")
        all_wfs = await manager.list(TENANT)
        page1 = await manager.list(TENANT, limit=2, offset=0)
        page2 = await manager.list(TENANT, limit=2, offset=2)
        page3 = await manager.list(TENANT, limit=2, offset=4)
        # Pages are non-overlapping
        assert set(w.id for w in page1).isdisjoint(set(w.id for w in page2))
        # All pages together cover every workflow exactly once
        assert len(page1) + len(page2) + len(page3) == len(all_wfs) == 5

    @pytest.mark.asyncio
    async def test_get_version_history_tracks_versions(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="WF")
        await manager.update(wf.id, TENANT, steps=[_step("s1")])
        new_steps = [_step("s1"), _step("s2")]
        new_edges = [_edge("e1", "s1", "s2")]
        await manager.update(wf.id, TENANT, steps=new_steps, edges=new_edges)
        history = await manager.get_version_history(wf.id, TENANT)
        assert len(history) == 3  # v1 (create) + v2 + v3
        assert history[0].version == 1
        assert history[-1].version == 3

    @pytest.mark.asyncio
    async def test_get_version_history_wrong_tenant_raises(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="WF")
        with pytest.raises(WorkflowNotFound):
            await manager.get_version_history(wf.id, "wrong-tenant")

    @pytest.mark.asyncio
    async def test_rollback_creates_new_version(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="WF", steps=[_step("s1")])
        # v2 — add a connected step
        await manager.update(
            wf.id, TENANT,
            steps=[_step("s1"), _step("s2")],
            edges=[_edge("e1", "s1", "s2")],
        )
        # Rollback to v1
        rolled = await manager.rollback(wf.id, TENANT, target_version=1)
        assert rolled.version == 3
        assert len(rolled.steps) == 1

    @pytest.mark.asyncio
    async def test_rollback_step_content_matches_target(self, manager):
        """Step data in rollback matches original, not the intermediate version."""
        s1 = _step("step-a", name="Step A")
        wf = await manager.create(tenant_id=TENANT, name="WF", steps=[s1])
        original_step_ids = {s.id for s in wf.steps}

        # v2 — completely different steps
        await manager.update(
            wf.id, TENANT,
            steps=[_step("step-x", name="Step X"), _step("step-y", name="Step Y")],
            edges=[_edge("ex", "step-x", "step-y")],
        )
        # Rollback to v1
        rolled = await manager.rollback(wf.id, TENANT, target_version=1)
        rolled_step_ids = {s.id for s in rolled.steps}
        assert rolled_step_ids == original_step_ids

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_version_raises(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="WF")
        with pytest.raises(WorkflowNotFound):
            await manager.rollback(wf.id, TENANT, target_version=99)

    @pytest.mark.asyncio
    async def test_duplicate_creates_independent_copy(self, manager):
        wf = await manager.create(
            tenant_id=TENANT,
            name="Original",
            steps=[_step("s1"), _step("s2")],
            edges=[_edge("e1", "s1", "s2")],
        )
        copy = await manager.duplicate(wf.id, TENANT)
        assert copy.id != wf.id
        assert copy.version == 1
        assert copy.status == WorkflowStatus.DRAFT
        assert len(copy.steps) == 2

    @pytest.mark.asyncio
    async def test_duplicate_step_ids_are_remapped(self, manager):
        wf = await manager.create(
            tenant_id=TENANT,
            name="Original",
            steps=[_step("s1"), _step("s2")],
            edges=[_edge("e1", "s1", "s2")],
        )
        copy = await manager.duplicate(wf.id, TENANT)
        original_step_ids = {s.id for s in wf.steps}
        copy_step_ids = {s.id for s in copy.steps}
        assert copy_step_ids.isdisjoint(original_step_ids)

    @pytest.mark.asyncio
    async def test_duplicate_edge_integrity(self, manager):
        """After duplicate, edge source/target IDs reference steps in the copy, not originals."""
        wf = await manager.create(
            tenant_id=TENANT,
            name="Original",
            steps=[_step("s1"), _step("s2")],
            edges=[_edge("e1", "s1", "s2")],
        )
        copy = await manager.duplicate(wf.id, TENANT)
        copy_step_ids = {s.id for s in copy.steps}
        for edge in copy.edges:
            assert edge.source_step_id in copy_step_ids, (
                f"Edge source {edge.source_step_id!r} not found in copy's steps"
            )
            assert edge.target_step_id in copy_step_ids, (
                f"Edge target {edge.target_step_id!r} not found in copy's steps"
            )

    @pytest.mark.asyncio
    async def test_duplicate_custom_name(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="Original")
        copy = await manager.duplicate(wf.id, TENANT, new_name="Clone")
        assert copy.name == "Clone"

    @pytest.mark.asyncio
    async def test_duplicate_default_name_suffix(self, manager):
        wf = await manager.create(tenant_id=TENANT, name="My Workflow")
        copy = await manager.duplicate(wf.id, TENANT)
        assert "My Workflow" in copy.name
        assert "copy" in copy.name.lower()

    @pytest.mark.asyncio
    async def test_export_json_is_valid_json(self, manager):
        wf = await manager.create(
            tenant_id=TENANT,
            name="Export Me",
            steps=[_step("s1")],
        )
        json_str = await manager.export_json(wf.id, TENANT)
        data = json.loads(json_str)
        assert data["name"] == "Export Me"
        assert data["tenant_id"] == TENANT

    @pytest.mark.asyncio
    async def test_export_json_contains_steps_and_edges(self, manager):
        wf = await manager.create(
            tenant_id=TENANT,
            name="WF",
            steps=[_step("s1"), _step("s2")],
            edges=[_edge("e1", "s1", "s2")],
        )
        json_str = await manager.export_json(wf.id, TENANT)
        data = json.loads(json_str)
        assert len(data["steps"]) == 2
        assert len(data["edges"]) == 1

    @pytest.mark.asyncio
    async def test_import_json_creates_new_workflow(self, manager):
        wf = await manager.create(
            tenant_id=TENANT,
            name="Exported WF",
            steps=[_step("s1")],
        )
        json_str = await manager.export_json(wf.id, TENANT)
        imported = await manager.import_json(json_str, tenant_id=TENANT)
        assert imported.id != wf.id
        assert imported.version == 1
        assert imported.status == WorkflowStatus.DRAFT

    @pytest.mark.asyncio
    async def test_import_json_remaps_step_ids(self, manager):
        wf = await manager.create(
            tenant_id=TENANT,
            name="WF",
            steps=[_step("s1"), _step("s2")],
            edges=[_edge("e1", "s1", "s2")],
        )
        json_str = await manager.export_json(wf.id, TENANT)
        imported = await manager.import_json(json_str, tenant_id=TENANT)
        original_ids = {s.id for s in wf.steps}
        imported_ids = {s.id for s in imported.steps}
        assert imported_ids.isdisjoint(original_ids)

    @pytest.mark.asyncio
    async def test_import_json_edge_integrity(self, manager):
        """Imported workflow's edges reference its own remapped step IDs."""
        wf = await manager.create(
            tenant_id=TENANT,
            name="WF",
            steps=[_step("s1"), _step("s2")],
            edges=[_edge("e1", "s1", "s2")],
        )
        json_str = await manager.export_json(wf.id, TENANT)
        imported = await manager.import_json(json_str, tenant_id=TENANT)
        imported_step_ids = {s.id for s in imported.steps}
        for edge in imported.edges:
            assert edge.source_step_id in imported_step_ids
            assert edge.target_step_id in imported_step_ids

    @pytest.mark.asyncio
    async def test_import_invalid_json_raises(self, manager):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            await manager.import_json("not json at all", TENANT)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests — realistic end-to-end scenarios with real tool / persona names
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkflowSmoke:
    """
    Real-world workflow scenarios using actual NEXUS tool names and personas.
    These tests validate that the workflow definition layer works correctly
    for the patterns it will actually serve in production.
    """

    @pytest.mark.asyncio
    async def test_research_pipeline_full_lifecycle(self, manager):
        """
        Realistic 3-step research workflow:
          fetch_data (web_search) → analyze (knowledge_search) → summarize (ACTION)

        Tests: create → validate → activate → export JSON → import → verify parity.
        """
        steps = [
            _step("fetch", name="Fetch Web Data", tool_name="web_search"),
            _step("analyze", name="Analyze Results", tool_name="knowledge_search"),
            _step("summarize", name="Generate Summary"),
        ]
        edges = [
            _edge("e1", "fetch", "analyze", EdgeType.DEFAULT),
            _edge("e2", "analyze", "summarize", EdgeType.DEFAULT),
        ]

        # Create
        wf = await manager.create(
            tenant_id=TENANT,
            name="Research Pipeline",
            description="Fetch → analyze → summarize",
            steps=steps,
            edges=edges,
            tags=["research", "v1"],
            created_by="smoke-test",
        )
        assert wf.status == WorkflowStatus.DRAFT
        assert len(wf.steps) == 3
        assert len(wf.edges) == 2

        # Activate
        active = await manager.activate(wf.id, TENANT)
        assert active.status == WorkflowStatus.ACTIVE

        # Verify it appears in the active list
        active_list = await manager.list(TENANT, status=WorkflowStatus.ACTIVE)
        assert any(w.id == wf.id for w in active_list)

        # Export → import: imported copy is structurally equivalent
        json_str = await manager.export_json(wf.id, TENANT)
        imported = await manager.import_json(json_str, TENANT)
        assert imported.name == wf.name
        assert len(imported.steps) == len(wf.steps)
        assert len(imported.edges) == len(wf.edges)
        # Imported step IDs are remapped (no collision)
        assert {s.id for s in imported.steps}.isdisjoint({s.id for s in wf.steps})

        # Pause → Archive lifecycle
        paused = await manager.pause(wf.id, TENANT)
        assert paused.status == WorkflowStatus.PAUSED
        archived = await manager.archive(wf.id, TENANT)
        assert archived.status == WorkflowStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_branch_workflow_with_condition_evaluation(self, manager):
        """
        Branching workflow that routes based on step output:
          check_status (BRANCH) → [status==ok → success, default → fallback]

        Tests: branch validation + condition evaluation with real contexts.
        """
        steps = [
            _step("check", name="Check API Status", step_type=StepType.BRANCH),
            _step("success", name="Handle Success"),
            _step("fallback", name="Handle Failure"),
            _step("done", name="Finalize"),
        ]
        edges = [
            _edge("e1", "check", "success", EdgeType.CONDITIONAL,
                  condition="${status} == 'ok'"),
            _edge("e2", "check", "fallback", EdgeType.DEFAULT),
            _edge("e3", "success", "done", EdgeType.DEFAULT),
            _edge("e4", "fallback", "done", EdgeType.DEFAULT),
        ]

        wf = await manager.create(
            tenant_id=TENANT,
            name="Status Check Branch",
            steps=steps, edges=edges,
        )
        assert wf.status == WorkflowStatus.DRAFT

        # Evaluate the conditional edge with real context
        ok_ctx = {"status": "ok"}
        error_ctx = {"status": "error"}

        ok_edge = next(e for e in wf.edges if e.edge_type == EdgeType.CONDITIONAL)
        assert ok_edge.condition is not None

        assert evaluate_condition(ok_edge.condition, ok_ctx) is True
        assert evaluate_condition(ok_edge.condition, error_ctx) is False

        # Default edge (no condition) is always true
        default_edge = next(e for e in wf.edges if e.edge_type == EdgeType.DEFAULT
                           and e.source_step_id == "check")
        assert evaluate_condition(default_edge.condition or "", {}) is True

    @pytest.mark.asyncio
    async def test_loop_workflow_valid(self, manager):
        """
        Loop that retries until a condition is met:
          retry_step (LOOP) → [loop_back → retry_step, default → done]
        """
        steps = [
            _step("retry", name="Retry API Call", step_type=StepType.LOOP),
            _step("done", name="Success"),
        ]
        edges = [
            _edge("loop", "retry", "retry", EdgeType.LOOP_BACK),
            _edge("exit", "retry", "done", EdgeType.DEFAULT),
        ]
        wf = await manager.create(
            tenant_id=TENANT,
            name="Retry Loop",
            steps=steps, edges=edges,
        )
        assert wf.status == WorkflowStatus.DRAFT

        # Validate directly — should be clean
        validator = WorkflowValidator()
        errors = validator.validate(wf)
        assert errors == []

    @pytest.mark.asyncio
    async def test_version_rollback_scenario(self, manager):
        """
        Real rollback scenario:
          v1: 2-step pipeline → v2: 4-step pipeline → rollback to v1
        Tests that rollback restores exact step content, not just count.
        """
        v1_steps = [
            _step("ingest", name="Ingest Data"),
            _step("output", name="Output Results"),
        ]
        v1_edges = [_edge("e1", "ingest", "output")]

        wf = await manager.create(
            tenant_id=TENANT,
            name="Evolving Pipeline",
            steps=v1_steps, edges=v1_edges,
        )
        v1_step_ids = {s.id for s in wf.steps}

        # Evolve to v2 (more steps)
        v2_steps = [
            _step("ingest", name="Ingest Data"),
            _step("clean", name="Clean Data"),
            _step("transform", name="Transform Data"),
            _step("output", name="Output Results"),
        ]
        v2_edges = [
            _edge("e1", "ingest", "clean"),
            _edge("e2", "clean", "transform"),
            _edge("e3", "transform", "output"),
        ]
        v2 = await manager.update(wf.id, TENANT, steps=v2_steps, edges=v2_edges)
        assert v2.version == 2
        assert len(v2.steps) == 4

        # Rollback to v1
        rolled = await manager.rollback(wf.id, TENANT, target_version=1)
        assert rolled.version == 3
        assert len(rolled.steps) == 2
        rolled_step_ids = {s.id for s in rolled.steps}
        assert rolled_step_ids == v1_step_ids  # exact same step objects restored

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, manager):
        """Workflows from different tenants are completely isolated."""
        tenant_a = "tenant-alpha"
        tenant_b = "tenant-beta"

        wf_a = await manager.create(tenant_id=tenant_a, name="Alpha WF")
        wf_b = await manager.create(tenant_id=tenant_b, name="Beta WF")

        # Each tenant sees only their own workflows
        a_list = await manager.list(tenant_a)
        b_list = await manager.list(tenant_b)
        assert all(w.tenant_id == tenant_a for w in a_list)
        assert all(w.tenant_id == tenant_b for w in b_list)

        # Cross-tenant get raises WorkflowNotFound
        with pytest.raises(WorkflowNotFound):
            await manager.get(wf_a.id, tenant_b)
        with pytest.raises(WorkflowNotFound):
            await manager.get(wf_b.id, tenant_a)

    @pytest.mark.asyncio
    async def test_complex_dag_with_diamond_and_branch(self, manager):
        """
        Complex workflow: diamond + branch structure.

        start → [fetch_a, fetch_b] → merge → check(BRANCH) → [ok → finish, default → error_handler]
        """
        steps = [
            _step("start", name="Start"),
            _step("fetch_a", name="Fetch Source A", tool_name="web_search"),
            _step("fetch_b", name="Fetch Source B", tool_name="knowledge_search"),
            _step("merge", name="Merge Results"),
            _step("check", name="Quality Check", step_type=StepType.BRANCH),
            _step("finish", name="Output"),
            _step("error_handler", name="Handle Error"),
        ]
        edges = [
            _edge("e1", "start", "fetch_a"),
            _edge("e2", "start", "fetch_b"),
            _edge("e3", "fetch_a", "merge"),
            _edge("e4", "fetch_b", "merge"),
            _edge("e5", "merge", "check"),
            _edge("e6", "check", "finish", EdgeType.CONDITIONAL,
                  condition="${quality_score} >= 80"),
            _edge("e7", "check", "error_handler", EdgeType.DEFAULT),
        ]

        wf = await manager.create(
            tenant_id=TENANT,
            name="Complex Research DAG",
            steps=steps, edges=edges,
        )
        assert wf.status == WorkflowStatus.DRAFT

        # Validate directly
        validator = WorkflowValidator()
        errors = validator.validate(wf)
        assert errors == []

        # Branch condition evaluation
        cond = "${quality_score} >= 80"
        assert evaluate_condition(cond, {"quality_score": 85}) is True
        assert evaluate_condition(cond, {"quality_score": 75}) is False

        # Activate the complex workflow
        active = await manager.activate(wf.id, TENANT)
        assert active.status == WorkflowStatus.ACTIVE
