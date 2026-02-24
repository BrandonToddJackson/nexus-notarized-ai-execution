"""Phase 30: Tests for DAG utility functions."""

import pytest

from nexus.exceptions import WorkflowValidationError
from nexus.types import EdgeType, StepType, WorkflowEdge, WorkflowStep
from nexus.workflows.dag import (
    evaluate_condition,
    get_children,
    get_entry_points,
    get_exit_points,
    get_parents,
    topological_sort,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def linear_steps_edges():
    """Simple 2-step linear graph: step_a → step_b."""
    wf_id = "wf-dag-test"
    step_a = WorkflowStep(
        id="step-a", workflow_id=wf_id, step_type=StepType.ACTION, name="step_a"
    )
    step_b = WorkflowStep(
        id="step-b", workflow_id=wf_id, step_type=StepType.ACTION, name="step_b"
    )
    edge = WorkflowEdge(
        id="edge-1",
        workflow_id=wf_id,
        source_step_id="step-a",
        target_step_id="step-b",
    )
    return [step_a, step_b], [edge]


@pytest.fixture
def cyclic_steps_edges():
    """Cyclic graph: step_a → step_b → step_a."""
    wf_id = "wf-cycle"
    step_a = WorkflowStep(
        id="step-a", workflow_id=wf_id, step_type=StepType.ACTION, name="step_a"
    )
    step_b = WorkflowStep(
        id="step-b", workflow_id=wf_id, step_type=StepType.ACTION, name="step_b"
    )
    edges = [
        WorkflowEdge(
            id="e-ab", workflow_id=wf_id,
            source_step_id="step-a", target_step_id="step-b",
        ),
        WorkflowEdge(
            id="e-ba", workflow_id=wf_id,
            source_step_id="step-b", target_step_id="step-a",
        ),
    ]
    return [step_a, step_b], edges


# ── Tests ────────────────────────────────────────────────────────────────────


def test_topological_sort_linear(linear_steps_edges):
    """Topological sort of a 2-step linear DAG returns [step_a, step_b]."""
    steps, edges = linear_steps_edges
    order = topological_sort(steps, edges)
    assert order == ["step-a", "step-b"]


def test_topological_sort_cycle_raises(cyclic_steps_edges):
    """Topological sort raises WorkflowValidationError on a cycle."""
    steps, edges = cyclic_steps_edges
    with pytest.raises(WorkflowValidationError, match="[Cc]ycle"):
        topological_sort(steps, edges)


def test_get_entry_points(linear_steps_edges):
    """Entry points are steps with no incoming edges."""
    steps, edges = linear_steps_edges
    entries = get_entry_points(steps, edges)
    assert entries == ["step-a"]


def test_get_exit_points(linear_steps_edges):
    """Exit points are steps with no outgoing edges."""
    steps, edges = linear_steps_edges
    exits = get_exit_points(steps, edges)
    assert exits == ["step-b"]


def test_get_children(linear_steps_edges):
    """step_a has one child: step_b."""
    _, edges = linear_steps_edges
    children = get_children("step-a", edges)
    assert len(children) == 1
    target_id, edge = children[0]
    assert target_id == "step-b"


def test_get_parents(linear_steps_edges):
    """step_b has one parent: step_a."""
    _, edges = linear_steps_edges
    parents = get_parents("step-b", edges)
    assert len(parents) == 1
    source_id, edge = parents[0]
    assert source_id == "step-a"


def test_evaluate_condition_equals_true():
    """Simple equality check evaluates to True."""
    assert evaluate_condition("${x} == 1", {"x": 1}) is True


def test_evaluate_condition_in_operator():
    """Membership 'in' operator with tuple RHS."""
    result = evaluate_condition(
        "${status} in ('done', 'ok')", {"status": "done"}
    )
    assert result is True
