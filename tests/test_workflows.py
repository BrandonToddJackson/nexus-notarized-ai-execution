"""Phase 30: Tests for WorkflowManager and WorkflowValidator."""

import json

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
from nexus.workflows.dag import evaluate_condition, topological_sort
from nexus.workflows.manager import WorkflowManager
from nexus.workflows.validator import WorkflowValidator

TENANT_A = "tenant-alpha-001"


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def manager():
    return WorkflowManager(validator=WorkflowValidator())


# ── WorkflowManager CRUD ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_workflow(manager):
    """Create a minimal workflow and verify defaults."""
    wf = await manager.create(TENANT_A, "My Workflow")
    assert wf.id
    assert wf.name == "My Workflow"
    assert wf.status == WorkflowStatus.DRAFT
    assert wf.version == 1
    assert wf.tenant_id == TENANT_A


@pytest.mark.asyncio
async def test_workflow_invalid_dag(cyclic_workflow_def):
    """Validator returns non-empty error list for a cyclic graph."""
    validator = WorkflowValidator()
    errors = validator.validate(cyclic_workflow_def)
    hard_errors = [e for e in errors if not e.startswith("WARNING:")]
    assert len(hard_errors) > 0
    assert any("ycle" in e for e in hard_errors)


@pytest.mark.asyncio
async def test_workflow_versioning(manager):
    """Updating name alone does not bump version; updating steps does."""
    wf = await manager.create(TENANT_A, "V1")
    assert wf.version == 1

    # Metadata-only update: no version bump
    wf2 = await manager.update(wf.id, TENANT_A, name="V1-renamed")
    assert wf2.version == 1
    assert wf2.name == "V1-renamed"

    # Structural update: version bumps
    new_step = WorkflowStep(
        id="s1", workflow_id=wf.id, step_type=StepType.ACTION, name="first"
    )
    wf3 = await manager.update(wf.id, TENANT_A, steps=[new_step])
    assert wf3.version == 2


@pytest.mark.asyncio
async def test_activate_workflow(manager, linear_workflow_def):
    """Activate a workflow with valid steps and edges."""
    wf = await manager.create(
        TENANT_A,
        "Activatable",
        steps=linear_workflow_def.steps,
        edges=linear_workflow_def.edges,
    )
    activated = await manager.activate(wf.id, TENANT_A)
    assert activated.status == WorkflowStatus.ACTIVE


@pytest.mark.asyncio
async def test_activate_empty_workflow(manager):
    """Activating an empty workflow succeeds (no structural errors)."""
    wf = await manager.create(TENANT_A, "Empty")
    activated = await manager.activate(wf.id, TENANT_A)
    assert activated.status == WorkflowStatus.ACTIVE


@pytest.mark.asyncio
async def test_pause_workflow(manager, linear_workflow_def):
    """Pause an active workflow."""
    wf = await manager.create(
        TENANT_A,
        "Pausable",
        steps=linear_workflow_def.steps,
        edges=linear_workflow_def.edges,
    )
    await manager.activate(wf.id, TENANT_A)
    paused = await manager.pause(wf.id, TENANT_A)
    assert paused.status == WorkflowStatus.PAUSED


@pytest.mark.asyncio
async def test_rollback(manager):
    """Rollback restores steps from a previous version as a new version."""
    wf = await manager.create(TENANT_A, "Rollback Test")
    assert wf.version == 1

    step_v2 = WorkflowStep(
        id="s-v2", workflow_id=wf.id, step_type=StepType.ACTION, name="v2_step"
    )
    wf2 = await manager.update(wf.id, TENANT_A, steps=[step_v2])
    assert wf2.version == 2

    step_v3 = WorkflowStep(
        id="s-v3", workflow_id=wf.id, step_type=StepType.ACTION, name="v3_step"
    )
    wf3 = await manager.update(wf.id, TENANT_A, steps=[step_v3])
    assert wf3.version == 3

    # Rollback to version 1 (empty steps) creates version 4
    rolled = await manager.rollback(wf.id, TENANT_A, target_version=1)
    assert rolled.version == 4
    assert rolled.steps == []


@pytest.mark.asyncio
async def test_duplicate(manager, linear_workflow_def):
    """Duplicate creates a new workflow with ' (copy)' suffix and new ID."""
    wf = await manager.create(
        TENANT_A,
        "Original",
        steps=linear_workflow_def.steps,
        edges=linear_workflow_def.edges,
    )
    dup = await manager.duplicate(wf.id, TENANT_A)
    assert dup.id != wf.id
    assert dup.name == "Original (copy)"
    assert dup.status == WorkflowStatus.DRAFT
    assert dup.version == 1
    assert len(dup.steps) == len(wf.steps)


@pytest.mark.asyncio
async def test_export_import(manager, linear_workflow_def):
    """Export to JSON and re-import produces an equivalent workflow."""
    wf = await manager.create(
        TENANT_A,
        "Exportable",
        steps=linear_workflow_def.steps,
        edges=linear_workflow_def.edges,
    )
    json_str = await manager.export_json(wf.id, TENANT_A)
    assert isinstance(json_str, str)

    imported = await manager.import_json(json_str, TENANT_A)
    assert imported.id != wf.id  # new ID
    assert imported.name == wf.name
    assert len(imported.steps) == len(wf.steps)
    assert len(imported.edges) == len(wf.edges)


def test_topological_sort_linear(linear_workflow_def):
    """Topological sort of a 2-step linear workflow preserves order."""
    order = topological_sort(linear_workflow_def.steps, linear_workflow_def.edges)
    assert order == ["step-a", "step-b"]


def test_branch_evaluation():
    """evaluate_condition handles == comparison with string context."""
    result = evaluate_condition("${status} == 'error'", {"status": "error"})
    assert result is True

    result_false = evaluate_condition("${status} == 'error'", {"status": "ok"})
    assert result_false is False


def test_loop_validation(loop_workflow_def):
    """A well-formed loop workflow passes validation with no hard errors."""
    validator = WorkflowValidator()
    errors = validator.validate(loop_workflow_def)
    hard_errors = [e for e in errors if not e.startswith("WARNING:")]
    assert hard_errors == []
