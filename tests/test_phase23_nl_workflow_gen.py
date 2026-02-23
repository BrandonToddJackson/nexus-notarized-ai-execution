"""Phase 23: Natural Language Workflow Generation — test suite.

All tests use mocks — no LLM calls, no I/O.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.config import NexusConfig
from nexus.exceptions import BudgetExceeded, WorkflowGenerationError, WorkflowNotFound
from nexus.types import (
    NodePosition,
    StepType,
    WorkflowDefinition,
    WorkflowStatus,
    WorkflowStep,
)
from nexus.workflows import WorkflowGenerator


# ── Helpers ───────────────────────────────────────────────────────────────────

VALID_WF_JSON = {
    "name": "Test Workflow",
    "description": "A simple test workflow",
    "steps": [
        {
            "id": "step1",
            "workflow_id": "",
            "step_type": "action",
            "name": "Search",
            "tool_name": "knowledge_search",
            "persona_name": "researcher",
        }
    ],
    "edges": [],
    "trigger": {"type": "manual", "config": {}},
    "tags": ["generated"],
}

VALID_JSON_STR = json.dumps(VALID_WF_JSON)


def make_step(**kwargs) -> WorkflowStep:
    defaults = {
        "id": "step1",
        "workflow_id": "wf1",
        "step_type": StepType.ACTION,
        "name": "Search",
        "tool_name": "knowledge_search",
        "persona_name": "researcher",
        "position": NodePosition(x=0.0, y=0.0),
    }
    defaults.update(kwargs)
    return WorkflowStep(**defaults)


def make_workflow(version: int = 1) -> WorkflowDefinition:
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)
    return WorkflowDefinition(
        id="wf1",
        tenant_id="tenant1",
        name="Test Workflow",
        description="A test workflow",
        version=version,
        status=WorkflowStatus.DRAFT,
        steps=[make_step(workflow_id="wf1")],
        edges=[],
        tags=["generated"],
        created_by="generator",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value={"content": VALID_JSON_STR, "usage": {}}
    )
    return llm


@pytest.fixture
def mock_tool_registry():
    reg = MagicMock()
    reg.list_tools = MagicMock(return_value=[])
    return reg


@pytest.fixture
def mock_persona_manager():
    mgr = MagicMock()
    mgr.list_personas = MagicMock(return_value=[])
    return mgr


@pytest.fixture
def mock_workflow_manager():
    mgr = MagicMock()
    mgr.create = AsyncMock(return_value=make_workflow())
    mgr.get = AsyncMock(return_value=make_workflow())
    mgr.update = AsyncMock(return_value=make_workflow(version=2))
    mgr.validator = MagicMock()
    mgr.validator.validate = MagicMock(return_value=[])
    return mgr


@pytest.fixture
def default_config():
    return NexusConfig(
        max_generation_refine_attempts=3,
        workflow_generation_model=None,
        secret_key="test-secret-key-that-is-long-enough-123",
    )


def make_generator(
    mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager, config=None
) -> WorkflowGenerator:
    return WorkflowGenerator(
        llm_client=mock_llm,
        tool_registry=mock_tool_registry,
        persona_manager=mock_persona_manager,
        workflow_manager=mock_workflow_manager,
        config=config or NexusConfig(
            max_generation_refine_attempts=3,
            workflow_generation_model=None,
            secret_key="test-secret-key-that-is-long-enough-123",
        ),
    )


# ── TestWorkflowGeneratorParseJson ────────────────────────────────────────────


class TestWorkflowGeneratorParseJson:

    def _gen(self):
        gen = MagicMock(spec=WorkflowGenerator)
        gen._parse_json = WorkflowGenerator._parse_json.__get__(gen)
        return gen

    def test_pure_json_string(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        result = gen._parse_json(VALID_JSON_STR)
        assert result["name"] == "Test Workflow"
        assert "steps" in result
        assert "edges" in result

    def test_json_fence(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        fenced = f"Here is the workflow:\n```json\n{VALID_JSON_STR}\n```\nDone."
        result = gen._parse_json(fenced)
        assert result["name"] == "Test Workflow"

    def test_plain_fence(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        fenced = f"```\n{VALID_JSON_STR}\n```"
        result = gen._parse_json(fenced)
        assert result["name"] == "Test Workflow"

    def test_prose_with_embedded_json(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        prose = f"Sure! Here is the result: {VALID_JSON_STR} That covers the workflow."
        result = gen._parse_json(prose)
        assert result["name"] == "Test Workflow"

    def test_non_json_string_raises(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        with pytest.raises(WorkflowGenerationError):
            gen._parse_json("This is just plain text with no JSON.")

    def test_empty_string_raises(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        with pytest.raises(WorkflowGenerationError):
            gen._parse_json("")

    def test_json_list_raises(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        with pytest.raises(WorkflowGenerationError):
            gen._parse_json('[{"name": "x", "steps": [], "edges": []}]')

    def test_missing_required_key_raises(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        no_steps = json.dumps({"name": "wf", "edges": []})
        with pytest.raises(WorkflowGenerationError, match="steps"):
            gen._parse_json(no_steps)

    def test_missing_trigger_normalized(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        no_trigger = json.dumps({"name": "wf", "steps": [{"id": "s1"}], "edges": []})
        result = gen._parse_json(no_trigger)
        assert result["trigger"] == {"type": "manual", "config": {}}

    def test_null_trigger_normalized(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        null_trigger = json.dumps({"name": "wf", "steps": [], "edges": [], "trigger": None})
        result = gen._parse_json(null_trigger)
        assert result["trigger"] == {"type": "manual", "config": {}}


# ── TestWorkflowGeneratorPreValidate ─────────────────────────────────────────


class TestWorkflowGeneratorPreValidate:

    def gen(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        return make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)

    def _valid_raw(self) -> dict:
        return {
            "name": "wf",
            "steps": [
                {
                    "id": "s1",
                    "step_type": "action",
                    "name": "Step 1",
                    "tool_name": "knowledge_search",
                    "persona_name": "researcher",
                }
            ],
            "edges": [],
            "trigger": {"type": "manual", "config": {}},
        }

    def test_valid_minimal(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(self._valid_raw())
        assert errors == []

    def test_no_steps_error(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {"name": "wf", "steps": [], "edges": []}
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("at least one step" in e for e in errors)

    def test_step_missing_id(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {"name": "wf", "steps": [{"step_type": "action", "name": "x"}], "edges": []}
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("missing" in e and "id" in e for e in errors)

    def test_duplicate_step_ids(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        step = {"id": "s1", "step_type": "action", "name": "x", "tool_name": "t"}
        raw = {"name": "wf", "steps": [step, dict(step)], "edges": []}
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("Duplicate" in e for e in errors)

    def test_invalid_step_type(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {"name": "wf", "steps": [{"id": "s1", "step_type": "alien", "name": "x"}], "edges": []}
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("invalid step_type" in e or "step_type" in e for e in errors)

    def test_action_missing_tool_and_persona(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {"name": "wf", "steps": [{"id": "s1", "step_type": "action", "name": "x"}], "edges": []}
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("tool_name" in e or "persona_name" in e for e in errors)

    def test_branch_missing_conditions(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {"name": "wf", "steps": [{"id": "s1", "step_type": "branch", "name": "b"}], "edges": []}
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("conditions" in e for e in errors)

    def test_loop_missing_iterator(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {"name": "wf", "steps": [{"id": "s1", "step_type": "loop", "name": "l"}], "edges": []}
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("iterator" in e for e in errors)

    def test_sub_workflow_missing_sub_workflow_id(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {"name": "wf", "steps": [{"id": "s1", "step_type": "sub_workflow", "name": "sw"}], "edges": []}
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("sub_workflow_id" in e for e in errors)

    def test_wait_missing_seconds(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {"name": "wf", "steps": [{"id": "s1", "step_type": "wait", "name": "w"}], "edges": []}
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("seconds" in e for e in errors)

    def test_wait_zero_seconds(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {"name": "wf", "steps": [{"id": "s1", "step_type": "wait", "name": "w", "config": {"seconds": 0}}], "edges": []}
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("seconds" in e for e in errors)

    def test_human_approval_missing_message(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {"name": "wf", "steps": [{"id": "s1", "step_type": "human_approval", "name": "ha"}], "edges": []}
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("message" in e for e in errors)

    def test_edge_missing_source_step_id(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {
            "name": "wf",
            "steps": [{"id": "s1", "step_type": "action", "name": "x", "tool_name": "t"}],
            "edges": [{"target_step_id": "s1"}],
        }
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("source_step_id" in e for e in errors)

    def test_invalid_edge_type(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {
            "name": "wf",
            "steps": [{"id": "s1", "step_type": "action", "name": "x", "tool_name": "t"}],
            "edges": [{"source_step_id": "s1", "target_step_id": "s1", "edge_type": "alien"}],
        }
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("edge_type" in e for e in errors)

    def test_conditional_edge_missing_condition(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        raw = {
            "name": "wf",
            "steps": [{"id": "s1", "step_type": "action", "name": "x", "tool_name": "t"}],
            "edges": [{"source_step_id": "s1", "target_step_id": "s1", "edge_type": "conditional"}],
        }
        errors = self.gen(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)._pre_validate(raw)
        assert any("condition" in e for e in errors)


# ── TestBuildContext ──────────────────────────────────────────────────────────


class TestBuildContext:

    def test_tool_context_with_tools(self, mock_llm, mock_persona_manager, mock_workflow_manager):
        from nexus.types import ToolDefinition

        reg = MagicMock()
        tool = ToolDefinition(
            name="knowledge_search",
            description="Search the knowledge base",
            parameters={"query": {"type": "string"}},
            resource_pattern="kb:*",
        )
        reg.list_tools = MagicMock(return_value=[tool])
        gen = make_generator(mock_llm, reg, mock_persona_manager, mock_workflow_manager)
        ctx = gen._build_tool_context()
        data = json.loads(ctx)
        assert isinstance(data, list)
        assert data[0]["name"] == "knowledge_search"

    def test_tool_context_empty_registry(self, mock_llm, mock_persona_manager, mock_workflow_manager):
        reg = MagicMock()
        reg.list_tools = MagicMock(return_value=[])
        gen = make_generator(mock_llm, reg, mock_persona_manager, mock_workflow_manager)
        ctx = gen._build_tool_context()
        assert "No tools" in ctx

    def test_persona_context_with_personas(self, mock_llm, mock_tool_registry, mock_workflow_manager):
        from nexus.types import PersonaContract

        mgr = MagicMock()
        persona = PersonaContract(
            name="researcher",
            description="A researcher persona",
            allowed_tools=["knowledge_search"],
            resource_scopes=["kb:*"],
            intent_patterns=["search for information"],
        )
        mgr.list_personas = MagicMock(return_value=[persona])
        gen = make_generator(mock_llm, mock_tool_registry, mgr, mock_workflow_manager)
        ctx = gen._build_persona_context()
        data = json.loads(ctx)
        assert isinstance(data, list)
        assert data[0]["name"] == "researcher"

    def test_persona_context_no_personas(self, mock_llm, mock_tool_registry, mock_workflow_manager):
        mgr = MagicMock()
        mgr.list_personas = MagicMock(return_value=[])
        gen = make_generator(mock_llm, mock_tool_registry, mgr, mock_workflow_manager)
        ctx = gen._build_persona_context()
        assert "No personas" in ctx


# ── TestWorkflowGeneratorGenerate ─────────────────────────────────────────────


class TestWorkflowGeneratorGenerate:

    @pytest.mark.asyncio
    async def test_success_first_attempt(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        result = await gen.generate("Create a search workflow", tenant_id="t1")
        assert result.id == "wf1"
        mock_workflow_manager.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auto_refine_non_json_then_valid(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        """First LLM call returns non-JSON; second returns valid JSON."""
        mock_llm.complete = AsyncMock(
            side_effect=[
                {"content": "Sorry, I cannot do that.", "usage": {}},
                {"content": VALID_JSON_STR, "usage": {}},
            ]
        )
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        result = await gen.generate("workflow description", tenant_id="t1")
        assert result is not None
        assert mock_llm.complete.await_count == 2

    @pytest.mark.asyncio
    async def test_auto_refine_invalid_dag_then_valid(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        """First pre-validation fails; second attempt returns valid."""
        invalid_json = json.dumps({
            "name": "Bad",
            "steps": [{"id": "s1", "step_type": "action", "name": "x"}],  # missing tool/persona
            "edges": [],
        })
        mock_llm.complete = AsyncMock(
            side_effect=[
                {"content": invalid_json, "usage": {}},
                {"content": VALID_JSON_STR, "usage": {}},
            ]
        )
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        result = await gen.generate("workflow description", tenant_id="t1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_exhausts_attempts_raises(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        mock_llm.complete = AsyncMock(return_value={"content": "not json at all", "usage": {}})
        config = NexusConfig(
            max_generation_refine_attempts=3,
            secret_key="test-secret-key-that-is-long-enough-123",
        )
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager, config)
        with pytest.raises(WorkflowGenerationError):
            await gen.generate("workflow", tenant_id="t1")
        assert mock_llm.complete.await_count == 3

    @pytest.mark.asyncio
    async def test_context_hints_passed_to_prompt(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        await gen.generate(
            "workflow",
            tenant_id="t1",
            context={"preferred_personas": ["researcher"], "preferred_tools": ["web_search"]},
        )
        call_args = mock_llm.complete.await_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "researcher" in user_msg
        assert "web_search" in user_msg

    @pytest.mark.asyncio
    async def test_trigger_type_in_user_message(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        await gen.generate("workflow", tenant_id="t1", context={"trigger_type": "cron"})
        call_args = mock_llm.complete.await_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "cron" in user_msg

    @pytest.mark.asyncio
    async def test_budget_exceeded_propagates(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        mock_llm.complete = AsyncMock(side_effect=BudgetExceeded("Budget exceeded"))
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        with pytest.raises(BudgetExceeded):
            await gen.generate("workflow", tenant_id="t1")

    @pytest.mark.asyncio
    async def test_create_called_exactly_once_on_success(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        await gen.generate("workflow", tenant_id="t1")
        mock_workflow_manager.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_created_by_generator_and_generated_tag(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        await gen.generate("workflow", tenant_id="t1")
        kwargs = mock_workflow_manager.create.await_args.kwargs
        assert kwargs.get("created_by") == "generator"
        assert "generated" in kwargs.get("tags", [])


# ── TestWorkflowGeneratorRefine ───────────────────────────────────────────────


class TestWorkflowGeneratorRefine:

    @pytest.mark.asyncio
    async def test_success_returns_version_2(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        result = await gen.refine("wf1", "t1", "Add an extra step")
        assert result.version == 2

    @pytest.mark.asyncio
    async def test_workflow_not_found_propagates(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        mock_workflow_manager.get = AsyncMock(side_effect=WorkflowNotFound("Not found", workflow_id="wf1"))
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        with pytest.raises(WorkflowNotFound):
            await gen.refine("wf1", "t1", "feedback")

    @pytest.mark.asyncio
    async def test_unparseable_json_raises(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        mock_llm.complete = AsyncMock(return_value={"content": "pure nonsense text", "usage": {}})
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        with pytest.raises(WorkflowGenerationError):
            await gen.refine("wf1", "t1", "feedback")

    @pytest.mark.asyncio
    async def test_invalid_dag_raises(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        mock_workflow_manager.validator.validate = MagicMock(
            return_value=["Cycle detected in workflow graph."]
        )
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        with pytest.raises(WorkflowGenerationError):
            await gen.refine("wf1", "t1", "feedback")

    @pytest.mark.asyncio
    async def test_old_workflow_json_in_system_prompt(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        await gen.refine("wf1", "t1", "my feedback")
        call_args = mock_llm.complete.await_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        # The existing workflow name should appear in the system prompt
        assert "Test Workflow" in system_msg

    @pytest.mark.asyncio
    async def test_feedback_in_system_prompt(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        await gen.refine("wf1", "t1", "my specific feedback string XYZ")
        call_args = mock_llm.complete.await_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "my specific feedback string XYZ" in system_msg


# ── TestWorkflowGeneratorExplain ──────────────────────────────────────────────


class TestWorkflowGeneratorExplain:

    @pytest.mark.asyncio
    async def test_returns_long_string(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        long_explanation = " ".join(["word"] * 60)  # 60 words
        mock_llm.complete = AsyncMock(return_value={"content": long_explanation, "usage": {}})
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        result = await gen.explain("wf1", "t1")
        assert len(result.split()) >= 50

    @pytest.mark.asyncio
    async def test_workflow_not_found_propagates(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        mock_workflow_manager.get = AsyncMock(side_effect=WorkflowNotFound("Not found", workflow_id="wf1"))
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        with pytest.raises(WorkflowNotFound):
            await gen.explain("wf1", "t1")

    @pytest.mark.asyncio
    async def test_does_not_call_create_or_update(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        await gen.explain("wf1", "t1")
        mock_workflow_manager.create.assert_not_awaited()
        mock_workflow_manager.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_workflow_json_in_system_prompt(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager)
        await gen.explain("wf1", "t1")
        call_args = mock_llm.complete.await_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "Test Workflow" in system_msg


# ── TestWorkflowGeneratorConfig ───────────────────────────────────────────────


class TestWorkflowGeneratorConfig:

    @pytest.mark.asyncio
    async def test_max_attempts_1_raises_after_one(self, mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        mock_llm.complete = AsyncMock(return_value={"content": "not json", "usage": {}})
        config = NexusConfig(
            max_generation_refine_attempts=1,
            secret_key="test-secret-key-that-is-long-enough-123",
        )
        gen = make_generator(mock_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager, config)
        with pytest.raises(WorkflowGenerationError):
            await gen.generate("workflow", tenant_id="t1")
        assert mock_llm.complete.await_count == 1

    @pytest.mark.asyncio
    async def test_workflow_generation_model_creates_new_client(self, mock_tool_registry, mock_persona_manager, mock_workflow_manager):
        config = NexusConfig(
            workflow_generation_model="openai/gpt-4o",
            secret_key="test-secret-key-that-is-long-enough-123",
        )
        base_llm = MagicMock()
        base_llm.complete = AsyncMock(return_value={"content": VALID_JSON_STR, "usage": {}})
        gen = make_generator(base_llm, mock_tool_registry, mock_persona_manager, mock_workflow_manager, config)
        assert gen._generation_model == "openai/gpt-4o"

    def test_default_config_values(self):
        config = NexusConfig(secret_key="test-secret-key-that-is-long-enough-123")
        assert config.max_generation_refine_attempts == 3
        assert config.workflow_generation_model is None


# ── TestWorkflowGeneratorExports ──────────────────────────────────────────────


class TestWorkflowGeneratorExports:

    def test_import_from_workflows(self):
        from nexus.workflows import WorkflowGenerator  # noqa: F401
        assert WorkflowGenerator is not None

    def test_import_prompts(self):
        from nexus.llm.prompts import (  # noqa: F401
            EXPLAIN_WORKFLOW,
            GENERATE_WORKFLOW,
            REFINE_WORKFLOW,
        )
        assert GENERATE_WORKFLOW
        assert REFINE_WORKFLOW
        assert EXPLAIN_WORKFLOW

    def test_import_exception(self):
        from nexus.exceptions import WorkflowGenerationError  # noqa: F401
        assert WorkflowGenerationError is not None
        assert issubclass(WorkflowGenerationError, Exception)
