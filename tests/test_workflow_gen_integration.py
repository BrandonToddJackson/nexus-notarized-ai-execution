"""Integration tests for WorkflowGenerator ↔ WorkflowManager seam — Gap 19.

Verifies:
  - Generator retries when DAG validation fails on first LLM response
  - Generator exhausts max retries and raises WorkflowGenerationError
  - Retry passes previous_errors to next LLM call
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from nexus.config import NexusConfig
from nexus.core.personas import PersonaManager
from nexus.exceptions import WorkflowGenerationError
from nexus.tools.registry import ToolRegistry
from nexus.types import (
    PersonaContract,
    RiskLevel,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowStatus,
)
from nexus.workflows.generator import WorkflowGenerator
from nexus.workflows.validator import WorkflowValidator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _config(max_attempts: int = 3) -> NexusConfig:
    return NexusConfig(
        database_url="sqlite+aiosqlite:///test.db",
        redis_url="redis://localhost:6379/15",
        secret_key="test-secret",
        max_generation_refine_attempts=max_attempts,
        workflow_generation_model=None,  # use injected llm_client directly
    )


def _researcher_persona() -> PersonaContract:
    return PersonaContract(
        name="researcher",
        description="Searches for information",
        allowed_tools=["knowledge_search"],
        resource_scopes=["kb:*"],
        intent_patterns=["search for information"],
        risk_tolerance=RiskLevel.LOW,
        max_ttl_seconds=120,
    )


def _knowledge_search_defn() -> ToolDefinition:
    return ToolDefinition(
        name="knowledge_search",
        description="Search the knowledge base",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        risk_level=RiskLevel.LOW,
        resource_pattern="kb:*",
    )


def _llm_response(content: str) -> dict:
    return {
        "content": content,
        "tool_calls": [],
        "usage": {"input_tokens": 10, "output_tokens": 50},
    }


# Workflow JSONs

_CYCLIC_WORKFLOW_JSON = json.dumps({
    "name": "cyclic_workflow",
    "description": "Has a cycle — should fail DAG validation",
    "steps": [
        {
            "id": "s1",
            "step_type": "action",
            "name": "step_one",
            "tool_name": "knowledge_search",
            "persona_name": "researcher",
        },
        {
            "id": "s2",
            "step_type": "action",
            "name": "step_two",
            "tool_name": "knowledge_search",
            "persona_name": "researcher",
        },
    ],
    "edges": [
        {"id": "e1", "source_step_id": "s1", "target_step_id": "s2", "edge_type": "default"},
        {"id": "e2", "source_step_id": "s2", "target_step_id": "s1", "edge_type": "default"},
    ],
})

_VALID_WORKFLOW_JSON = json.dumps({
    "name": "valid_workflow",
    "description": "Single action step — passes all validation",
    "steps": [
        {
            "id": "s1",
            "step_type": "action",
            "name": "search_step",
            "tool_name": "knowledge_search",
            "persona_name": "researcher",
        }
    ],
    "edges": [],
})


def _make_generator(
    mock_llm: AsyncMock,
    max_attempts: int = 3,
) -> WorkflowGenerator:
    """Build a WorkflowGenerator with mocked LLM and real WorkflowValidator."""
    registry = ToolRegistry()
    registry.register(_knowledge_search_defn(), lambda q: q)

    persona_manager = PersonaManager([_researcher_persona()])

    # WorkflowManager mock with a REAL validator
    mock_wm = MagicMock()
    mock_wm.validator = WorkflowValidator()

    async def _mock_create(tenant_id, name, description, steps, edges, **kwargs):
        return WorkflowDefinition(
            id="generated-wf-001",
            tenant_id=tenant_id,
            name=name,
            description=description,
            version=1,
            steps=steps,
            edges=edges,
            status=WorkflowStatus.DRAFT,
        )

    mock_wm.create = _mock_create

    config = _config(max_attempts=max_attempts)

    return WorkflowGenerator(
        llm_client=mock_llm,
        tool_registry=registry,
        persona_manager=persona_manager,
        workflow_manager=mock_wm,
        config=config,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGeneratorRetryOnValidationFailure:

    async def test_retries_when_dag_validation_fails(self):
        """First LLM response produces a cyclic graph → fails DAG validation → retries.

        Second response produces a valid workflow → succeeds.
        """
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=[
            _llm_response(_CYCLIC_WORKFLOW_JSON),   # attempt 1: cycle → DAG error
            _llm_response(_VALID_WORKFLOW_JSON),    # attempt 2: valid
        ])

        gen = _make_generator(mock_llm, max_attempts=3)
        result = await gen.generate("Find information about NEXUS", "tenant-test")

        assert result is not None
        assert result.name == "valid_workflow"
        # LLM was called exactly twice
        assert mock_llm.complete.await_count == 2

    async def test_max_retries_exhausted_raises_generation_error(self):
        """When all attempts return invalid JSON, WorkflowGenerationError is raised."""
        mock_llm = AsyncMock()
        # Always return invalid (non-JSON) content
        mock_llm.complete = AsyncMock(return_value=_llm_response("this is not json at all"))

        gen = _make_generator(mock_llm, max_attempts=3)

        with pytest.raises(WorkflowGenerationError):
            await gen.generate("broken description", "tenant-test")

        # All 3 attempts exhausted
        assert mock_llm.complete.await_count == 3

    async def test_previous_errors_forwarded_on_retry(self):
        """On retry, the LLM prompt must contain the previous errors."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=[
            _llm_response(_CYCLIC_WORKFLOW_JSON),  # fails
            _llm_response(_VALID_WORKFLOW_JSON),   # succeeds
        ])

        gen = _make_generator(mock_llm, max_attempts=3)
        await gen.generate("test description", "tenant-test")

        # Second call's messages must reference previous errors
        second_call_messages = mock_llm.complete.call_args_list[1].kwargs["messages"]
        user_message = next(m for m in second_call_messages if m["role"] == "user")
        assert "previous attempt" in user_message["content"].lower() or "error" in user_message["content"].lower()

    async def test_single_valid_attempt_does_not_retry(self):
        """When first response is valid, LLM is called exactly once."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=_llm_response(_VALID_WORKFLOW_JSON))

        gen = _make_generator(mock_llm, max_attempts=3)
        result = await gen.generate("valid description", "tenant-test")

        assert result is not None
        assert mock_llm.complete.await_count == 1


class TestGeneratorMaxAttemptsConfig:

    async def test_max_attempts_1_raises_immediately_on_failure(self):
        """With max_attempts=1, no retry happens."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=_llm_response("not json"))

        gen = _make_generator(mock_llm, max_attempts=1)

        with pytest.raises(WorkflowGenerationError):
            await gen.generate("description", "tenant-test")

        assert mock_llm.complete.await_count == 1

    async def test_all_cyclic_attempts_raises_after_n_tries(self):
        """If all N attempts produce cyclic graphs, raises after N attempts."""
        mock_llm = AsyncMock()
        # 5 cyclic responses
        mock_llm.complete = AsyncMock(return_value=_llm_response(_CYCLIC_WORKFLOW_JSON))

        gen = _make_generator(mock_llm, max_attempts=2)

        with pytest.raises(WorkflowGenerationError):
            await gen.generate("cyclic description", "tenant-test")

        assert mock_llm.complete.await_count == 2
