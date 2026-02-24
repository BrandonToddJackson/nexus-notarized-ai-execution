"""Phase 30: WorkflowGenerator tests using local Ollama models.

These tests call a real LLM (ollama/qwen2.5-coder:7b or qwen2.5vl:7b running
locally at http://localhost:11434) to verify that natural-language descriptions
are actually converted into valid, persisted WorkflowDefinitions.

Run with:
    pytest tests/test_workflow_generator.py -v -m slow
"""

import pytest

from nexus.core.personas import PersonaManager
from nexus.exceptions import WorkflowGenerationError
from nexus.llm.client import LLMClient
from nexus.tools.registry import ToolRegistry
from nexus.types import RiskLevel, StepType, ToolDefinition
from nexus.workflows.generator import WorkflowGenerator
from nexus.workflows.manager import WorkflowManager
from nexus.workflows.validator import WorkflowValidator

TENANT = "tenant-ollama-test"

# Valid step_type values (lowercase enum values)
_VALID_STEP_TYPES = {st.value for st in StepType}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_generator(model: str) -> WorkflowGenerator:
    """Build a WorkflowGenerator wired to a specific local Ollama model."""
    from nexus.config import NexusConfig
    from nexus.types import PersonaContract

    cfg = NexusConfig()

    registry = ToolRegistry()
    # Register a handful of representative tools so the LLM has real context.
    for name, desc, pattern in [
        ("knowledge_search", "Search the internal knowledge base", "kb:*"),
        ("web_search", "Search the web for public information", "web:*"),
        ("send_email", "Send an email to a recipient", "email:*"),
        ("file_read", "Read a file from the filesystem", "file:read:*"),
        ("file_write", "Write content to a file", "file:write:*"),
        ("http_request", "Make an HTTP request to an external API", "web:api:*"),
    ]:
        registry.register(
            ToolDefinition(
                name=name,
                description=desc,
                parameters={"type": "object", "properties": {}, "required": []},
                risk_level=RiskLevel.LOW,
                resource_pattern=pattern,
            ),
            lambda **kw: None,  # no-op impl for context purposes
        )

    personas = [
        PersonaContract(
            name="researcher",
            description="Searches and retrieves information",
            allowed_tools=["knowledge_search", "web_search", "web_fetch", "file_read"],
            resource_scopes=["kb:*", "web:*", "file:read:*"],
            intent_patterns=["search for information", "find data about", "look up", "research"],
            risk_tolerance=RiskLevel.LOW,
            max_ttl_seconds=60,
        ),
        PersonaContract(
            name="communicator",
            description="Sends emails and messages",
            allowed_tools=["knowledge_search", "send_email", "file_read"],
            resource_scopes=["kb:*", "email:*", "file:read:*"],
            intent_patterns=["send email", "notify", "communicate"],
            risk_tolerance=RiskLevel.HIGH,
            max_ttl_seconds=60,
        ),
        PersonaContract(
            name="operator",
            description="Executes code and system operations",
            allowed_tools=["knowledge_search", "file_read", "file_write", "http_request"],
            resource_scopes=["kb:*", "file:*", "web:api:*"],
            intent_patterns=["execute", "run", "fetch data", "write file"],
            risk_tolerance=RiskLevel.HIGH,
            max_ttl_seconds=180,
        ),
    ]
    persona_manager = PersonaManager(personas)

    llm = LLMClient(model=model)

    return WorkflowGenerator(
        llm_client=llm,
        tool_registry=registry,
        persona_manager=persona_manager,
        workflow_manager=WorkflowManager(validator=WorkflowValidator()),
        config=cfg,
    )


# ── Slow tests (real Ollama LLM) ──────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.asyncio
async def test_generate_simple_workflow_coder_model():
    """qwen2.5-coder:7b generates a WorkflowDefinition from a plain-English task."""
    gen = _make_generator("ollama/qwen2.5-coder:7b")
    result = await gen.generate(
        description="Search the knowledge base for recent AI safety papers and send a summary email",
        tenant_id=TENANT,
    )

    assert result.name, "Generated workflow must have a non-empty name"
    assert result.tenant_id == TENANT
    assert len(result.steps) >= 1, "Workflow must have at least one step"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_generated_workflow_steps_have_valid_types():
    """All step_type values in the generated workflow are valid StepType enum members."""
    gen = _make_generator("ollama/qwen2.5-coder:7b")
    result = await gen.generate(
        description="Fetch data from an external API and write the results to a file",
        tenant_id=TENANT,
    )

    for step in result.steps:
        assert step.step_type.value in _VALID_STEP_TYPES, (
            f"Step '{step.name}' has invalid step_type: {step.step_type!r}"
        )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_generated_workflow_passes_dag_validation():
    """WorkflowValidator reports no hard errors on the LLM-generated workflow."""
    gen = _make_generator("ollama/qwen2.5-coder:7b")
    result = await gen.generate(
        description="Read a configuration file and send a Slack notification with its contents",
        tenant_id=TENANT,
    )

    validator = WorkflowValidator()
    errors = validator.validate(result)
    hard_errors = [e for e in errors if not e.startswith("WARNING:")]
    assert hard_errors == [], (
        f"Generated workflow failed DAG validation:\n" + "\n".join(hard_errors)
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_generate_workflow_vision_model():
    """qwen2.5vl:7b (multimodal) also generates a valid workflow for a general task."""
    gen = _make_generator("ollama/qwen2.5vl:7b")
    result = await gen.generate(
        description="Search the web for the latest news and email a summary to the team",
        tenant_id=TENANT,
    )

    assert result.name, "Generated workflow must have a non-empty name"
    assert len(result.steps) >= 1, "Workflow must have at least one step"
    for step in result.steps:
        assert step.step_type.value in _VALID_STEP_TYPES


@pytest.mark.slow
@pytest.mark.asyncio
async def test_generate_workflow_is_persisted():
    """After generate(), the workflow exists in WorkflowManager's in-memory store."""
    gen = _make_generator("ollama/qwen2.5-coder:7b")
    result = await gen.generate(
        description="Look up customer data and write a report file",
        tenant_id=TENANT,
    )

    # The generator persists via WorkflowManager — retrieve by ID to confirm
    retrieved = await gen._workflow_manager.get(result.id, TENANT)
    assert retrieved.id == result.id
    assert retrieved.name == result.name
