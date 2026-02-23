"""WorkflowGenerator — translate natural-language descriptions into validated WorkflowDefinitions.

Calls the LLM, parses JSON output, runs structural pre-validation, then delegates
full DAG validation to WorkflowValidator before persisting via WorkflowManager.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from nexus.config import NexusConfig
from nexus.core.personas import PersonaManager
from nexus.exceptions import WorkflowGenerationError
from nexus.llm.client import LLMClient
from nexus.llm.prompts import EXPLAIN_WORKFLOW, GENERATE_WORKFLOW, REFINE_WORKFLOW
from nexus.tools.registry import ToolRegistry
from nexus.types import (
    EdgeType,
    StepType,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowStep,
)
from nexus.workflows.manager import WorkflowManager

# Required top-level keys that must be present in the LLM response
_REQUIRED_KEYS = {"name", "steps", "edges"}

# Valid step_type values
_VALID_STEP_TYPES = {st.value for st in StepType}

# Valid edge_type values
_VALID_EDGE_TYPES = {et.value for et in EdgeType}


class WorkflowGenerator:
    """Translate a natural-language description into a validated WorkflowDefinition.

    Args:
        llm_client:       Primary LLM client for generation.
        tool_registry:    Registry to enumerate available tools.
        persona_manager:  Manager to enumerate available personas.
        workflow_manager: Manager for persisting the generated workflow.
        config:           Application configuration.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        persona_manager: PersonaManager,
        workflow_manager: WorkflowManager,
        config: NexusConfig,
    ) -> None:
        self._llm = llm_client
        self._tool_registry = tool_registry
        self._persona_manager = persona_manager
        self._workflow_manager = workflow_manager
        self._config = config
        self._generation_model: Optional[str] = config.workflow_generation_model

    # ── Context builders ──────────────────────────────────────────────────────

    def _build_tool_context(self) -> str:
        """Return a JSON string listing available tools, or a placeholder."""
        all_tools = self._tool_registry.list_tools()
        if not all_tools:
            return "No tools registered."
        items = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "resource_pattern": t.resource_pattern,
            }
            for t in all_tools
        ]
        return json.dumps(items, indent=2)

    def _build_persona_context(self) -> str:
        """Return a JSON string listing available personas, or a placeholder."""
        all_personas = self._persona_manager.list_personas()
        if not all_personas:
            return "No personas registered."
        items = [
            {
                "name": p.name,
                "description": p.description,
                "allowed_tools": list(p.allowed_tools),
                "risk_tolerance": p.risk_tolerance,
            }
            for p in all_personas
        ]
        return json.dumps(items, indent=2)

    # ── LLM call ─────────────────────────────────────────────────────────────

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM with the given messages and return the content string."""
        if self._generation_model:
            client = LLMClient(model=self._generation_model)
        else:
            client = self._llm
        response = await client.complete(
            messages=messages, temperature=0.1, max_tokens=4096
        )
        return response["content"]

    # ── JSON parsing ─────────────────────────────────────────────────────────

    def _parse_json(self, raw: str) -> dict[str, Any]:
        """Extract and validate the JSON dict from an LLM response string.

        Tries four strategies in order:
          1. Direct json.loads on the stripped string.
          2. Extract content from ```json ... ``` fences.
          3. Extract content from ``` ... ``` fences (no language tag).
          4. Find the first {...} block in free-form prose.

        Raises:
            WorkflowGenerationError: if no valid JSON dict is found or required
                keys are missing.
        """
        text = raw.strip()
        if not text:
            raise WorkflowGenerationError("LLM returned an empty response.")

        # Strategy 1: direct parse — detect list early and reject
        try:
            direct = json.loads(text)
            if not isinstance(direct, dict):
                raise WorkflowGenerationError(
                    "LLM response JSON is not an object (got list or scalar)."
                )
            parsed = direct
        except (json.JSONDecodeError, ValueError):
            parsed = None

        if parsed is None:
            # Strategy 2: ```json ... ```
            m = re.search(r"```json\s*([\s\S]+?)\s*```", text)
            if m:
                try:
                    obj = json.loads(m.group(1).strip())
                    if isinstance(obj, dict):
                        parsed = obj
                except (json.JSONDecodeError, ValueError):
                    pass

        if parsed is None:
            # Strategy 3: ``` ... ``` (no language tag)
            m = re.search(r"```\s*([\s\S]+?)\s*```", text)
            if m:
                try:
                    obj = json.loads(m.group(1).strip())
                    if isinstance(obj, dict):
                        parsed = obj
                except (json.JSONDecodeError, ValueError):
                    pass

        if parsed is None:
            # Strategy 4: first {...} block in free-form prose
            m = re.search(r"\{[\s\S]+\}", text)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    if isinstance(obj, dict):
                        parsed = obj
                except (json.JSONDecodeError, ValueError):
                    pass

        if parsed is None:
            raise WorkflowGenerationError(
                f"LLM response does not contain valid JSON: {text[:200]!r}"
            )

        missing = _REQUIRED_KEYS - parsed.keys()
        if missing:
            raise WorkflowGenerationError(
                f"LLM response JSON is missing required keys: {sorted(missing)}"
            )

        # Normalise trigger field
        if "trigger" not in parsed or parsed["trigger"] is None:
            parsed["trigger"] = {"type": "manual", "config": {}}

        return parsed

    # ── Pre-validation ────────────────────────────────────────────────────────

    def _pre_validate(self, raw: dict[str, Any]) -> list[str]:
        """Structural checks before constructing WorkflowStep/Edge objects.

        Returns a list of error strings.  Empty list means the raw dict looks
        structurally sound.
        """
        errors: list[str] = []
        steps: list[Any] = raw.get("steps", [])
        edges: list[Any] = raw.get("edges", [])

        if not steps:
            errors.append("Workflow must have at least one step.")
            return errors  # further step checks would be meaningless

        step_ids: set[str] = set()
        seen_ids: set[str] = set()

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"Step {i} is not a dict.")
                continue

            sid = step.get("id")
            if not sid:
                errors.append(f"Step {i} is missing required field 'id'.")
                continue

            if sid in seen_ids:
                errors.append(f"Duplicate step id: {sid!r}.")
            else:
                seen_ids.add(sid)
                step_ids.add(sid)

            stype = step.get("step_type")
            if stype not in _VALID_STEP_TYPES:
                errors.append(
                    f"Step {sid!r} has invalid step_type {stype!r}. "
                    f"Valid: {sorted(_VALID_STEP_TYPES)}"
                )
                continue  # skip type-specific checks if type is unknown

            cfg = step.get("config") or {}

            if stype == StepType.ACTION.value:
                if not step.get("tool_name") and not step.get("persona_name"):
                    errors.append(
                        f"Action step {sid!r} must have 'tool_name' or 'persona_name'."
                    )

            elif stype == StepType.BRANCH.value:
                if not cfg.get("conditions"):
                    errors.append(
                        f"Branch step {sid!r} must have config.conditions."
                    )

            elif stype == StepType.LOOP.value:
                if not cfg.get("iterator"):
                    errors.append(
                        f"Loop step {sid!r} must have config.iterator."
                    )

            elif stype == StepType.SUB_WORKFLOW.value:
                if not cfg.get("sub_workflow_id"):
                    errors.append(
                        f"Sub-workflow step {sid!r} must have config.sub_workflow_id."
                    )

            elif stype == StepType.WAIT.value:
                seconds = cfg.get("seconds")
                if seconds is None or (isinstance(seconds, (int, float)) and seconds <= 0):
                    errors.append(
                        f"Wait step {sid!r} must have config.seconds > 0."
                    )

            elif stype == StepType.HUMAN_APPROVAL.value:
                if not cfg.get("message"):
                    errors.append(
                        f"Human-approval step {sid!r} must have config.message."
                    )

        for j, edge in enumerate(edges):
            if not isinstance(edge, dict):
                errors.append(f"Edge {j} is not a dict.")
                continue

            if not edge.get("source_step_id"):
                errors.append(f"Edge {j} is missing 'source_step_id'.")

            etype = edge.get("edge_type", EdgeType.DEFAULT.value)
            if etype not in _VALID_EDGE_TYPES:
                errors.append(
                    f"Edge {j} has invalid edge_type {etype!r}. "
                    f"Valid: {sorted(_VALID_EDGE_TYPES)}"
                )

            if etype == EdgeType.CONDITIONAL.value and not edge.get("condition"):
                errors.append(
                    f"Conditional edge {j} (source={edge.get('source_step_id')!r}) "
                    "must have a 'condition' string."
                )

        return errors

    # ── WorkflowDefinition construction ──────────────────────────────────────

    def _raw_json_to_workflow_definition(
        self, raw: dict[str, Any], tenant_id: str
    ) -> WorkflowDefinition:
        """Convert the parsed LLM JSON to a WorkflowDefinition (without persisting)."""
        wf_id = ""  # temporary; will be replaced by WorkflowManager.create

        steps = [
            WorkflowStep(
                id=step["id"],
                workflow_id=wf_id,
                step_type=StepType(step["step_type"]),
                name=step.get("name", step["id"]),
                description=step.get("description", ""),
                tool_name=step.get("tool_name"),
                tool_params=step.get("tool_params") or {},
                persona_name=step.get("persona_name") or "researcher",
                position=step.get("position") or {"x": 0.0, "y": 0.0},
                config=step.get("config") or {},
                timeout_seconds=step.get("timeout_seconds", 30),
                retry_policy=step.get("retry_policy") or {},
            )
            for step in raw["steps"]
        ]

        edges = [
            WorkflowEdge(
                id=edge.get("id") or str(uuid4()),
                workflow_id=wf_id,
                source_step_id=edge["source_step_id"],
                target_step_id=edge["target_step_id"],
                edge_type=EdgeType(edge.get("edge_type", EdgeType.DEFAULT.value)),
                condition=edge.get("condition"),
                data_mapping=edge.get("data_mapping") or {},
            )
            for edge in raw["edges"]
        ]

        now = datetime.now(tz=timezone.utc)
        return WorkflowDefinition(
            id=wf_id,
            tenant_id=tenant_id,
            name=raw["name"],
            description=raw.get("description", ""),
            version=1,
            steps=steps,
            edges=edges,
            tags=raw.get("tags", ["generated"]),
            settings=raw.get("settings") or {},
            trigger_config=raw.get("trigger") or {},
            created_at=now,
            updated_at=now,
        )

    def _workflow_definition_to_dict(self, wf: WorkflowDefinition) -> dict[str, Any]:
        """Serialize a WorkflowDefinition to a plain dict for prompt injection."""
        return {
            "id": wf.id,
            "name": wf.name,
            "description": wf.description,
            "version": wf.version,
            "status": wf.status.value,
            "tags": wf.tags,
            "steps": [
                {
                    "id": s.id,
                    "step_type": s.step_type.value,
                    "name": s.name,
                    "description": s.description,
                    "tool_name": s.tool_name,
                    "tool_params": s.tool_params,
                    "persona_name": s.persona_name,
                    "position": {"x": s.position.x, "y": s.position.y},
                    "config": s.config,
                    "timeout_seconds": s.timeout_seconds,
                }
                for s in wf.steps
            ],
            "edges": [
                {
                    "id": e.id,
                    "source_step_id": e.source_step_id,
                    "target_step_id": e.target_step_id,
                    "edge_type": e.edge_type.value,
                    "condition": e.condition,
                }
                for e in wf.edges
            ],
            "trigger": wf.trigger_config,
            "settings": wf.settings,
        }

    # ── Generation attempt ────────────────────────────────────────────────────

    async def _attempt_generate(
        self,
        description: str,
        context: Optional[dict[str, Any]] = None,
        previous_errors: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], WorkflowDefinition]:
        """Single generation attempt.  Returns (raw_json, workflow_candidate)."""
        tool_ctx = self._build_tool_context()
        persona_ctx = self._build_persona_context()

        system_prompt = GENERATE_WORKFLOW.format(
            tool_context=tool_ctx,
            persona_context=persona_ctx,
        )

        user_parts = [description]
        ctx = context or {}
        if ctx.get("preferred_personas"):
            user_parts.append(f"Preferred personas: {ctx['preferred_personas']}")
        if ctx.get("preferred_tools"):
            user_parts.append(f"Preferred tools: {ctx['preferred_tools']}")
        if ctx.get("trigger_type"):
            user_parts.append(f"Trigger type: {ctx['trigger_type']}")
        if previous_errors:
            user_parts.append(
                "Previous attempt failed with these errors — please fix them:\n"
                + "\n".join(f"  - {e}" for e in previous_errors)
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

        raw_text = await self._call_llm(messages)
        raw_json = self._parse_json(raw_text)

        pre_errors = self._pre_validate(raw_json)
        if pre_errors:
            raise WorkflowGenerationError(
                f"Generated workflow failed pre-validation: {pre_errors}",
                details={"errors": pre_errors},
            )

        # Build candidate object for full DAG validation
        candidate = self._raw_json_to_workflow_definition(raw_json, tenant_id="__validate__")
        validation_errors = self._workflow_manager.validator.validate(candidate)
        hard_errors = [e for e in validation_errors if not e.startswith("WARNING:")]
        if hard_errors:
            raise WorkflowGenerationError(
                f"Generated workflow failed DAG validation: {hard_errors}",
                details={"errors": hard_errors},
            )

        return raw_json, candidate

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate(
        self,
        description: str,
        tenant_id: str,
        context: Optional[dict[str, Any]] = None,
    ) -> WorkflowDefinition:
        """Generate a new workflow from a natural-language description.

        Retries up to ``config.max_generation_refine_attempts`` times if the
        LLM returns invalid JSON or a structurally invalid graph.

        Args:
            description: Free-text description of what the workflow should do.
            tenant_id:   Tenant that will own the workflow.
            context:     Optional hints dict with keys:
                           ``preferred_personas``, ``preferred_tools``,
                           ``trigger_type``.

        Returns:
            The persisted WorkflowDefinition (status=DRAFT, version=1).

        Raises:
            WorkflowGenerationError: if all attempts are exhausted.
            BudgetExceeded: if the tenant's LLM budget is exceeded.
        """
        max_attempts = self._config.max_generation_refine_attempts
        last_errors: list[str] = []

        for attempt in range(max_attempts):
            try:
                raw_json, workflow_candidate = await self._attempt_generate(
                    description,
                    context=context,
                    previous_errors=last_errors if attempt > 0 else None,
                )
                created = await self._workflow_manager.create(
                    tenant_id=tenant_id,
                    name=raw_json["name"],
                    description=raw_json.get("description", description[:200]),
                    steps=workflow_candidate.steps,
                    edges=workflow_candidate.edges,
                    trigger_config=raw_json.get("trigger"),
                    settings=raw_json.get("settings"),
                    tags=raw_json.get("tags", ["generated"]),
                    created_by="generator",
                )
                return created
            except WorkflowGenerationError as exc:
                last_errors = exc.details.get("errors", [str(exc)])
                if attempt == max_attempts - 1:
                    raise

        # Should be unreachable, but satisfy type checker
        raise WorkflowGenerationError(
            f"Failed to generate a valid workflow after {max_attempts} attempts."
        )

    async def refine(
        self,
        workflow_id: str,
        tenant_id: str,
        feedback: str,
    ) -> WorkflowDefinition:
        """Refine an existing workflow based on natural-language feedback.

        Args:
            workflow_id: ID of the workflow to refine.
            tenant_id:   Owning tenant.
            feedback:    Free-text feedback describing desired changes.

        Returns:
            The updated WorkflowDefinition (version incremented).

        Raises:
            WorkflowNotFound:       if the workflow does not exist.
            WorkflowGenerationError: if the LLM produces unparseable or invalid output.
        """
        existing = await self._workflow_manager.get(
            workflow_id=workflow_id, tenant_id=tenant_id
        )
        existing_dict = self._workflow_definition_to_dict(existing)
        tool_ctx = self._build_tool_context()
        persona_ctx = self._build_persona_context()

        system_prompt = REFINE_WORKFLOW.format(
            current_workflow_json=json.dumps(existing_dict, indent=2),
            feedback=feedback,
            tool_context=tool_ctx,
            persona_context=persona_ctx,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": feedback},
        ]

        raw_text = await self._call_llm(messages)
        raw_json = self._parse_json(raw_text)

        pre_errors = self._pre_validate(raw_json)
        if pre_errors:
            raise WorkflowGenerationError(
                f"Refined workflow failed pre-validation: {pre_errors}",
                details={"errors": pre_errors},
            )

        candidate = self._raw_json_to_workflow_definition(raw_json, tenant_id=tenant_id)
        validation_errors = self._workflow_manager.validator.validate(candidate)
        hard_errors = [e for e in validation_errors if not e.startswith("WARNING:")]
        if hard_errors:
            raise WorkflowGenerationError(
                f"Refined workflow failed DAG validation: {hard_errors}",
                details={"errors": hard_errors},
            )

        updated = await self._workflow_manager.update(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            name=raw_json["name"],
            description=raw_json.get("description", existing.description),
            steps=candidate.steps,
            edges=candidate.edges,
            trigger_config=raw_json.get("trigger", existing.trigger_config),
            settings=raw_json.get("settings", existing.settings),
            tags=raw_json.get("tags", existing.tags),
        )
        return updated

    async def explain(
        self,
        workflow_id: str,
        tenant_id: str,
    ) -> str:
        """Generate a plain-English explanation of a workflow.

        Args:
            workflow_id: ID of the workflow to explain.
            tenant_id:   Owning tenant.

        Returns:
            A multi-paragraph explanation string.

        Raises:
            WorkflowNotFound: if the workflow does not exist.
        """
        workflow = await self._workflow_manager.get(
            workflow_id=workflow_id, tenant_id=tenant_id
        )
        workflow_dict = self._workflow_definition_to_dict(workflow)

        system_prompt = EXPLAIN_WORKFLOW.format(
            workflow_json=json.dumps(workflow_dict, indent=2)
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please explain this workflow."},
        ]

        return await self._call_llm(messages)
