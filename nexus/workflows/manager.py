"""
WorkflowManager — lifecycle management for WorkflowDefinition objects.

Supports both in-memory operation (no repository, for tests and CLI) and
full persistence when a Repository is provided, matching the pattern used
by Ledger and PersonaManager.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from nexus.config import NexusConfig
from nexus.exceptions import WorkflowNotFound, WorkflowValidationError
from nexus.types import (
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowStatus,
    WorkflowStep,
)

from .validator import WorkflowValidator


class WorkflowManager:
    """
    Manages the full lifecycle of WorkflowDefinition objects.

    All methods are async for consistency with the rest of the framework even
    though in-memory operations are synchronous internally.

    Args:
        repository:  Optional Repository instance for persistence.
                     When None, all state is kept in-memory (useful for tests).
        validator:   WorkflowValidator instance.  A default instance is created
                     if not supplied.
        config:      NexusConfig instance.  A default instance is created if
                     not supplied.
    """

    def __init__(
        self,
        repository: Any = None,
        validator: Optional[WorkflowValidator] = None,
        config: Optional[NexusConfig] = None,
    ) -> None:
        self._repository = repository
        self._validator = validator or WorkflowValidator()
        self._config = config or NexusConfig()

        # in-memory stores (always populated, even when repository is present)
        self._store: dict[str, WorkflowDefinition] = {}
        # version history: workflow_id → ordered list of WorkflowDefinition snapshots
        self._version_history: dict[str, list[WorkflowDefinition]] = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _hard_errors(self, errors: list[str]) -> list[str]:
        return [e for e in errors if not e.startswith("WARNING:")]

    def _validate_or_raise(self, workflow: WorkflowDefinition) -> None:
        errors = self._validator.validate(
            workflow, max_steps=self._config.max_workflow_steps
        )
        hard = self._hard_errors(errors)
        if hard:
            raise WorkflowValidationError(
                "Workflow validation failed", violations=hard
            )

    def _save(self, workflow: WorkflowDefinition) -> None:
        """Persist to in-memory store and append to version history."""
        self._store[workflow.id] = workflow
        history = self._version_history.setdefault(workflow.id, [])
        # Only append if this version is new
        if not history or history[-1].version < workflow.version:
            history.append(workflow)

    async def _persist(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Best-effort repository call; silently ignores NotImplementedError."""
        if self._repository is None:
            return
        fn = getattr(self._repository, method, None)
        if fn is None:
            return
        try:
            await fn(*args, **kwargs)
        except NotImplementedError:
            pass

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def create(
        self,
        tenant_id: str,
        name: str,
        description: str = "",
        steps: Optional[list[WorkflowStep]] = None,
        edges: Optional[list[WorkflowEdge]] = None,
        created_by: str = "",
        tags: Optional[list[str]] = None,
        settings: Optional[dict[str, Any]] = None,
        trigger_config: Optional[dict[str, Any]] = None,
    ) -> WorkflowDefinition:
        """
        Validate and persist a new workflow in DRAFT status.

        Returns:
            The created WorkflowDefinition (version=1, status=draft).

        Raises:
            WorkflowValidationError: if the workflow graph is structurally invalid.
        """
        now = datetime.now(tz=timezone.utc)
        workflow = WorkflowDefinition(
            id=str(uuid4()),
            tenant_id=tenant_id,
            name=name,
            description=description,
            version=1,
            status=WorkflowStatus.DRAFT,
            steps=steps or [],
            edges=edges or [],
            created_by=created_by,
            tags=tags or [],
            settings=settings or {},
            trigger_config=trigger_config or {},
            created_at=now,
            updated_at=now,
        )
        self._validate_or_raise(workflow)
        self._save(workflow)
        await self._persist("create_workflow", workflow)
        return workflow

    async def get(self, workflow_id: str, tenant_id: str) -> WorkflowDefinition:
        """
        Load a workflow by ID.

        Raises:
            WorkflowNotFound: if not found or tenant mismatch.
        """
        workflow = self._store.get(workflow_id)
        if workflow is None and self._repository is not None:
            try:
                workflow = await self._repository.get_workflow(workflow_id, tenant_id)
                if workflow is not None:
                    self._store[workflow.id] = workflow
            except (NotImplementedError, AttributeError):
                pass

        if workflow is None or workflow.tenant_id != tenant_id:
            raise WorkflowNotFound(
                f"Workflow '{workflow_id}' not found.", workflow_id=workflow_id
            )
        return workflow

    async def update(
        self,
        workflow_id: str,
        tenant_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        steps: Optional[list[WorkflowStep]] = None,
        edges: Optional[list[WorkflowEdge]] = None,
        tags: Optional[list[str]] = None,
        settings: Optional[dict[str, Any]] = None,
        trigger_config: Optional[dict[str, Any]] = None,
        status: Optional[WorkflowStatus] = None,
    ) -> WorkflowDefinition:
        """
        Update a workflow.

        If steps or edges change the version is incremented and the updated
        graph is re-validated.  Metadata-only changes (name, tags, etc.) do
        not increment the version.

        Raises:
            WorkflowNotFound: if the workflow does not exist.
            WorkflowValidationError: if the updated graph is invalid.
        """
        existing = await self.get(workflow_id, tenant_id)
        structure_changed = steps is not None or edges is not None

        updates: dict[str, Any] = {"updated_at": datetime.now(tz=timezone.utc)}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if steps is not None:
            updates["steps"] = steps
        if edges is not None:
            updates["edges"] = edges
        if tags is not None:
            updates["tags"] = tags
        if settings is not None:
            updates["settings"] = settings
        if trigger_config is not None:
            updates["trigger_config"] = trigger_config
        if status is not None:
            updates["status"] = status
        if structure_changed:
            updates["version"] = existing.version + 1

        updated = existing.model_copy(update=updates)

        if structure_changed:
            self._validate_or_raise(updated)

        self._save(updated)
        await self._persist("update_workflow", workflow_id, updates)
        return updated

    async def list(
        self,
        tenant_id: str,
        status: Optional[WorkflowStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[WorkflowDefinition]:
        """Return workflows for a tenant, optionally filtered by status."""
        results = [
            wf for wf in self._store.values()
            if wf.tenant_id == tenant_id
            and (status is None or wf.status == status)
        ]
        # Sort by creation time descending, then paginate
        results.sort(key=lambda w: w.created_at, reverse=True)
        return results[offset: offset + limit]

    # ── Status transitions ────────────────────────────────────────────────────

    async def activate(self, workflow_id: str, tenant_id: str) -> WorkflowDefinition:
        """
        Validate and activate a workflow.

        Raises:
            WorkflowValidationError: if graph has hard errors.
        """
        workflow = await self.get(workflow_id, tenant_id)
        # Re-validate before activating (steps may have changed externally)
        self._validate_or_raise(workflow)
        return await self.update(workflow_id, tenant_id, status=WorkflowStatus.ACTIVE)

    async def pause(self, workflow_id: str, tenant_id: str) -> WorkflowDefinition:
        """Pause an active workflow."""
        return await self.update(workflow_id, tenant_id, status=WorkflowStatus.PAUSED)

    async def archive(self, workflow_id: str, tenant_id: str) -> WorkflowDefinition:
        """Archive a workflow (terminal state — triggers are disabled)."""
        return await self.update(workflow_id, tenant_id, status=WorkflowStatus.ARCHIVED)

    # ── Versioning ────────────────────────────────────────────────────────────

    async def get_version_history(
        self, workflow_id: str, tenant_id: str
    ) -> list[WorkflowDefinition]:
        """
        Return all saved versions of a workflow, oldest first.

        Verifies tenant ownership via get().
        """
        await self.get(workflow_id, tenant_id)  # ownership check
        return list(self._version_history.get(workflow_id, []))

    async def rollback(
        self, workflow_id: str, tenant_id: str, target_version: int
    ) -> WorkflowDefinition:
        """
        Create a new version whose steps/edges match target_version.

        The rollback itself becomes a new version (not a destructive rewrite).

        Raises:
            WorkflowNotFound: if target_version does not exist in history.
        """
        history = await self.get_version_history(workflow_id, tenant_id)
        target = next((v for v in history if v.version == target_version), None)
        if target is None:
            raise WorkflowNotFound(
                f"Version {target_version} not found for workflow '{workflow_id}'.",
                workflow_id=workflow_id,
            )
        return await self.update(
            workflow_id,
            tenant_id,
            steps=list(target.steps),
            edges=list(target.edges),
        )

    # ── Copy / import / export ────────────────────────────────────────────────

    async def duplicate(
        self,
        workflow_id: str,
        tenant_id: str,
        new_name: Optional[str] = None,
    ) -> WorkflowDefinition:
        """
        Deep-copy a workflow with a new ID, version=1, and status=DRAFT.

        Steps and edges get new IDs; internal step/edge references within the
        copy are remapped accordingly.
        """
        source = await self.get(workflow_id, tenant_id)
        return await self._clone_as_new(source, tenant_id, new_name)

    async def _clone_as_new(
        self,
        source: WorkflowDefinition,
        tenant_id: str,
        name: Optional[str] = None,
    ) -> WorkflowDefinition:
        """Internal: deep copy source into a fresh workflow."""
        step_id_map: dict[str, str] = {}
        new_steps: list[WorkflowStep] = []
        for step in source.steps:
            new_id = str(uuid4())
            step_id_map[step.id] = new_id
            new_steps.append(step.model_copy(update={"id": new_id}))

        new_edges: list[WorkflowEdge] = []
        for edge in source.edges:
            new_edges.append(
                edge.model_copy(
                    update={
                        "id": str(uuid4()),
                        "source_step_id": step_id_map.get(edge.source_step_id, edge.source_step_id),
                        "target_step_id": step_id_map.get(edge.target_step_id, edge.target_step_id),
                    }
                )
            )

        return await self.create(
            tenant_id=tenant_id,
            name=name or f"{source.name} (copy)",
            description=source.description,
            steps=new_steps,
            edges=new_edges,
            tags=list(source.tags),
            settings=copy.deepcopy(source.settings),
            trigger_config=copy.deepcopy(source.trigger_config),
        )

    async def export_json(self, workflow_id: str, tenant_id: str) -> str:
        """
        Serialize a workflow to a JSON string.

        The output includes all fields needed for import_json to reconstruct
        an equivalent workflow.
        """
        workflow = await self.get(workflow_id, tenant_id)
        return workflow.model_dump_json(indent=2)

    async def import_json(
        self,
        json_str: str,
        tenant_id: str,
        created_by: str = "",
    ) -> WorkflowDefinition:
        """
        Deserialize and import a workflow from a JSON string.

        All step, edge, and workflow IDs are remapped to fresh UUIDs to avoid
        collisions with existing workflows.

        Raises:
            WorkflowValidationError: if the parsed workflow graph is invalid.
            ValueError: if the JSON is malformed.
        """
        data = json.loads(json_str)
        # Parse as WorkflowDefinition to validate field structure
        source = WorkflowDefinition.model_validate(data)
        # Clone with remapped IDs under the target tenant
        return await self._clone_as_new(source, tenant_id, name=source.name)
