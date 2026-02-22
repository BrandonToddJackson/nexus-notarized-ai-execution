"""
WorkflowValidator — structural correctness checker for WorkflowDefinition.

All 9 checks are non-destructive reads of the workflow graph.  Warnings
(soft issues) are returned with a "WARNING:" prefix so callers can choose
to treat them differently from hard errors.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Optional

from nexus.exceptions import WorkflowValidationError
from nexus.types import EdgeType, StepType, WorkflowDefinition

from .dag import get_children, get_entry_points, get_parallel_group, topological_sort

if TYPE_CHECKING:
    from nexus.core.personas import PersonaManager
    from nexus.tools.registry import ToolRegistry


class WorkflowValidator:
    """
    Validates the structural integrity of a WorkflowDefinition.

    Usage::

        validator = WorkflowValidator()
        errors = validator.validate(workflow, registry=registry, persona_manager=pm)
        hard_errors = [e for e in errors if not e.startswith("WARNING:")]
        if hard_errors:
            raise WorkflowValidationError("Invalid workflow", violations=hard_errors)

    All checks are run even if earlier ones fail, so callers get the full
    error list in one shot.
    """

    def validate(
        self,
        workflow: WorkflowDefinition,
        registry: Optional["ToolRegistry"] = None,
        persona_manager: Optional["PersonaManager"] = None,
        max_steps: int = 50,
    ) -> list[str]:
        """
        Run all structural checks on a WorkflowDefinition.

        Args:
            workflow:        The workflow to validate.
            registry:        Optional ToolRegistry; skips check 8 if None.
            persona_manager: Optional PersonaManager; skips check 9 if None.
            max_steps:       Maximum allowed steps (default 50, override from config).

        Returns:
            List of error strings.  Empty list means the workflow is valid.
            Items prefixed "WARNING:" are soft warnings, not hard failures.
        """
        errors: list[str] = []
        steps = workflow.steps
        edges = workflow.edges
        step_ids = {s.id for s in steps}

        # ── Check 3: Edge validity ────────────────────────────────────────────
        # Must run first so that graph algorithms work on a consistent set.
        valid_edges = []
        for edge in edges:
            edge_ok = True
            if edge.source_step_id not in step_ids:
                errors.append(
                    f"Edge '{edge.id}': source_step_id '{edge.source_step_id}' "
                    "references a step that does not exist."
                )
                edge_ok = False
            if edge.target_step_id not in step_ids:
                errors.append(
                    f"Edge '{edge.id}': target_step_id '{edge.target_step_id}' "
                    "references a step that does not exist."
                )
                edge_ok = False
            if edge_ok:
                valid_edges.append(edge)

        # ── Check 7: Step count limit ─────────────────────────────────────────
        if len(steps) > max_steps:
            errors.append(
                f"Workflow has {len(steps)} steps; maximum allowed is {max_steps}."
            )

        # ── Check 1: DAG acyclicity ───────────────────────────────────────────
        # LOOP_BACK edges are intentional back-edges — exclude them from the
        # cycle check so that valid loop steps don't trigger a false positive.
        dag_edges = [e for e in valid_edges if e.edge_type != EdgeType.LOOP_BACK]
        try:
            topological_sort(steps, dag_edges)
        except WorkflowValidationError as exc:
            errors.extend(exc.violations)

        # ── Check 2: Connectivity ─────────────────────────────────────────────
        # A workflow is connected if all steps belong to one component.
        # We check this using undirected BFS: treat every edge as bidirectional
        # so isolated nodes (no edges at all) are correctly identified as
        # disconnected from the main graph even though they are technically
        # "entry points" with no incoming edges.
        if steps:
            # Build undirected adjacency from the valid (well-referenced) edges
            undirected: dict[str, set[str]] = {s.id: set() for s in steps}
            for edge in valid_edges:
                undirected[edge.source_step_id].add(edge.target_step_id)
                undirected[edge.target_step_id].add(edge.source_step_id)

            # BFS from the first step — all steps should be reachable
            start = steps[0].id
            visited: set[str] = set()
            queue: deque[str] = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in undirected[node]:
                    queue.append(neighbor)

            for step in steps:
                if step.id not in visited:
                    errors.append(
                        f"Step '{step.name}' (id={step.id!r}) is disconnected — "
                        "not reachable from the rest of the workflow graph."
                    )

            # Also flag the degenerate case where all steps have incoming edges
            # (pure cycle with no entry point) since topological_sort may catch
            # it, but we want a clear connectivity message too.
            # Exclude LOOP_BACK edges — they are intentional and must not make a
            # LOOP step appear to have no entry point.
            entry_points = get_entry_points(steps, dag_edges)
            if not entry_points:
                errors.append(
                    "No entry points found — every step has at least one incoming "
                    "edge.  This usually indicates a cycle."
                )

        # ── Check 4: Branch completeness ──────────────────────────────────────
        for step in steps:
            if step.step_type == StepType.BRANCH:
                outgoing = get_children(step.id, valid_edges)
                if len(outgoing) < 2:
                    errors.append(
                        f"Branch step '{step.name}' must have at least 2 outgoing "
                        f"edges (has {len(outgoing)})."
                    )
                edge_types = [e.edge_type for _, e in outgoing]
                if EdgeType.DEFAULT not in edge_types:
                    errors.append(
                        f"Branch step '{step.name}' is missing a 'default' edge "
                        "(the else/fallback case)."
                    )

        # ── Check 5: Loop safety ──────────────────────────────────────────────
        for step in steps:
            if step.step_type == StepType.LOOP:
                outgoing = get_children(step.id, valid_edges)
                edge_types = [e.edge_type for _, e in outgoing]
                if EdgeType.LOOP_BACK not in edge_types:
                    errors.append(
                        f"Loop step '{step.name}' must have exactly one outgoing "
                        "edge with edge_type 'loop_back'."
                    )
                if EdgeType.DEFAULT not in edge_types:
                    errors.append(
                        f"Loop step '{step.name}' must have a 'default' outgoing "
                        "edge (the exit/termination condition)."
                    )

        # ── Check 6: Parallel convergence (warning only) ──────────────────────
        # Track warned group signatures to avoid duplicate warnings when multiple
        # PARALLEL steps belong to the same fork (they share identical parents).
        warned_groups: set[frozenset[str]] = set()
        for step in steps:
            if step.step_type == StepType.PARALLEL:
                group = get_parallel_group(step.id, steps, valid_edges)
                group_key = frozenset(group)
                if group_key in warned_groups:
                    continue
                for member_id in group:
                    children = get_children(member_id, valid_edges)
                    if not children:
                        errors.append(
                            f"WARNING: Parallel step group member (id={member_id!r}) "
                            "has no outgoing edges — parallel branches may never converge."
                        )
                        warned_groups.add(group_key)
                        break  # one warning per group is enough

        # ── Check 8: Tool references ──────────────────────────────────────────
        if registry is not None:
            registered_tools = {t.name for t in registry.list_tools()}
            for step in steps:
                if step.tool_name and step.tool_name not in registered_tools:
                    errors.append(
                        f"Step '{step.name}': tool '{step.tool_name}' is not "
                        "registered in the ToolRegistry."
                    )

        # ── Check 9: Persona references ───────────────────────────────────────
        if persona_manager is not None:
            for step in steps:
                if step.persona_name and persona_manager.get_persona(step.persona_name) is None:
                    errors.append(
                        f"Step '{step.name}': persona '{step.persona_name}' not found "
                        "in PersonaManager."
                    )

        return errors
