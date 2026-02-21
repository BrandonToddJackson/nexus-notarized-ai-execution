"""Manages multi-step task decomposition and execution tracking.

A ChainPlan is an immutable execution plan. Once created, steps CANNOT be modified.
The chain tracks which steps have been completed via seal_ids.
"""

from datetime import datetime
from typing import Optional, Any

from nexus.types import ChainPlan, ChainStatus


class ChainManager:
    """Manages multi-step task decomposition and execution tracking."""

    def create_chain(self, tenant_id: str, task: str, steps: list[dict]) -> ChainPlan:
        """Create immutable chain plan.

        Args:
            tenant_id: Tenant context
            task: Original user request
            steps: Planned steps from LLM decomposition.
                   Format: [{"action": "...", "tool": "...", "params": {...}, "persona": "..."}, ...]

        Returns:
            New ChainPlan in PLANNING status
        """
        return ChainPlan(
            tenant_id=tenant_id,
            task=task,
            steps=steps,
            status=ChainStatus.PLANNING,
        )

    def advance(self, chain: ChainPlan, seal_id: str) -> ChainPlan:
        """Record completed step. Append seal_id, update status.

        Args:
            chain: Current chain plan
            seal_id: ID of the seal for the completed step

        Returns:
            Updated chain plan
        """
        updated_seals = chain.seals + [seal_id]
        new_status = ChainStatus.EXECUTING
        completed_at = None
        if len(updated_seals) >= len(chain.steps):
            new_status = ChainStatus.COMPLETED
            completed_at = datetime.utcnow()
        return chain.model_copy(update={
            "seals": updated_seals,
            "status": new_status,
            "completed_at": completed_at,
        })

    def fail(self, chain: ChainPlan, error: str) -> ChainPlan:
        """Mark chain as failed. Record error and completed_at.

        Args:
            chain: Current chain plan
            error: Error message

        Returns:
            Updated chain with FAILED status
        """
        return chain.model_copy(update={
            "status": ChainStatus.FAILED,
            "error": error,
            "completed_at": datetime.utcnow(),
        })

    def escalate(self, chain: ChainPlan, reason: str) -> ChainPlan:
        """Mark chain as escalated. Record reason.

        Args:
            chain: Current chain plan
            reason: Why escalation is needed

        Returns:
            Updated chain with ESCALATED status
        """
        return chain.model_copy(update={
            "status": ChainStatus.ESCALATED,
            "error": reason,
            "completed_at": datetime.utcnow(),
        })

    def get_current_step(self, chain: ChainPlan) -> Optional[dict]:
        """Return the next unexecuted step, or None if all done.

        Args:
            chain: Current chain plan

        Returns:
            Next step dict, or None if chain is complete
        """
        current_index = len(chain.seals)
        if current_index >= len(chain.steps):
            return None
        return chain.steps[current_index]

    def is_complete(self, chain: ChainPlan) -> bool:
        """True if all steps have seals.

        Args:
            chain: Current chain plan

        Returns:
            True if len(chain.seals) >= len(chain.steps)
        """
        return len(chain.seals) >= len(chain.steps)
