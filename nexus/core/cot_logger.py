"""Captures chain-of-thought reasoning between tool calls.

Reasoning steps are associated with seal IDs and included in the final seal.
"""


class CoTLogger:
    """Captures chain-of-thought reasoning between tool calls."""

    def __init__(self):
        self._traces: dict[str, list[str]] = {}  # seal_id -> reasoning steps

    def log(self, seal_id: str, step: str) -> None:
        """Append a reasoning step.

        Args:
            seal_id: Seal this reasoning belongs to
            step: Human-readable reasoning step
        """
        if seal_id not in self._traces:
            self._traces[seal_id] = []
        self._traces[seal_id].append(step)

    def get_trace(self, seal_id: str) -> list[str]:
        """Return all reasoning steps for a seal.

        Args:
            seal_id: Seal to get trace for

        Returns:
            Ordered list of reasoning steps
        """
        return self._traces.get(seal_id, [])

    def clear(self, seal_id: str) -> None:
        """Clear trace after seal is finalized.

        Args:
            seal_id: Seal to clear trace for
        """
        self._traces.pop(seal_id, None)
