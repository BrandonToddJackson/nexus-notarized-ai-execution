"""Decision gate: Can the agent retry/fallback, or must it escalate to a human?"""

from datetime import datetime, timezone
from typing import Any

from nexus.types import ChainPlan, ReasoningDecision
from nexus.exceptions import ToolError

# Substrings in the error message that indicate a transient (retryable) failure
_TRANSIENT_KEYWORDS = frozenset([
    "timeout", "timed out", "rate limit", "rate_limit",
    "429", "503", "502", "connection", "temporarily unavailable",
    "too many requests", "server error",
])

_MAX_RETRIES = 2


class EscalateGate:
    """Decision gate: retry, fallback, or escalate to human?"""

    def decide(self, error: Exception, retry_count: int, chain: ChainPlan) -> ReasoningDecision:
        """Decide whether to retry, use fallback, or escalate.

        Logic:
        - If retry_count < 2 AND error is transient (timeout, rate limit): RETRY
        - If error is ToolError AND retry_count < 2: RETRY with different tool
        - Otherwise: ESCALATE with full context

        Args:
            error: The exception that occurred
            retry_count: How many retries have been attempted
            chain: Current chain plan for context

        Returns:
            ReasoningDecision.RETRY or ReasoningDecision.ESCALATE
        """
        if retry_count >= _MAX_RETRIES:
            return ReasoningDecision.ESCALATE

        # Check for transient / retryable errors
        error_str = str(error).lower()
        is_transient = (
            isinstance(error, (TimeoutError, ConnectionError, OSError))
            or any(kw in error_str for kw in _TRANSIENT_KEYWORDS)
        )
        if is_transient:
            return ReasoningDecision.RETRY

        # ToolErrors (bad params, tool not found) may be retryable with adjusted input
        if isinstance(error, ToolError):
            return ReasoningDecision.RETRY

        return ReasoningDecision.ESCALATE

    def build_escalation_context(self, chain: ChainPlan, error: Exception) -> dict:
        """Build human-readable escalation context.

        Includes: what was tried, what failed, recommendation for human.

        Args:
            chain: The chain that needs escalation
            error: The error that triggered escalation

        Returns:
            Dict with escalation details for human review
        """
        steps_completed = len(chain.seals)
        steps_total = len(chain.steps)
        remaining_steps = chain.steps[steps_completed:]

        return {
            "chain_id": chain.id,
            "tenant_id": chain.tenant_id,
            "task": chain.task,
            "escalated_at": datetime.now(timezone.utc).isoformat(),
            "progress": {
                "steps_completed": steps_completed,
                "steps_total": steps_total,
                "completion_pct": round(steps_completed / max(steps_total, 1) * 100, 1),
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "details": getattr(error, "details", {}),
            },
            "completed_seal_ids": list(chain.seals),
            "remaining_steps": remaining_steps,
            "recommendation": _build_recommendation(error, steps_completed, steps_total),
        }


def _build_recommendation(error: Exception, completed: int, total: int) -> str:
    """Generate a human-readable recommendation based on the error type."""
    if isinstance(error, ToolError):
        tool = getattr(error, "tool_name", "unknown")
        return (
            f"Tool '{tool}' failed after retries. "
            "Verify the tool is available and parameters are correct, then retry."
        )
    error_str = str(error).lower()
    if any(kw in error_str for kw in _TRANSIENT_KEYWORDS):
        return (
            "Transient infrastructure error persisted across retries. "
            "Check service health and retry after a short wait."
        )
    if completed == 0:
        return "Chain failed before executing any steps. Review task decomposition and persona configuration."
    return (
        f"Chain failed at step {completed}/{total}. "
        "Review the error, correct the configuration, and resume from the last successful step."
    )
