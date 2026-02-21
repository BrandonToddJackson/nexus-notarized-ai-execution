"""Decision gate: Is the result sufficient, or does the chain need another step?"""

from typing import Any

from nexus.types import ChainPlan, Seal, ReasoningDecision, ActionStatus

_MAX_RETRIES = 2


class ContinueCompleteGate:
    """Decision gate: continue chain or mark complete?"""

    def __init__(self):
        # Track per-(chain_id, step_index) retry counts across calls
        self._retry_counts: dict[str, int] = {}

    def decide(self, chain: ChainPlan, latest_result: Any, latest_seal: Seal) -> ReasoningDecision:
        """Decide whether to continue the chain or mark it complete.

        Logic:
        - If latest_seal.status == FAILED: return RETRY (if retries < 2) or ESCALATE
        - If chain has more steps: return CONTINUE
        - If chain is complete: return COMPLETE

        Args:
            chain: Current chain plan
            latest_result: Result from the most recent step
            latest_seal: Seal from the most recent step

        Returns:
            ReasoningDecision.CONTINUE, .COMPLETE, .RETRY, or .ESCALATE
        """
        if latest_seal.status == ActionStatus.FAILED:
            key = f"{chain.id}:{latest_seal.step_index}"
            count = self._retry_counts.get(key, 0) + 1
            self._retry_counts[key] = count
            if count < _MAX_RETRIES:
                return ReasoningDecision.RETRY
            return ReasoningDecision.ESCALATE

        # More steps remain in the plan
        if len(chain.seals) < len(chain.steps):
            return ReasoningDecision.CONTINUE

        return ReasoningDecision.COMPLETE
