"""Decision gate: Does the agent have enough context to act, or should it retrieve more?

Prevents the agent from acting on insufficient information.
Circuit breaker prevents infinite retrieval loops.
"""

from nexus.types import RetrievedContext, ReasoningDecision


class ThinkActGate:
    """Decision gate: think more or act now?"""

    def __init__(self, confidence_threshold: float = 0.80, max_think_loops: int = 3):
        """
        Args:
            confidence_threshold: Minimum context confidence to ACT (0.0-1.0)
            max_think_loops: Maximum retrieval loops before forcing ACT
        """
        self.confidence_threshold = confidence_threshold
        self.max_think_loops = max_think_loops

    def decide(self, context: RetrievedContext, loop_count: int = 0) -> ReasoningDecision:
        """Decide whether to think (retrieve more) or act (execute tool).

        Logic:
        - If context.confidence >= threshold AND loop_count < max: ACT
        - If context.confidence < threshold AND loop_count < max: THINK (retrieve more)
        - If loop_count >= max: ACT anyway (circuit breaker)

        Args:
            context: Current retrieval context with confidence score
            loop_count: How many retrieval loops have already happened

        Returns:
            ReasoningDecision.THINK or ReasoningDecision.ACT
        """
        # Circuit breaker: force ACT after max_think_loops regardless of confidence
        if loop_count >= self.max_think_loops:
            return ReasoningDecision.ACT

        if context.confidence >= self.confidence_threshold:
            return ReasoningDecision.ACT

        return ReasoningDecision.THINK
