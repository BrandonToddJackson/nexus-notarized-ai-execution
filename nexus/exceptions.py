"""Typed exception hierarchy. Every error NEXUS can raise."""


class NexusError(Exception):
    """Base exception for all NEXUS errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class AnomalyDetected(NexusError):
    """One or more anomaly gates failed."""
    def __init__(self, message: str, gate_results: list = None, chain_id: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.gate_results = gate_results or []
        self.chain_id = chain_id


class PersonaViolation(NexusError):
    """Action violates persona behavioral contract."""
    pass


class GateFailure(NexusError):
    """A specific gate check failed."""
    def __init__(self, message: str, gate_name: str = "", score: float = 0.0, threshold: float = 0.0, **kwargs):
        super().__init__(message, **kwargs)
        self.gate_name = gate_name
        self.score = score
        self.threshold = threshold


class ChainAborted(NexusError):
    """Chain execution was aborted."""
    def __init__(self, message: str, completed_steps: int = 0, total_steps: int = 0, **kwargs):
        super().__init__(message, **kwargs)
        self.completed_steps = completed_steps
        self.total_steps = total_steps


class ToolError(NexusError):
    """Tool execution failed."""
    def __init__(self, message: str, tool_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class BudgetExceeded(NexusError):
    """Tenant LLM budget exhausted."""
    pass


class SealIntegrityError(NexusError):
    """Seal fingerprint chain is broken — possible tampering."""
    pass


class EscalationRequired(NexusError):
    """Agent cannot handle this; needs human intervention."""
    def __init__(self, message: str, context: dict = None, **kwargs):
        super().__init__(message, **kwargs)
        self.context = context or {}
