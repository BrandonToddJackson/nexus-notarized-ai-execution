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


# ── Phase 15: Workflow, Trigger, Credential, MCP Exceptions ─────────────────


class WorkflowError(NexusError):
    """Base exception for all workflow-related errors."""
    pass


class WorkflowNotFound(WorkflowError):
    """Requested workflow does not exist or is not accessible by this tenant."""
    def __init__(self, message: str, workflow_id: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.workflow_id = workflow_id


class WorkflowValidationError(WorkflowError):
    """Workflow definition is structurally invalid (cycles, missing steps, etc.)."""
    def __init__(self, message: str, violations: list = None, **kwargs):
        super().__init__(message, **kwargs)
        self.violations = violations or []


class WorkflowGenerationError(WorkflowError):
    """Raised when the WorkflowGenerator cannot produce a valid workflow from LLM output."""


class TriggerError(NexusError):
    """A workflow trigger failed to fire or configure."""
    def __init__(self, message: str, trigger_type: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.trigger_type = trigger_type


class CredentialError(NexusError):
    """Credential retrieval or decryption failed."""
    def __init__(self, message: str, credential_id: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.credential_id = credential_id


class CredentialNotFound(CredentialError):
    """Credential ID not found or does not belong to this tenant."""
    pass


class MCPConnectionError(NexusError):
    """MCP server connection or tool discovery failed."""
    def __init__(self, message: str, server_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.server_name = server_name


class MCPToolError(ToolError):
    """MCP tool execution failed."""
    pass


class SandboxError(ToolError):
    """Code sandbox execution failed (memory limit, timeout, forbidden import)."""
    pass


class AmbiguityResolutionError(NexusError):
    """
    Raised when AmbiguityResolver cannot proceed.

    Reasons: session not found, session not active, answer validation failure,
    session expired.
    """
    pass
