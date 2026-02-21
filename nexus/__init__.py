"""NEXUS — Notarized AI Execution.

Usage:
    from nexus import Nexus, PersonaContract, ToolDefinition

    nx = Nexus()
    result = await nx.run("Analyze customer churn from Q3 data", persona="analyst")
"""

from nexus.types import (
    Seal, ChainPlan, PersonaContract, ToolDefinition, SkillDefinition,
    GateResult, AnomalyResult, IntentDeclaration, KnowledgeDocument,
    GateVerdict, ActionStatus, ChainStatus, TrustTier, RiskLevel,
    ReasoningDecision, CostRecord, RetrievedContext,
)
from nexus.exceptions import (
    NexusError, AnomalyDetected, PersonaViolation, GateFailure,
    ChainAborted, ToolError, BudgetExceeded, SealIntegrityError,
    EscalationRequired,
)
from nexus.version import __version__

__all__ = [
    "Seal", "ChainPlan", "PersonaContract", "ToolDefinition", "SkillDefinition",
    "GateResult", "AnomalyResult", "IntentDeclaration", "KnowledgeDocument",
    "GateVerdict", "ActionStatus", "ChainStatus", "TrustTier", "RiskLevel",
    "ReasoningDecision", "CostRecord", "RetrievedContext",
    "NexusError", "AnomalyDetected", "PersonaViolation", "GateFailure",
    "ChainAborted", "ToolError", "BudgetExceeded", "SealIntegrityError",
    "EscalationRequired",
    "__version__",
]
