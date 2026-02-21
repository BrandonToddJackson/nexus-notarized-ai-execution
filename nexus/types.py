"""All shared types, enums, and type aliases. Everything imports from here."""

from enum import Enum
from typing import Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


# ── Enums ──────────────────────────────────────────────────────────────

class GateVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # gate not applicable (e.g., no drift baseline yet)

class ActionStatus(str, Enum):
    PENDING = "pending"
    DECLARED = "declared"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"
    BLOCKED = "blocked"
    COMPENSATED = "compensated"

class ChainStatus(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # some steps completed, chain aborted
    ESCALATED = "escalated"

class TrustTier(str, Enum):
    COLD_START = "cold_start"       # new persona, no history
    ESTABLISHED = "established"     # 50+ successful actions
    TRUSTED = "trusted"             # 500+ successful, <1% anomaly rate

class RiskLevel(str, Enum):
    LOW = "low"         # read-only operations
    MEDIUM = "medium"   # write operations, reversible
    HIGH = "high"       # write operations, irreversible (send email, delete data)
    CRITICAL = "critical"  # infrastructure changes, financial transactions

class ReasoningDecision(str, Enum):
    THINK = "think"       # need more context, loop back to RAG
    ACT = "act"           # confidence sufficient, execute tool
    CONTINUE = "continue" # result insufficient, next step needed
    COMPLETE = "complete" # result satisfies intent
    ESCALATE = "escalate" # can't handle, hand to human
    RETRY = "retry"       # transient failure, try again

class PersonaStatus(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    REVOKED = "revoked"


# ── Core Data Shapes ───────────────────────────────────────────────────

class GateResult(BaseModel):
    """Result of a single anomaly gate check."""
    gate_name: str                      # "scope", "intent", "ttl", "drift"
    verdict: GateVerdict
    score: float                        # 0.0-1.0, meaning varies by gate
    threshold: float                    # configured threshold for this gate
    details: str                        # human-readable explanation
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AnomalyResult(BaseModel):
    """Combined result of all 4 gates."""
    gates: list[GateResult]             # exactly 4 entries
    overall_verdict: GateVerdict        # FAIL if ANY gate fails
    risk_level: RiskLevel
    persona_id: str
    action_fingerprint: str             # hash of the action for drift comparison

class IntentDeclaration(BaseModel):
    """What the agent declares it intends to do before doing it."""
    task_description: str               # original user task
    planned_action: str                 # what the agent wants to do NOW
    tool_name: str                      # which tool it wants to use
    tool_params: dict[str, Any]         # parameters for the tool
    resource_targets: list[str]         # what resources it will access
    reasoning: str                      # WHY it chose this action (CoT)
    confidence: float = 0.0             # 0.0-1.0, how confident in this plan

class Seal(BaseModel):
    """Immutable notarized record of an action. The core NEXUS artifact."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chain_id: str                       # parent chain
    step_index: int                     # position in chain
    tenant_id: str
    persona_id: str
    intent: IntentDeclaration
    anomaly_result: AnomalyResult
    tool_name: str
    tool_params: dict[str, Any]
    tool_result: Any = None
    status: ActionStatus = ActionStatus.PENDING
    cot_trace: list[str] = Field(default_factory=list)  # reasoning steps
    fingerprint: str = ""               # Merkle chain hash
    parent_fingerprint: str = ""        # previous seal's fingerprint
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class ChainPlan(BaseModel):
    """Immutable execution plan for a multi-step task."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    task: str                           # original user request
    steps: list[dict[str, Any]]         # planned steps from LLM decomposition
    status: ChainStatus = ChainStatus.PLANNING
    seals: list[str] = Field(default_factory=list)  # seal IDs in order
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class PersonaContract(BaseModel):
    """Behavioral contract defining what a persona can and cannot do."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str                           # "researcher", "analyst", "operator"
    description: str
    allowed_tools: list[str]            # tool names this persona can use
    resource_scopes: list[str]          # resource patterns (e.g., "db:analytics.*")
    intent_patterns: list[str]          # natural language intent descriptions for embedding matching
    max_ttl_seconds: int = 120          # maximum time persona can stay active
    risk_tolerance: RiskLevel = RiskLevel.MEDIUM
    trust_tier: TrustTier = TrustTier.COLD_START
    version: int = 1
    is_active: bool = True

class ToolDefinition(BaseModel):
    """Registration record for a tool the agent can use."""
    name: str                           # unique identifier
    description: str                    # what it does (used by LLM for selection)
    parameters: dict[str, Any]          # JSON Schema for params
    risk_level: RiskLevel = RiskLevel.LOW
    resource_pattern: str = "*"         # what resources it accesses
    timeout_seconds: int = 30
    requires_approval: bool = False     # if True, needs human approval for HIGH+ risk

class SkillDefinition(BaseModel):
    """Named bundle of tools with execution order."""
    name: str
    description: str
    tool_sequence: list[str]            # ordered tool names
    persona: str                        # which persona owns this skill

class KnowledgeDocument(BaseModel):
    """A document in the tenant's knowledge base."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    namespace: str                      # logical grouping (e.g., "product_docs")
    source: str                         # filename or URL
    content: str                        # raw text content
    chunks: list[str] = Field(default_factory=list)  # chunked text
    access_level: str = "internal"      # "public", "internal", "restricted", "confidential"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RetrievedContext(BaseModel):
    """Context assembled from RAG retrieval for an action."""
    query: str
    documents: list[dict[str, Any]]     # retrieved chunks with scores
    confidence: float                   # overall retrieval confidence (0.0-1.0)
    sources: list[str]                  # source document IDs
    namespace: str

class CostRecord(BaseModel):
    """Token usage tracking for a single LLM call."""
    tenant_id: str
    chain_id: str
    seal_id: Optional[str] = None
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
