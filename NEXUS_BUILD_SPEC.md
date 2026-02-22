# NEXUS BUILD SPECIFICATION
## Complete Engineering Blueprint — Zero Questions, Zero Guessing

> **For:** Claude Code (or any engineer)
> **Goal:** Build a working, shippable AI agent framework in <24 hours
> **Rule:** If something isn't specified here, it's not in v1. Don't improvise.

---

## BUILD PROGRESS

| Phase | Status | Tests | Notes |
|-------|--------|-------|-------|
| Phase 0: Foundation | ✅ COMPLETE | — | types, exceptions, config, version |
| Phase 1: Core Security | ✅ COMPLETE | 15/15 ✓ | personas, anomaly (4 gates), notary, ledger, chain, verifier, output_validator, cot_logger — **real assertions** |
| Phase 2: Core Cognitive | ✅ COMPLETE | 46/46 ✓ | embeddings (lazy), store (ChromaDB), context, think_act, continue_complete, escalate — **real assertions** |
| Phase 3: Execution Layer | ✅ COMPLETE | 14/14 ✓ | registry, selector, sandbox, executor, plugin decorator, builtins, skills — **real assertions** |
| Phase 4: Engine | ✅ COMPLETE | 7/7 ✓ | engine.py wires Phases 1-3 — **real assertions** |
| Phase 5: Persistence | ✅ COMPLETE | 26/26 ✓ | repository (all 17 methods), seed (idempotent, 5 personas), smoketest in tests/test_repository.py |
| Phase 6: LLM | ✅ COMPLETE | ✓ | litellm client, prompts.py, cost_tracker — wired to engine per-step |
| Phase 7: Cache | ✅ COMPLETE | 10/10 ✓ | FingerprintCache (Redis list, max 1000, frequency map); wired to engine for Gate 4 drift baseline |
| Phase 8: Auth | ✅ COMPLETE | ✓ | rate_limiter, jwt, middleware — all implemented and tested |
| Phase 9: API | ✅ COMPLETE | ✓ | All route handlers: execute, stream (SSE), ledger, personas, tools, knowledge, health, auth |
| Phase 10: CLI | ✅ COMPLETE | ✓ | 11 commands: init, run, dev, seed, verify, replay, inspect, audit, gates, config, tools |
| Phase 11: Frontend | ✅ COMPLETE | — | React + Vite + Tailwind (17 files): Execute, Ledger, Personas, Tools, Knowledge, Settings, Dashboard, Login; SSE streaming GateVisualizer; auth flow |
| Phase 12: Infrastructure | ✅ COMPLETE | — | Dockerfile, docker-compose.yml, docker-compose.quickstart.yml, CI pipeline, Makefile |
| Phase 13: Tests | ✅ COMPLETE | 590/590 ✓ | All phases real assertions; 0 stubs; 0 skips; trust, locks, config/loader covered |
| Phase 14: Examples & Docs | ✅ COMPLETE | — | quickstart, custom_tool, local_llm, customer_support, code_review; docs/quickstart.md, philosophy.md, architecture.md, comparison.md, api-reference.md, tutorials/ |

**Last verified:** 2026-02-22
**Test suite:** `.venv312/bin/pytest tests/ -v` → **590/590 passed**

**Audit fixes applied (2026-02-21):**
- `ledger.py`: `Seal(**s.__dict__)` → filter `_sa_instance_state` (SQLAlchemy internal key was crashing DB retrieval)
- `engine.py`: Gate 4 fingerprints now stored after each EXECUTED seal via `fingerprint_store.store()` — drift detection was permanently disabled before this fix
- `output_validator.py`: `EMAIL_PATTERN` now used in `validate()` alongside SSN and CC checks
- `notary.py`: `verify_chain()` defensively sorts seals by step_index before verification
- `context.py`: bare `except` now logs warning before falling back
- `stream.py`: SSE task properly awaited on client disconnect (was leaking background tasks)
- `verifier.py`: resource target mismatches now logged (were silently discarded)
- `cli/verify.py`: fixed unreachable "CHAIN COMPROMISED" branch — `verify_chain()` raises, never returns False
- `cli/run.py`: fixed duplicate ToolExecutor instantiation (second instance was passed to NexusEngine, losing registry)
- `cli/main.py`: `--version` flag now works (needed `is_eager=True` + `invoke_without_command=True`)
- `api/routes/execute.py`: AnomalyDetected now returns proper ExecuteResponse (not `{"detail":"..."}`)

**Known non-blockers:**
- `datetime.utcnow()` deprecation warnings (503 in test run) — harmless in 3.12, fix before 3.14
- Gate 4 drift uses frequency distribution, not time-series — acceptable for v1
- Sandbox is timeout-only isolation (no filesystem jail) — v2 scope per spec

---

## TABLE OF CONTENTS

1. [What NEXUS Is (30 seconds)](#1-what-nexus-is)
2. [Tech Stack (locked)](#2-tech-stack)
3. [Build Order (strict)](#3-build-order)
4. [Phase 0: Foundation](#4-phase-0-foundation)
5. [Phase 1: Core Security](#5-phase-1-core-security)
6. [Phase 2: Core Cognitive](#6-phase-2-core-cognitive)
7. [Phase 3: Execution Layer](#7-phase-3-execution-layer)
8. [Phase 4: Engine (Integration)](#8-phase-4-engine)
9. [Phase 5: Persistence](#9-phase-5-persistence)
10. [Phase 6: LLM Integration](#10-phase-6-llm-integration)
11. [Phase 7: Cache Layer](#11-phase-7-cache)
12. [Phase 8: Auth](#12-phase-8-auth)
13. [Phase 9: API](#13-phase-9-api)
14. [Phase 10: CLI](#14-phase-10-cli)
15. [Phase 11: Frontend](#15-phase-11-frontend)
16. [Phase 12: Infrastructure](#16-phase-12-infrastructure)
17. [Phase 13: Tests](#17-phase-13-tests)
18. [Phase 14: Examples & Docs](#18-phase-14-examples-and-docs)
19. [Integration Wiring Diagram](#19-integration-wiring)
20. [Acceptance Criteria](#20-acceptance-criteria)

---

## 1. What NEXUS Is

NEXUS is an AI agent framework where **every action is notarized**. Before an agent executes anything, it must: declare intent, assume a behavioral persona, pass 4 anomaly gates, get the action sealed in an immutable ledger. If anything looks wrong, the action is blocked.

**One-liner:** "The agent framework where AI actions are accountable."

**NOT multi-agent.** One agent, multiple personas (behavioral contracts). A persona is not a separate entity — it's a constrained operating mode with allowed tools, resource scopes, and intent patterns.

---

## 2. Tech Stack (Locked — Do Not Substitute)

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.11+ |
| API | FastAPI + Uvicorn | latest |
| Database | PostgreSQL | 15+ |
| ORM | SQLAlchemy (async) | 2.0+ |
| Migrations | Alembic | latest |
| Cache | Redis | 7+ |
| Vector Store | ChromaDB (default, swappable) | latest |
| LLM | litellm (multi-provider) | latest |
| Embeddings | sentence-transformers (default) | latest |
| Frontend | React + Vite + Tailwind | latest |
| CLI | Typer + Rich | latest |
| Auth | PyJWT | latest |
| Validation | Pydantic v2 | latest |
| Testing | pytest + pytest-asyncio | latest |
| Container | Docker + docker-compose | latest |

**Key decision: Use `litellm` for LLM abstraction.** It already handles Anthropic, OpenAI, Ollama, 100+ providers. Don't build a custom LLM client. This saves ~200 lines and eliminates provider bugs.

---

## 3. Build Order (Strict — Follow This Sequence)

```
Phase 0: Foundation     → types.py, exceptions.py, config.py, __init__.py
Phase 1: Core Security  → personas, anomaly, notary, ledger, chain, verifier, output_validator, cot_logger
Phase 2: Core Cognitive → knowledge store, embeddings, context builder, reasoning gates
Phase 3: Execution      → tool registry, plugin decorator, selector, sandbox, executor, built-in tools, skills
Phase 4: Engine         → engine.py (wires Phases 1-3 together)
Phase 5: Persistence    → DB models, repository, migrations, seed
Phase 6: LLM            → litellm wrapper, prompts, intent embeddings, cost tracker
Phase 7: Cache          → Redis client, fingerprint cache, locks
Phase 8: Auth           → JWT, middleware, API keys, rate limits
Phase 9: API            → FastAPI app, all routes, schemas, streaming, events
Phase 10: CLI           → Typer commands, project templates
Phase 11: Frontend      → React app, all pages, components
Phase 12: Infrastructure→ Dockerfile, docker-compose, CI, Makefile
Phase 13: Tests         → All test files
Phase 14: Examples/Docs → README, quickstart, tutorials, working examples
```

**Why this order:** Each phase only imports from previous phases. No circular dependencies. No forward references. Claude Code can build sequentially and test each phase before moving on.

---

## 4. Phase 0: Foundation

### `nexus/types.py`

```python
"""All shared types, enums, and type aliases. Everything imports from here."""

from enum import Enum
from typing import TypedDict, Any, Optional
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
```

### `nexus/exceptions.py`

```python
"""Typed exception hierarchy. Every error NEXUS can raise."""

class NexusError(Exception):
    """Base exception for all NEXUS errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

class AnomalyDetected(NexusError):
    """One or more anomaly gates failed."""
    def __init__(self, message: str, gate_results: list = None, **kwargs):
        super().__init__(message, **kwargs)
        self.gate_results = gate_results or []

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
```

### `nexus/config.py`

```python
"""Application configuration. All env vars defined here with defaults."""

from pydantic_settings import BaseSettings
from typing import Optional

class NexusConfig(BaseSettings):
    # ── App ──
    app_name: str = "nexus"
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = "change-me-in-production"

    # ── Database ──
    database_url: str = "postgresql+asyncpg://nexus:nexus@localhost:5432/nexus"

    # ── Redis ──
    redis_url: str = "redis://localhost:6379/0"

    # ── LLM (litellm) ──
    default_llm_model: str = "anthropic/claude-sonnet-4-20250514"
    llm_api_key: Optional[str] = None          # set ANTHROPIC_API_KEY or OPENAI_API_KEY in env
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1               # low temp for deterministic declarations

    # ── Embeddings ──
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    embedding_dimensions: int = 384

    # ── Vector Store ──
    chroma_persist_dir: str = "./data/chroma"

    # ── Anomaly Gates ──
    gate_intent_threshold: float = 0.75         # cosine similarity minimum
    gate_drift_sigma: float = 2.5               # standard deviations for drift detection
    gate_default_ttl: int = 120                 # seconds

    # ── Cost ──
    default_budget_usd: float = 50.0            # per tenant per month
    budget_alert_pct: float = 0.8               # alert at 80%

    # ── Auth ──
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    api_key_prefix: str = "nxs_"

    # ── Rate Limits ──
    rate_limit_requests_per_minute: int = 60
    rate_limit_chains_per_hour: int = 100

    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]  # Vite dev server

    model_config = {"env_prefix": "NEXUS_", "env_file": ".env", "extra": "ignore"}

config = NexusConfig()
```

### `nexus/__init__.py`

```python
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
```

### `nexus/version.py`

```python
__version__ = "0.1.0"
```

---

## 5. Phase 1: Core Security

### `nexus/core/personas.py`

**Public API:**

```python
class PersonaManager:
    """Manages persona lifecycle: load, activate, validate, revoke."""

    def __init__(self, personas: list[PersonaContract]):
        """Load persona contracts (from YAML or DB)."""

    def activate(self, persona_id: str, tenant_id: str) -> PersonaContract:
        """
        Activate a persona for use. Returns the contract.
        Raises PersonaViolation if persona doesn't exist or is disabled.
        Sets activation timestamp for TTL tracking.
        """

    def validate_action(self, persona: PersonaContract, tool_name: str, resource_targets: list[str]) -> bool:
        """
        Check if action is within persona's behavioral contract.
        - tool_name must be in persona.allowed_tools
        - each resource_target must match at least one persona.resource_scopes pattern
        Returns True or raises PersonaViolation with specific reason.
        """

    def revoke(self, persona_id: str) -> None:
        """Deactivate persona. Called after every action (ephemeral identity)."""

    def get_ttl_remaining(self, persona_id: str) -> int:
        """Seconds remaining before persona TTL expires."""
```

**Implementation notes:**
- Store active personas in a dict with activation timestamps
- Resource scope matching: glob patterns (e.g., `db:analytics.*` matches `db:analytics.customers`)
- TTL is checked in the anomaly gate, not here — but PersonaManager tracks the activation time

### `nexus/core/anomaly.py`

**Public API:**

```python
class AnomalyEngine:
    """4-gate anomaly detection. The security core of NEXUS."""

    def __init__(self, config: NexusConfig, embedding_service=None):
        """
        config: gate thresholds
        embedding_service: for Gate 2 intent matching (injected from Phase 6)
        """

    async def check(
        self,
        persona: PersonaContract,
        intent: IntentDeclaration,
        activation_time: datetime,
    ) -> AnomalyResult:
        """
        Run all 4 gates. Returns AnomalyResult.
        Gates run IN ORDER. If Gate 1 fails, still run all gates for diagnostics.
        """

    def _gate1_scope(self, persona: PersonaContract, intent: IntentDeclaration) -> GateResult:
        """
        SCOPE CHECK: Is tool_name in persona.allowed_tools?
        Are resource_targets within persona.resource_scopes?
        Score: 1.0 if all match, 0.0 if any don't.
        Threshold: 1.0 (binary pass/fail).
        """

    async def _gate2_intent(self, persona: PersonaContract, intent: IntentDeclaration) -> GateResult:
        """
        INTENT SIMILARITY: Embed intent.planned_action, compare cosine similarity
        against persona.intent_patterns embeddings.
        Score: max cosine similarity across all intent_patterns.
        Threshold: config.gate_intent_threshold (default 0.75).
        If no embedding_service, SKIP this gate (cold start mode).
        """

    def _gate3_ttl(self, persona: PersonaContract, activation_time: datetime) -> GateResult:
        """
        TTL CHECK: Has persona been active longer than max_ttl_seconds?
        Score: remaining_seconds / max_ttl_seconds (1.0 = just activated).
        Threshold: 0.0 (any time remaining = pass).
        """

    def _gate4_drift(self, persona: PersonaContract, intent: IntentDeclaration) -> GateResult:
        """
        BEHAVIORAL DRIFT: Is this action's fingerprint within N sigma of
        the persona's historical baseline?
        Fingerprint = hash(tool_name + sorted(resource_targets) + intent_category).
        Score: how many sigma from mean. Lower is better.
        Threshold: config.gate_drift_sigma (default 2.5).
        If <10 historical samples, SKIP (insufficient baseline).
        """
```

**Implementation notes:**
- Gate 2 uses `embedding_service` which may be None at startup. If None, gate returns SKIP verdict.
- Gate 4 needs historical fingerprints. Accept an optional `fingerprint_store` (dict or Redis). If empty, SKIP.
- `AnomalyResult.overall_verdict` = FAIL if ANY gate verdict is FAIL. SKIP gates don't cause failure.
- Fingerprint for drift: `hashlib.sha256(f"{tool_name}:{sorted_targets}:{intent_category}".encode()).hexdigest()[:16]`

### `nexus/core/notary.py`

**Public API:**

```python
class Notary:
    """Creates and verifies immutable seals. The audit backbone."""

    def __init__(self):
        self._last_fingerprint: str = ""  # Merkle chain state

    def create_seal(
        self,
        chain_id: str,
        step_index: int,
        tenant_id: str,
        persona_id: str,
        intent: IntentDeclaration,
        anomaly_result: AnomalyResult,
    ) -> Seal:
        """
        Create a new seal in PENDING status.
        Fingerprint = SHA256(previous_fingerprint + seal_content_hash).
        This creates the Merkle chain — any tampering breaks the chain.
        """

    def finalize_seal(self, seal: Seal, tool_result: Any, status: ActionStatus, error: str = None) -> Seal:
        """
        Finalize seal after execution. Updates status, result, completed_at.
        Does NOT recompute fingerprint — that was set at creation.
        """

    def verify_chain(self, seals: list[Seal]) -> bool:
        """
        Verify Merkle chain integrity. Recompute each fingerprint
        and check it matches. Returns False if any seal was tampered with.
        Raises SealIntegrityError with details on which seal broke.
        """

    @staticmethod
    def _compute_fingerprint(previous: str, seal_content: str) -> str:
        """SHA256(previous_fingerprint + SHA256(seal_content))"""
```

**Implementation notes:**
- `seal_content` for hashing = `f"{seal.chain_id}:{seal.step_index}:{seal.tenant_id}:{seal.persona_id}:{seal.intent.tool_name}:{seal.intent.tool_params}:{seal.anomaly_result.overall_verdict}"`
- Merkle chain ensures you can't delete or modify a seal without breaking all subsequent fingerprints

### `nexus/core/ledger.py`

**Public API:**

```python
class Ledger:
    """Immutable audit log. Append-only. All seals go here."""

    def __init__(self, repository=None):
        """repository: injected DB repository for persistence (Phase 5)."""
        self._memory_store: list[Seal] = []  # in-memory fallback

    async def append(self, seal: Seal) -> None:
        """Append seal. If repository available, persist. Always keep in memory."""

    async def get_chain(self, chain_id: str) -> list[Seal]:
        """Get all seals for a chain, ordered by step_index."""

    async def get_by_tenant(self, tenant_id: str, limit: int = 100, offset: int = 0) -> list[Seal]:
        """Paginated seal history for a tenant."""

    async def verify_integrity(self, chain_id: str) -> bool:
        """Verify Merkle chain integrity for all seals in a chain."""
```

### `nexus/core/chain.py`

**Public API:**

```python
class ChainManager:
    """Manages multi-step task decomposition and execution tracking."""

    def create_chain(self, tenant_id: str, task: str, steps: list[dict]) -> ChainPlan:
        """
        Create immutable chain plan.
        steps format: [{"action": "search_kb", "tool": "knowledge_search", "params": {...}}, ...]
        Once created, steps CANNOT be modified (immutable plan).
        """

    def advance(self, chain: ChainPlan, seal_id: str) -> ChainPlan:
        """Record completed step. Append seal_id, update status."""

    def fail(self, chain: ChainPlan, error: str) -> ChainPlan:
        """Mark chain as failed. Record error and completed_at."""

    def escalate(self, chain: ChainPlan, reason: str) -> ChainPlan:
        """Mark chain as escalated. Record reason."""

    def get_current_step(self, chain: ChainPlan) -> Optional[dict]:
        """Return the next unexecuted step, or None if all done."""

    def is_complete(self, chain: ChainPlan) -> bool:
        """True if all steps have seals."""
```

### `nexus/core/verifier.py`

**Public API:**

```python
class IntentVerifier:
    """Cross-checks declared intent against actual tool parameters.
    Prevents the LLM from declaring 'I want to read a file' then executing 'rm -rf'.
    """

    def verify(self, intent: IntentDeclaration, tool_name: str, tool_params: dict) -> bool:
        """
        Check:
        1. intent.tool_name == tool_name (exact match)
        2. intent.tool_params keys are subset of tool_params keys
        3. intent.resource_targets match tool_params values that look like resources
           (file paths, URLs, database tables)
        Returns True or raises PersonaViolation.
        """
```

### `nexus/core/output_validator.py`

**Public API:**

```python
class OutputValidator:
    """Validates tool output matches the declared intent."""

    async def validate(self, intent: IntentDeclaration, tool_result: Any) -> tuple[bool, str]:
        """
        Returns (is_valid, reason).
        Checks:
        1. Result is not None/empty (unless intent was a delete)
        2. Basic PII scan: regex for SSN (\\d{3}-\\d{2}-\\d{4}),
           credit card (\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}),
           email patterns in unexpected places
        3. If result is text, basic coherence check (not just error messages)
        """
```

### `nexus/core/cot_logger.py`

**Public API:**

```python
class CoTLogger:
    """Captures chain-of-thought reasoning between tool calls."""

    def __init__(self):
        self._traces: dict[str, list[str]] = {}  # seal_id -> reasoning steps

    def log(self, seal_id: str, step: str) -> None:
        """Append a reasoning step."""

    def get_trace(self, seal_id: str) -> list[str]:
        """Return all reasoning steps for a seal."""

    def clear(self, seal_id: str) -> None:
        """Clear trace after seal is finalized."""
```

---

## 6. Phase 2: Core Cognitive

### `nexus/knowledge/store.py`

**Public API:**

```python
class KnowledgeStore:
    """Tenant-scoped vector store for RAG retrieval."""

    def __init__(self, persist_dir: str, embedding_fn=None):
        """
        persist_dir: ChromaDB persistence directory
        embedding_fn: callable that takes list[str] -> list[list[float]]
        """

    async def ingest(self, document: KnowledgeDocument) -> str:
        """
        Chunk document.content (500 chars, 50 char overlap).
        Embed chunks. Store in ChromaDB collection named '{tenant_id}_{namespace}'.
        Return document ID.
        """

    async def query(
        self, tenant_id: str, namespace: str, query: str,
        access_level: str = "internal", n_results: int = 5
    ) -> RetrievedContext:
        """
        Embed query. Search ChromaDB collection. Filter by access_level.
        Return RetrievedContext with documents, confidence, sources.
        Confidence = average similarity score of top results.
        """

    async def delete(self, tenant_id: str, namespace: str, document_id: str) -> None:
        """Remove a document and its chunks from the store."""

    def list_namespaces(self, tenant_id: str) -> list[str]:
        """List all knowledge namespaces for a tenant."""
```

### `nexus/knowledge/embeddings.py`

**Public API:**

```python
class EmbeddingService:
    """Unified embedding service. Used by RAG AND Gate 2 intent matching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load sentence-transformers model."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of vectors."""

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts. Returns 0.0-1.0."""

    def similarities(self, query: str, candidates: list[str]) -> list[float]:
        """Cosine similarity of query against each candidate."""
```

### `nexus/knowledge/context.py`

**Public API:**

```python
class ContextBuilder:
    """Assembles retrieval context for each action. This is what makes the agent useful."""

    def __init__(self, knowledge_store: KnowledgeStore):
        pass

    async def build(
        self,
        tenant_id: str,
        task: str,
        persona: PersonaContract,
        session_history: list[dict] = None,
    ) -> RetrievedContext:
        """
        1. Determine which namespaces this persona can access (from resource_scopes)
        2. Query knowledge store across allowed namespaces
        3. Merge results, deduplicate, rank by relevance
        4. Include session_history (previous chain step results) for continuity
        5. Return assembled context with confidence score
        """
```

### `nexus/reasoning/think_act.py`

**Public API:**

```python
class ThinkActGate:
    """Decision gate: Does the agent have enough context to act, or should it retrieve more?"""

    def __init__(self, confidence_threshold: float = 0.80, max_think_loops: int = 3):
        pass

    def decide(self, context: RetrievedContext, loop_count: int = 0) -> ReasoningDecision:
        """
        If context.confidence >= threshold AND loop_count < max: ACT
        If context.confidence < threshold AND loop_count < max: THINK (retrieve more)
        If loop_count >= max: ACT anyway (circuit breaker — prevent infinite loops)
        Returns ReasoningDecision.THINK or .ACT
        """
```

### `nexus/reasoning/continue_complete.py`

**Public API:**

```python
class ContinueCompleteGate:
    """Decision gate: Is the result sufficient, or does the chain need another step?"""

    def decide(self, chain: ChainPlan, latest_result: Any, latest_seal: Seal) -> ReasoningDecision:
        """
        If latest_seal.status == FAILED: return RETRY (if retries < 2) or ESCALATE
        If chain has more steps: return CONTINUE
        If chain is complete: return COMPLETE
        """
```

### `nexus/reasoning/escalate.py`

**Public API:**

```python
class EscalateGate:
    """Decision gate: Can the agent retry/fallback, or must it escalate to a human?"""

    def decide(self, error: Exception, retry_count: int, chain: ChainPlan) -> ReasoningDecision:
        """
        If retry_count < 2 AND error is transient (timeout, rate limit): RETRY
        If error is ToolError AND alternative tool exists: RETRY with different tool
        Otherwise: ESCALATE with full context (what was tried, what failed, recommendation)
        """

    def build_escalation_context(self, chain: ChainPlan, error: Exception) -> dict:
        """Build human-readable escalation context with all attempts and recommendations."""
```

---

## 7. Phase 3: Execution Layer

### `nexus/tools/registry.py`

**Public API:**

```python
class ToolRegistry:
    """Central registry of all available tools."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._implementations: dict[str, callable] = {}

    def register(self, definition: ToolDefinition, implementation: callable) -> None:
        """Register a tool with its definition and implementation function."""

    def get(self, name: str) -> tuple[ToolDefinition, callable]:
        """Get tool definition and implementation. Raises ToolError if not found."""

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""

    def list_for_persona(self, persona: PersonaContract) -> list[ToolDefinition]:
        """List tools available to a specific persona."""

    def get_schema_for_llm(self, persona: PersonaContract) -> list[dict]:
        """
        Format tool definitions for LLM function calling.
        Only includes tools in persona.allowed_tools.
        Returns list of dicts matching OpenAI/Anthropic tool format.
        """
```

### `nexus/tools/plugin.py`

**Public API:**

```python
def tool(
    name: str = None,
    description: str = None,
    risk_level: RiskLevel = RiskLevel.LOW,
    resource_pattern: str = "*",
    timeout_seconds: int = 30,
    requires_approval: bool = False,
):
    """
    Decorator to register a function as a NEXUS tool.

    Usage:
        @tool(name="web_search", description="Search the web", risk_level=RiskLevel.LOW)
        async def web_search(query: str) -> str:
            ...

    Auto-generates ToolDefinition from function signature + type hints.
    Extracts JSON Schema from type annotations for parameters.
    """
```

### `nexus/tools/selector.py`

**Public API:**

```python
class ToolSelector:
    """Generates tool execution plans from task + context."""

    def __init__(self, registry: ToolRegistry, llm_client=None):
        pass

    async def select(
        self,
        task: str,
        persona: PersonaContract,
        context: RetrievedContext,
    ) -> list[IntentDeclaration]:
        """
        Given a task, available tools (filtered by persona), and retrieved context,
        ask the LLM to generate an ordered list of tool calls with parameters.
        Returns list of IntentDeclarations (the execution plan).
        If llm_client is None, use rule-based matching (for testing).
        """
```

### `nexus/tools/sandbox.py`

**Public API:**

```python
class Sandbox:
    """Constrained execution environment for tool calls."""

    async def execute(
        self,
        tool_fn: callable,
        params: dict,
        timeout: int = 30,
    ) -> Any:
        """
        Execute tool function with constraints:
        - asyncio.wait_for with timeout
        - Catch and wrap all exceptions as ToolError
        - Log execution time
        Returns tool result or raises ToolError.
        """
```

**Implementation note:** For v1, "sandbox" is timeout + error wrapping. True filesystem/network isolation is v2. The architecture slot exists but the implementation is lightweight.

### `nexus/tools/executor.py`

**Public API:**

```python
class ToolExecutor:
    """Orchestrates: validate → sandbox execute → capture result."""

    def __init__(self, registry: ToolRegistry, sandbox: Sandbox, verifier: IntentVerifier):
        pass

    async def execute(self, intent: IntentDeclaration) -> tuple[Any, Optional[str]]:
        """
        1. Get tool from registry
        2. Verify intent matches tool call (IntentVerifier)
        3. Execute in sandbox
        4. Return (result, error_string_or_None)
        """
```

### Built-in Tools: `nexus/tools/builtin/`

Each built-in tool is a decorated function:

```python
# nexus/tools/builtin/web.py
@tool(name="web_search", description="Search the web for information", risk_level=RiskLevel.LOW)
async def web_search(query: str) -> str:
    """Uses a simple HTTP request to a search API. Returns text results."""
    # v1: stub that returns "Search results for: {query}"
    # Production: integrate with Serper, Tavily, or similar
    return f"Search results for: {query}"

@tool(name="web_fetch", description="Fetch content from a URL", risk_level=RiskLevel.LOW)
async def web_fetch(url: str) -> str:
    """Fetch webpage content."""
    # v1: httpx.get(url).text[:5000]
    pass
```

```python
# nexus/tools/builtin/files.py
@tool(name="file_read", description="Read a file's contents", risk_level=RiskLevel.LOW, resource_pattern="file:*")
async def file_read(path: str) -> str:
    pass

@tool(name="file_write", description="Write content to a file", risk_level=RiskLevel.MEDIUM, resource_pattern="file:*")
async def file_write(path: str, content: str) -> str:
    pass
```

```python
# nexus/tools/builtin/comms.py
@tool(name="send_email", description="Send an email", risk_level=RiskLevel.HIGH, requires_approval=True)
async def send_email(to: str, subject: str, body: str) -> str:
    # v1: stub that logs the email content
    return f"Email queued to {to}: {subject}"
```

```python
# nexus/tools/builtin/data.py
@tool(name="compute_stats", description="Compute statistics on data", risk_level=RiskLevel.LOW)
async def compute_stats(data: list, metrics: list[str] = None) -> dict:
    pass

@tool(name="knowledge_search", description="Search the knowledge base", risk_level=RiskLevel.LOW, resource_pattern="kb:*")
async def knowledge_search(query: str, namespace: str = "default") -> str:
    # This tool is special — it calls KnowledgeStore.query() internally
    # Wired up in Engine (Phase 4)
    pass
```

**CRITICAL:** Built-in tools in v1 can be stubs that return formatted strings. They need to be **registered** and **callable** so the full pipeline works end-to-end. Actual integrations (real web search, real email) are v1.1.

### `nexus/skills/definitions.py`

```python
# Default skills loaded from YAML or defined in code
DEFAULT_SKILLS = [
    SkillDefinition(
        name="research",
        description="Search knowledge base and web for information",
        tool_sequence=["knowledge_search", "web_search"],
        persona="researcher",
    ),
    SkillDefinition(
        name="analyze_data",
        description="Query data and compute statistics",
        tool_sequence=["knowledge_search", "compute_stats"],
        persona="analyst",
    ),
    SkillDefinition(
        name="write_and_send",
        description="Draft content and send via email",
        tool_sequence=["knowledge_search", "file_write", "send_email"],
        persona="communicator",
    ),
]
```

---

## 8. Phase 4: Engine (Integration)

### `nexus/core/engine.py`

**This is the most important file. It wires EVERYTHING together.**

```python
class NexusEngine:
    """
    Single entry point. Orchestrates: decompose → retrieve → plan → [gate → execute → validate] per step.

    Constructor dependencies (all injected):
        - persona_manager: PersonaManager
        - anomaly_engine: AnomalyEngine
        - notary: Notary
        - ledger: Ledger
        - chain_manager: ChainManager
        - knowledge_store: KnowledgeStore
        - context_builder: ContextBuilder
        - tool_registry: ToolRegistry
        - tool_selector: ToolSelector
        - tool_executor: ToolExecutor
        - output_validator: OutputValidator
        - cot_logger: CoTLogger
        - think_act_gate: ThinkActGate
        - continue_complete_gate: ContinueCompleteGate
        - escalate_gate: EscalateGate
        - llm_client: LLMClient (from Phase 6)
        - config: NexusConfig
    """

    async def run(self, task: str, tenant_id: str, persona_name: str = None) -> ChainPlan:
        """
        FULL EXECUTION LOOP:

        1. DECOMPOSE: Ask LLM to break task into chain steps
           → chain = chain_manager.create_chain(tenant_id, task, steps)

        2. For each step in chain:

           a. RETRIEVE: Build context for this step
              → context = context_builder.build(tenant_id, step, persona, session_history)

           b. THINK/ACT GATE: Enough context?
              → decision = think_act_gate.decide(context, loop_count)
              → if THINK: loop back to (a) with refined query (max 3 loops)

           c. PLAN: Select tool for this step
              → intent = tool_selector.select(step, persona, context)

           d. ACTIVATE PERSONA: Ephemeral identity
              → persona = persona_manager.activate(persona_name, tenant_id)

           e. GATE CHECK: Run 4 anomaly gates
              → anomaly_result = anomaly_engine.check(persona, intent, activation_time)

           f. CREATE SEAL: Notarize intent (PENDING status)
              → seal = notary.create_seal(chain_id, step_index, tenant_id, ...)

           g. If anomaly_result.overall_verdict == FAIL:
              → seal = notary.finalize_seal(seal, None, BLOCKED)
              → ledger.append(seal)
              → persona_manager.revoke(persona_name)
              → raise AnomalyDetected(...)

           h. VERIFY: Cross-check intent vs actual params
              → verifier.verify(intent, tool_name, tool_params)

           i. EXECUTE: Run tool in sandbox
              → result, error = tool_executor.execute(intent)

           j. VALIDATE OUTPUT: Check result
              → is_valid, reason = output_validator.validate(intent, result)

           k. FINALIZE SEAL: Record result
              → seal = notary.finalize_seal(seal, result, EXECUTED or FAILED)
              → ledger.append(seal)

           l. REVOKE PERSONA: Destroy identity
              → persona_manager.revoke(persona_name)

           m. CONTINUE/COMPLETE GATE: Next step?
              → decision = continue_complete_gate.decide(chain, result, seal)
              → if CONTINUE: advance chain, loop to next step
              → if COMPLETE: done
              → if ESCALATE: escalate_gate.build_escalation_context(...)

        3. Return completed chain with all seal IDs
        """

    async def _decompose_task(self, task: str, tenant_id: str) -> list[dict]:
        """
        Ask LLM to break task into steps.
        Prompt (see Phase 6 prompts.py for exact text):
          "Break this task into 1-5 concrete steps.
           Each step: {action, tool, params, persona}.
           Return JSON array."
        Parse LLM response as JSON.
        """
```

**CRITICAL implementation detail:** The engine is a big async function with a for-loop over chain steps, calling the right components in order. It's orchestration code, not complex logic. Most of the intelligence is in the components it calls.

---

## 9. Phase 5: Persistence

### `nexus/db/models.py`

**SQLAlchemy models — these map 1:1 to the Pydantic types but are ORM models:**

```python
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean, Text, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime
import uuid

class Base(DeclarativeBase):
    pass

class TenantModel(Base):
    __tablename__ = "tenants"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    api_key_hash = Column(String, nullable=False)
    budget_usd = Column(Float, default=50.0)
    budget_used_usd = Column(Float, default=0.0)
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class PersonaModel(Base):
    __tablename__ = "personas"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    allowed_tools = Column(JSON, default=list)  # list[str]
    resource_scopes = Column(JSON, default=list)  # list[str]
    intent_patterns = Column(JSON, default=list)  # list[str]
    max_ttl_seconds = Column(Integer, default=120)
    risk_tolerance = Column(String, default="medium")
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_persona_tenant_name", "tenant_id", "name", unique=True),)

class SealModel(Base):
    __tablename__ = "seals"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chain_id = Column(String, ForeignKey("chains.id"), nullable=False, index=True)
    step_index = Column(Integer, nullable=False)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    persona_id = Column(String, nullable=False)
    intent = Column(JSON, nullable=False)  # IntentDeclaration serialized
    anomaly_result = Column(JSON, nullable=False)  # AnomalyResult serialized
    tool_name = Column(String, nullable=False)
    tool_params = Column(JSON, default=dict)
    tool_result = Column(JSON, nullable=True)
    status = Column(String, default="pending")
    cot_trace = Column(JSON, default=list)
    fingerprint = Column(String, nullable=False)
    parent_fingerprint = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

    __table_args__ = (Index("ix_seal_chain_step", "chain_id", "step_index"),)

class ChainModel(Base):
    __tablename__ = "chains"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    task = Column(Text, nullable=False)
    steps = Column(JSON, nullable=False)  # planned steps
    status = Column(String, default="planning")
    seal_ids = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

class CostModel(Base):
    __tablename__ = "costs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    chain_id = Column(String, nullable=False)
    seal_id = Column(String, nullable=True)
    model = Column(String, nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)

class KnowledgeDocModel(Base):
    __tablename__ = "knowledge_docs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    namespace = Column(String, nullable=False)
    source = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)  # hash of content for dedup
    access_level = Column(String, default="internal")
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_knowledge_tenant_ns", "tenant_id", "namespace"),)
```

### `nexus/db/database.py`

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from nexus.config import config

engine = create_async_engine(config.database_url, echo=config.debug)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session

async def init_db():
    from nexus.db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

### `nexus/db/repository.py`

**Public API:**

```python
class Repository:
    """All database operations. Every query is tenant-scoped."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── Tenants ──
    async def get_tenant(self, tenant_id: str) -> TenantModel
    async def create_tenant(self, name: str, api_key_hash: str) -> TenantModel

    # ── Personas ──
    async def list_personas(self, tenant_id: str) -> list[PersonaModel]
    async def get_persona(self, tenant_id: str, persona_id: str) -> PersonaModel
    async def get_persona_by_name(self, tenant_id: str, name: str) -> PersonaModel
    async def upsert_persona(self, tenant_id: str, data: dict) -> PersonaModel

    # ── Seals ──
    async def create_seal(self, seal: Seal) -> SealModel
    async def update_seal(self, seal_id: str, updates: dict) -> SealModel
    async def list_seals(self, tenant_id: str, limit: int = 100, offset: int = 0) -> list[SealModel]
    async def get_chain_seals(self, chain_id: str) -> list[SealModel]

    # ── Chains ──
    async def create_chain(self, chain: ChainPlan) -> ChainModel
    async def update_chain(self, chain_id: str, updates: dict) -> ChainModel
    async def list_chains(self, tenant_id: str, limit: int = 50) -> list[ChainModel]

    # ── Costs ──
    async def add_cost(self, cost: CostRecord) -> CostModel
    async def get_tenant_cost(self, tenant_id: str, month: str = None) -> float

    # ── Knowledge ──
    async def add_knowledge_doc(self, doc: KnowledgeDocument) -> KnowledgeDocModel
    async def list_knowledge_docs(self, tenant_id: str, namespace: str = None) -> list[KnowledgeDocModel]
```

### `nexus/db/seed.py`

```python
async def seed_database(session: AsyncSession):
    """
    Create demo data:
    - 1 tenant: "demo" with API key "nxs_demo_key_12345"
    - 5 default personas: researcher, analyst, creator, communicator, operator
    - Register all built-in tools
    """
```

Exact persona definitions for seed:

```python
DEFAULT_PERSONAS = [
    {
        "name": "researcher",
        "description": "Searches and retrieves information from knowledge bases and the web",
        "allowed_tools": ["knowledge_search", "web_search", "web_fetch", "file_read"],
        "resource_scopes": ["kb:*", "web:*", "file:read:*"],
        "intent_patterns": ["search for information", "find data about", "look up", "research"],
        "risk_tolerance": "low",
        "max_ttl_seconds": 60,
    },
    {
        "name": "analyst",
        "description": "Analyzes data and computes statistics",
        "allowed_tools": ["knowledge_search", "compute_stats", "file_read", "file_write"],
        "resource_scopes": ["kb:*", "file:*", "data:*"],
        "intent_patterns": ["analyze data", "compute statistics", "calculate", "summarize findings"],
        "risk_tolerance": "medium",
        "max_ttl_seconds": 120,
    },
    {
        "name": "creator",
        "description": "Creates content: documents, reports, summaries",
        "allowed_tools": ["knowledge_search", "file_write"],
        "resource_scopes": ["kb:*", "file:write:*"],
        "intent_patterns": ["write", "create", "draft", "generate content", "compose"],
        "risk_tolerance": "low",
        "max_ttl_seconds": 90,
    },
    {
        "name": "communicator",
        "description": "Sends emails and messages",
        "allowed_tools": ["knowledge_search", "send_email", "file_read"],
        "resource_scopes": ["kb:*", "email:*", "file:read:*"],
        "intent_patterns": ["send email", "notify", "communicate", "message"],
        "risk_tolerance": "high",
        "max_ttl_seconds": 60,
    },
    {
        "name": "operator",
        "description": "Executes code and system operations",
        "allowed_tools": ["knowledge_search", "file_read", "file_write", "compute_stats"],
        "resource_scopes": ["kb:*", "file:*", "system:*"],
        "intent_patterns": ["execute", "run", "deploy", "configure", "operate"],
        "risk_tolerance": "high",
        "max_ttl_seconds": 180,
    },
]
```

---

## 10. Phase 6: LLM Integration

### `nexus/llm/client.py`

```python
import litellm
from nexus.config import config

class LLMClient:
    """Thin wrapper around litellm for NEXUS-specific usage."""

    def __init__(self, model: str = None):
        self.model = model or config.default_llm_model
        litellm.drop_params = True  # ignore unsupported params per provider

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = None,
        max_tokens: int = None,
        response_format: dict = None,
    ) -> dict:
        """
        Call LLM via litellm.acompletion().
        Returns {"content": str, "tool_calls": list, "usage": {"input_tokens": int, "output_tokens": int}}.
        Wraps litellm exceptions as NexusError.
        """
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=temperature or config.llm_temperature,
            max_tokens=max_tokens or config.llm_max_tokens,
            response_format=response_format,
        )
        # Extract and normalize response
        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "tool_calls": choice.message.tool_calls or [],
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        }
```

### `nexus/llm/prompts.py`

**Every prompt template used by the system. No magic strings anywhere else.**

```python
DECOMPOSE_TASK = """You are a task planner for an AI agent system.
Break the following task into 1-5 concrete steps.
Each step should be a single tool call.

Available tools: {tool_list}
Available personas: {persona_list}

Task: {task}

Respond with a JSON array:
[
  {{"action": "description of step", "tool": "tool_name", "params": {{}}, "persona": "persona_name"}},
  ...
]

Rules:
- Each step uses exactly ONE tool
- Choose the most appropriate persona for each step
- Keep it simple — fewer steps is better
- Parameters should be specific and actionable
"""

DECLARE_INTENT = """You are about to execute a tool call. Declare your intent.

Task context: {task_context}
Retrieved knowledge: {retrieved_context}
Current step: {current_step}
Tool: {tool_name}
Available parameters: {tool_schema}

Respond with JSON:
{{
  "planned_action": "what you intend to do",
  "tool_params": {{}},
  "resource_targets": ["list of resources you'll access"],
  "reasoning": "why you chose this action",
  "confidence": 0.0-1.0
}}
"""

SELECT_TOOL = """Given this task step, choose the best tool and parameters.

Step: {step_description}
Available tools: {tool_list}
Context: {context}

Respond with JSON:
{{
  "tool": "tool_name",
  "params": {{}},
  "reasoning": "why this tool"
}}
"""
```

### `nexus/llm/cost_tracker.py`

```python
class CostTracker:
    """Track LLM token usage per tenant."""

    # litellm provides cost calculation via litellm.completion_cost()

    async def record(self, tenant_id: str, chain_id: str, seal_id: str, model: str, usage: dict) -> CostRecord:
        """
        Calculate cost using litellm.completion_cost().
        Check if tenant is over budget.
        If over 80%: log warning.
        If over 100%: raise BudgetExceeded.
        Return CostRecord.
        """
```

---

## 11. Phase 7: Cache

### `nexus/cache/redis_client.py`

```python
import redis.asyncio as redis
from nexus.config import config

class RedisClient:
    """Tenant-namespaced Redis operations."""

    def __init__(self):
        self.pool = redis.ConnectionPool.from_url(config.redis_url)
        self.client = redis.Redis(connection_pool=self.pool)

    def _key(self, tenant_id: str, key: str) -> str:
        return f"nexus:{tenant_id}:{key}"

    async def get(self, tenant_id: str, key: str) -> Optional[str]: ...
    async def set(self, tenant_id: str, key: str, value: str, ttl: int = None) -> None: ...
    async def delete(self, tenant_id: str, key: str) -> None: ...
    async def incr(self, tenant_id: str, key: str) -> int: ...
    async def health(self) -> bool: ...
```

### `nexus/cache/fingerprints.py`

```python
class FingerprintCache:
    """Stores behavioral fingerprints for Gate 4 drift detection."""

    def __init__(self, redis: RedisClient):
        pass

    async def store(self, tenant_id: str, persona_id: str, fingerprint: str) -> None:
        """Append fingerprint to persona's history (max 1000)."""

    async def get_baseline(self, tenant_id: str, persona_id: str) -> dict:
        """Return {mean_fingerprint_frequency, std_dev, sample_count}."""
```

---

## 12. Phase 8: Auth

### `nexus/auth/jwt.py`

```python
class JWTManager:
    async def create_token(self, tenant_id: str, role: str = "user") -> str: ...
    async def verify_token(self, token: str) -> dict: ...  # returns {"tenant_id": str, "role": str}
```

### `nexus/auth/middleware.py`

```python
# FastAPI middleware that:
# 1. Checks Authorization header (Bearer JWT or API key starting with "nxs_")
# 2. Extracts tenant_id
# 3. Sets request.state.tenant_id
# 4. Returns 401 if invalid
```

### `nexus/auth/rate_limiter.py`

```python
class RateLimiter:
    """Redis-backed rate limiting."""

    def __init__(self, redis: RedisClient):
        pass

    async def check(self, tenant_id: str, action: str = "api") -> bool:
        """
        Increment counter in Redis with TTL.
        action="api": config.rate_limit_requests_per_minute per minute
        action="chain": config.rate_limit_chains_per_hour per hour
        Returns True if allowed, raises NexusError if exceeded.
        """
```

---

## 13. Phase 9: API

### `nexus/api/main.py`

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init_db, connect Redis, load personas, register tools
    yield
    # Shutdown: close connections

app = FastAPI(title="NEXUS", version="0.1.0", lifespan=lifespan)
# Mount: auth middleware, CORS, /v1/ router
```

### Route Contracts

**POST /v1/execute**
```python
# Request:
class ExecuteRequest(BaseModel):
    task: str                           # "Analyze customer churn from Q3 data"
    persona: Optional[str] = None       # "analyst" — if None, engine auto-selects
    config: Optional[dict] = None       # override config per request

# Response:
class ExecuteResponse(BaseModel):
    chain_id: str
    status: str                         # ChainStatus value
    seals: list[SealResponse]           # all seals in order
    result: Any                         # final output
    cost: CostSummary
    duration_ms: int

class SealResponse(BaseModel):
    id: str
    step_index: int
    persona: str
    tool: str
    status: str
    gates: list[GateResponse]
    reasoning: list[str]                # CoT trace
    created_at: str

class GateResponse(BaseModel):
    name: str
    verdict: str
    score: float
    threshold: float

class CostSummary(BaseModel):
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
```

**POST /v1/execute/stream** — Same request, SSE response:
```
event: chain_started
data: {"chain_id": "...", "steps": 3}

event: step_started
data: {"step": 0, "persona": "researcher", "tool": "knowledge_search"}

event: gate_result
data: {"step": 0, "gate": "scope", "verdict": "pass", "score": 1.0}

event: gate_result
data: {"step": 0, "gate": "intent", "verdict": "pass", "score": 0.89}

event: gate_result
data: {"step": 0, "gate": "ttl", "verdict": "pass", "score": 0.95}

event: gate_result
data: {"step": 0, "gate": "drift", "verdict": "skip", "score": 0.0}

event: seal_created
data: {"step": 0, "seal_id": "...", "status": "executed", "result_preview": "..."}

event: step_completed
data: {"step": 0}

... (repeat for each step)

event: chain_completed
data: {"chain_id": "...", "status": "completed", "cost": {...}}
```

**Other routes (standard CRUD, abbreviated):**

| Method | Path | Purpose |
|--------|------|---------|
| GET | /v1/ledger | Paginated seal history |
| GET | /v1/ledger/{chain_id} | Seals for a specific chain |
| GET | /v1/personas | List personas |
| POST | /v1/personas | Create persona |
| PUT | /v1/personas/{id} | Update persona |
| GET | /v1/tools | List registered tools |
| POST | /v1/knowledge/ingest | Upload document to knowledge base |
| GET | /v1/knowledge/query | Query knowledge base |
| GET | /v1/health | Health check |
| POST | /v1/auth/token | Generate JWT from API key |

---

## 14. Phase 10: CLI

```bash
nexus init my-project      # Scaffold project directory
nexus dev                   # Start API + frontend in dev mode
nexus run "task text"       # Execute a task, print seal summary
nexus persona list          # List configured personas
nexus persona add           # Interactive persona creation
nexus seed                  # Seed database with defaults
```

`nexus init` generates:
```
my-project/
├── .env                    # From template, with TODO placeholders
├── personas.yaml           # Default 5 personas
├── tools.yaml              # Built-in tool config
├── main.py                 # from nexus import Nexus; ...
├── knowledge/              # Empty dir for knowledge docs
└── docker-compose.yml      # Copy of quickstart compose
```

---

## 15. Phase 11: Frontend

**React + Vite + Tailwind. Single-page app. 10 pages.**

| Page | Route | Purpose |
|------|-------|---------|
| Login | /login | API key or JWT login |
| Dashboard | / | Live chain feed, gate visualizer, cost ticker |
| Execute | /execute | Text input to run tasks, see results stream |
| Ledger | /ledger | Filterable audit trail, expand seals |
| Chain Detail | /chain/:id | Full chain with step-by-step visualization |
| Personas | /personas | CRUD persona contracts |
| Tools | /tools | View registered tools + skills |
| Knowledge | /knowledge | Upload docs, view namespaces |
| Settings | /settings | API keys, LLM config, budget |
| 404 | * | Not found |

**Key component: GateVisualizer** — Shows 4 gates as a horizontal pipeline. Each gate lights green/red/gray as results stream in via SSE. This is the visual signature of NEXUS.

**Key component: SealCard** — Compact card showing: persona name, tool used, gate results (4 dots), status badge, CoT toggle.

**CRITICAL:** Frontend must connect to SSE endpoint for real-time updates. Use EventSource API. Reconnect with exponential backoff on disconnect.

---

## 16. Phase 12: Infrastructure

### `docker-compose.yml`

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres, redis, chroma]

  frontend:
    build: ./frontend
    ports: ["5173:5173"]
    depends_on: [api]

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: nexus
      POSTGRES_PASSWORD: nexus
      POSTGRES_DB: nexus
    ports: ["5432:5432"]
    volumes: [pgdata:/var/lib/postgresql/data]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  chroma:
    image: chromadb/chroma:latest
    ports: ["8001:8000"]
    volumes: [chromadata:/chroma/chroma]

volumes:
  pgdata:
  chromadata:
```

### `Dockerfile`

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]"
COPY . .

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
USER nobody
EXPOSE 8000
CMD ["uvicorn", "nexus.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `Makefile`

```makefile
dev:        docker compose up --build
test:       pytest tests/ -v
lint:       ruff check . && mypy nexus/
seed:       python -m nexus.db.seed
migrate:    alembic upgrade head
clean:      docker compose down -v
```

---

## 19. Integration Wiring

```
User Request
    │
    ▼
[API: /v1/execute]
    │
    ▼
[Auth Middleware] ──→ extracts tenant_id from JWT/API key
    │
    ▼
[Rate Limiter] ──→ checks Redis counter
    │
    ▼
[NexusEngine.run()]
    │
    ├─→ [LLMClient.complete(DECOMPOSE_TASK)] ──→ returns chain steps
    │
    ├─→ [ChainManager.create_chain()] ──→ immutable plan
    │
    │   ┌──── FOR EACH STEP ────────────────────────────┐
    │   │                                                 │
    │   ├─→ [ContextBuilder.build()]                      │
    │   │     └─→ [KnowledgeStore.query()]                │
    │   │     └─→ [EmbeddingService.embed()]              │
    │   │                                                 │
    │   ├─→ [ThinkActGate.decide()]                       │
    │   │     └─→ if THINK: loop back to ContextBuilder   │
    │   │                                                 │
    │   ├─→ [ToolSelector.select()]                       │
    │   │     └─→ [LLMClient.complete(DECLARE_INTENT)]    │
    │   │                                                 │
    │   ├─→ [PersonaManager.activate()]                   │
    │   │                                                 │
    │   ├─→ [AnomalyEngine.check()]                       │
    │   │     ├─→ Gate 1: scope check (PersonaManager)    │
    │   │     ├─→ Gate 2: intent sim (EmbeddingService)   │
    │   │     ├─→ Gate 3: TTL check (activation_time)     │
    │   │     └─→ Gate 4: drift (FingerprintCache)        │
    │   │                                                 │
    │   ├─→ [Notary.create_seal()] ──→ PENDING            │
    │   │                                                 │
    │   ├─→ if FAIL: finalize BLOCKED, raise AnomalyDetected
    │   │                                                 │
    │   ├─→ [IntentVerifier.verify()]                     │
    │   │                                                 │
    │   ├─→ [ToolExecutor.execute()]                      │
    │   │     └─→ [Sandbox.execute(tool_fn, params)]      │
    │   │                                                 │
    │   ├─→ [OutputValidator.validate()]                   │
    │   │                                                 │
    │   ├─→ [Notary.finalize_seal()] ──→ EXECUTED         │
    │   │                                                 │
    │   ├─→ [Ledger.append(seal)]                         │
    │   │     └─→ [Repository.create_seal()]              │
    │   │                                                 │
    │   ├─→ [CostTracker.record()]                        │
    │   │                                                 │
    │   ├─→ [PersonaManager.revoke()]                     │
    │   │                                                 │
    │   ├─→ [ContinueCompleteGate.decide()]               │
    │   │     └─→ CONTINUE: next step                     │
    │   │     └─→ COMPLETE: break loop                    │
    │   │     └─→ ESCALATE: raise EscalationRequired      │
    │   │                                                 │
    │   └─────────────────────────────────────────────────┘
    │
    ▼
[Return ExecuteResponse with chain + seals + cost]
```

---

## 20. Acceptance Criteria

**The system is "working" when ALL of these pass:**

1. **`docker compose up` starts all services** — API, Postgres, Redis, ChromaDB, frontend all healthy

2. **`POST /v1/execute {"task": "What is NEXUS?"}` returns a completed chain** with:
   - At least 1 seal with status "executed"
   - All 4 gate results present in each seal
   - Valid Merkle fingerprint chain
   - CoT trace with at least 1 entry
   - Cost tracking > 0 tokens

3. **A multi-step task returns multiple seals** — e.g., "Research AI trends and write a summary" should produce 2+ seals (research step + writing step)

4. **An out-of-scope action gets BLOCKED** — e.g., researcher persona trying to use send_email tool → seal with status "blocked", Gate 1 verdict "fail"

5. **SSE streaming works** — `/v1/execute/stream` sends gate_result events in real-time

6. **Frontend shows the dashboard** — live chain feed, gate visualizer rendering, seal cards

7. **Ledger is queryable** — `GET /v1/ledger` returns historical seals, filterable by tenant

8. **Tenant isolation** — two different API keys see different data, knowledge, and personas

9. **CLI works** — `nexus run "hello"` executes a task and prints the seal summary

10. **Tests pass** — `pytest tests/` with >80% of tests green

---

## APPENDIX: File Checklist

Build these files in this exact order. Check each off as completed.

```
[ ] nexus/__init__.py
[ ] nexus/version.py
[ ] nexus/types.py
[ ] nexus/exceptions.py
[ ] nexus/config.py
[ ] nexus/core/personas.py
[ ] nexus/core/anomaly.py
[ ] nexus/core/notary.py
[ ] nexus/core/ledger.py
[ ] nexus/core/chain.py
[ ] nexus/core/verifier.py
[ ] nexus/core/output_validator.py
[ ] nexus/core/cot_logger.py
[ ] nexus/knowledge/embeddings.py
[ ] nexus/knowledge/store.py
[ ] nexus/knowledge/context.py
[ ] nexus/reasoning/think_act.py
[ ] nexus/reasoning/continue_complete.py
[ ] nexus/reasoning/escalate.py
[ ] nexus/tools/plugin.py
[ ] nexus/tools/registry.py
[ ] nexus/tools/selector.py
[ ] nexus/tools/sandbox.py
[ ] nexus/tools/executor.py
[ ] nexus/tools/builtin/web.py
[ ] nexus/tools/builtin/files.py
[ ] nexus/tools/builtin/comms.py
[ ] nexus/tools/builtin/data.py
[ ] nexus/skills/definitions.py
[ ] nexus/core/engine.py
[ ] nexus/db/models.py
[ ] nexus/db/database.py
[ ] nexus/db/repository.py
[ ] nexus/db/seed.py
[ ] nexus/llm/client.py
[ ] nexus/llm/prompts.py
[ ] nexus/llm/cost_tracker.py
[ ] nexus/cache/redis_client.py
[ ] nexus/cache/fingerprints.py
[ ] nexus/auth/jwt.py
[ ] nexus/auth/middleware.py
[ ] nexus/auth/rate_limiter.py
[ ] nexus/api/main.py
[ ] nexus/api/schemas.py (mirrors types.py for API I/O)
[ ] nexus/api/routes/execute.py
[ ] nexus/api/routes/stream.py
[ ] nexus/api/routes/ledger.py
[ ] nexus/api/routes/personas.py
[ ] nexus/api/routes/tools.py
[ ] nexus/api/routes/knowledge.py
[ ] nexus/api/routes/health.py
[ ] nexus/api/routes/auth.py
[ ] nexus/cli/main.py
[ ] nexus/cli/commands/init.py
[ ] nexus/cli/commands/run.py
[ ] nexus/cli/commands/dev.py
[ ] nexus/cli/commands/seed.py
[ ] frontend/package.json
[ ] frontend/src/App.jsx
[ ] frontend/src/lib/api.js
[ ] frontend/src/lib/auth.js
[ ] frontend/src/lib/stream.js
[ ] frontend/src/pages/Login.jsx
[ ] frontend/src/pages/Dashboard.jsx
[ ] frontend/src/pages/Execute.jsx
[ ] frontend/src/pages/Ledger.jsx
[ ] frontend/src/pages/Personas.jsx
[ ] frontend/src/pages/Tools.jsx
[ ] frontend/src/pages/Knowledge.jsx
[ ] frontend/src/pages/Settings.jsx
[ ] frontend/src/components/SealCard.jsx
[ ] frontend/src/components/GateVisualizer.jsx
[ ] frontend/src/components/ChainView.jsx
[ ] frontend/src/components/LiveFeed.jsx
[ ] Dockerfile
[ ] docker-compose.yml
[ ] pyproject.toml
[ ] .env.example
[ ] Makefile
[ ] README.md
[ ] tests/conftest.py
[ ] tests/test_engine.py
[ ] tests/test_anomaly.py
[ ] tests/test_tools.py
[ ] tests/test_api.py
[ ] examples/quickstart/main.py
```

**Total: ~80 files. This is the MVP. Ship this, then iterate.**
