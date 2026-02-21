"""Pydantic models for API request/response. Mirrors types.py for API I/O."""

from pydantic import BaseModel, Field
from typing import Any, Optional


# ── Requests ──

class ExecuteRequest(BaseModel):
    task: str = Field(..., max_length=10000)  # "Analyze customer churn from Q3 data"
    persona: Optional[str] = None             # "analyst" — if None, engine auto-selects
    config: Optional[dict] = None             # override config per request


class CreatePersonaRequest(BaseModel):
    name: str
    description: str
    allowed_tools: list[str]
    resource_scopes: list[str]
    intent_patterns: list[str]
    max_ttl_seconds: int = Field(default=120, ge=1, le=86400)
    risk_tolerance: str = "medium"


class IngestRequest(BaseModel):
    content: str = Field(..., max_length=500000)
    namespace: str = "default"
    source: str = "upload"
    access_level: str = "internal"
    metadata: dict = {}


class KnowledgeQueryRequest(BaseModel):
    query: str
    namespace: str = "default"
    n_results: int = Field(default=5, ge=1, le=50)


class TokenRequest(BaseModel):
    api_key: str


# ── Responses ──

class GateResponse(BaseModel):
    name: str
    verdict: str
    score: float
    threshold: float

class SealResponse(BaseModel):
    id: str
    step_index: int
    persona: str
    tool: str
    status: str
    gates: list[GateResponse]
    reasoning: list[str]                # CoT trace
    result: Any = None
    created_at: str
    error: Optional[str] = None

class CostSummary(BaseModel):
    input_tokens: int
    output_tokens: int
    total_cost_usd: float

class ExecuteResponse(BaseModel):
    chain_id: str
    status: str                         # ChainStatus value
    seals: list[SealResponse]           # all seals in order
    result: Any                         # final output
    cost: CostSummary
    duration_ms: int

class PersonaResponse(BaseModel):
    id: str
    name: str
    description: str
    allowed_tools: list[str]
    resource_scopes: list[str]
    intent_patterns: list[str]
    max_ttl_seconds: int
    risk_tolerance: str
    version: int
    is_active: bool

class ToolResponse(BaseModel):
    name: str
    description: str
    risk_level: str
    resource_pattern: str
    requires_approval: bool

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, bool]

class TokenResponse(BaseModel):
    token: str
    tenant_id: str
    expires_in: int
