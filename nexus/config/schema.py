"""Pydantic models for YAML configuration validation.

These mirror nexus/types.py structures but accept string inputs
(e.g., risk_level: "low") and coerce them to the correct enums.
"""

from pydantic import BaseModel, Field, field_validator

from nexus.types import RiskLevel, TrustTier


class PersonaYAML(BaseModel):
    """Validated schema for a persona entry in personas.yaml."""

    name: str
    description: str
    allowed_tools: list[str]
    resource_scopes: list[str]
    intent_patterns: list[str]
    risk_tolerance: RiskLevel = RiskLevel.MEDIUM
    max_ttl_seconds: int = 120
    trust_tier: TrustTier = TrustTier.COLD_START

    @field_validator("risk_tolerance", mode="before")
    @classmethod
    def coerce_risk(cls, v):
        if isinstance(v, str):
            return RiskLevel(v.lower())
        return v

    @field_validator("trust_tier", mode="before")
    @classmethod
    def coerce_tier(cls, v):
        if isinstance(v, str):
            return TrustTier(v.lower())
        return v


class ToolYAML(BaseModel):
    """Validated schema for a tool entry in tools.yaml."""

    name: str
    description: str
    risk_level: RiskLevel = RiskLevel.LOW
    resource_pattern: str = "*"
    timeout_seconds: int = 30
    requires_approval: bool = False

    @field_validator("risk_level", mode="before")
    @classmethod
    def coerce_risk(cls, v):
        if isinstance(v, str):
            return RiskLevel(v.lower())
        return v


class PersonasConfig(BaseModel):
    """Root schema for personas.yaml."""
    personas: list[PersonaYAML] = Field(default_factory=list)


class ToolsConfig(BaseModel):
    """Root schema for tools.yaml."""
    tools: list[ToolYAML] = Field(default_factory=list)
