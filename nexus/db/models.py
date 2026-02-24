"""All ORM models. These map 1:1 to the Pydantic types but are SQLAlchemy models.

Tables: tenants, personas, seals, chains, costs, knowledge_docs
All tables have tenant_id for isolation. Indexes on common query patterns.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean, Text, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone
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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)


class PersonaModel(Base):
    __tablename__ = "personas"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    allowed_tools = Column(JSON, default=list)
    resource_scopes = Column(JSON, default=list)
    intent_patterns = Column(JSON, default=list)
    max_ttl_seconds = Column(Integer, default=120)
    risk_tolerance = Column(String, default="medium")
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (Index("ix_persona_tenant_name", "tenant_id", "name", unique=True),)


class SealModel(Base):
    __tablename__ = "seals"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chain_id = Column(String, ForeignKey("chains.id"), nullable=False, index=True)
    step_index = Column(Integer, nullable=False)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    persona_id = Column(String, nullable=False)
    intent = Column(JSON, nullable=False)
    anomaly_result = Column(JSON, nullable=False)
    tool_name = Column(String, nullable=False)
    tool_params = Column(JSON, default=dict)
    tool_result = Column(JSON, nullable=True)
    status = Column(String, default="pending")
    cot_trace = Column(JSON, default=list)
    fingerprint = Column(String, nullable=False)
    parent_fingerprint = Column(String, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

    __table_args__ = (Index("ix_seal_chain_step", "chain_id", "step_index"),)


class ChainModel(Base):
    __tablename__ = "chains"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    task = Column(Text, nullable=False)
    steps = Column(JSON, nullable=False)
    status = Column(String, default="planning")
    seal_ids = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
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
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class KnowledgeDocModel(Base):
    __tablename__ = "knowledge_docs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    namespace = Column(String, nullable=False)
    source = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)
    access_level = Column(String, default="internal")
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (Index("ix_knowledge_tenant_ns", "tenant_id", "namespace"),)


# ── Phase 15: Workflow, Trigger, Credential, MCP ORM Models ─────────────────


class WorkflowModel(Base):
    __tablename__ = "workflows"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    version = Column(Integer, default=1)
    status = Column(String, default="draft")        # WorkflowStatus value
    trigger_config = Column(JSON, default=dict)
    steps = Column(JSON, default=list)              # serialized WorkflowStep list
    edges = Column(JSON, default=list)              # serialized WorkflowEdge list
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String, default="")
    tags = Column(JSON, default=list)
    settings = Column(JSON, default=dict)

    __table_args__ = (
        Index("ix_workflow_tenant_status", "tenant_id", "status"),
        UniqueConstraint("tenant_id", "name", "version", name="uq_workflow_tenant_name_version"),
    )


class WorkflowExecutionModel(Base):
    __tablename__ = "workflow_executions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String, ForeignKey("workflows.id"), nullable=False, index=True)
    workflow_version = Column(Integer, nullable=False)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    trigger_type = Column(String, nullable=False)   # TriggerType value
    trigger_data = Column(JSON, default=dict)
    chain_id = Column(String, default="")
    status = Column(String, default="planning")     # ChainStatus value
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)
    step_results = Column(JSON, default=dict)

    __table_args__ = (Index("ix_wf_exec_tenant_workflow_started", "tenant_id", "workflow_id", "started_at"),)


class TriggerModel(Base):
    __tablename__ = "triggers"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String, ForeignKey("workflows.id"), nullable=False, index=True)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    trigger_type = Column(String, nullable=False)   # TriggerType value
    enabled = Column(Boolean, default=True)
    config = Column(JSON, default=dict)
    webhook_path = Column(String, nullable=True, unique=True)
    last_triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_trigger_tenant_enabled", "tenant_id", "enabled"),
        Index("ix_trigger_webhook_path", "webhook_path"),
    )


class CredentialModel(Base):
    __tablename__ = "credentials"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    credential_type = Column(String, nullable=False)    # CredentialType value
    service_name = Column(String, nullable=False)
    encrypted_data = Column(Text, nullable=False)       # AES-256-GCM ciphertext
    scoped_personas = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_credential_tenant_name"),
        Index("ix_credential_tenant_service", "tenant_id", "service_name"),
    )


class AmbiguitySessionModel(Base):
    """Persists an ambiguity clarification session. Added in Phase 23.1."""
    __tablename__ = "ambiguity_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    original_description = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="active")
    questions_json = Column(JSON, nullable=False, default=list)
    answers_json = Column(JSON, nullable=False, default=list)
    current_round = Column(Integer, nullable=False, default=1)
    max_rounds = Column(Integer, nullable=False, default=3)
    specificity_history_json = Column(JSON, nullable=False, default=list)
    plan_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_ambiguity_tenant_status", "tenant_id", "status"),
    )


class MCPServerModel(Base):
    __tablename__ = "mcp_servers"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    url = Column(String, nullable=False)
    transport = Column(String, nullable=False)          # "stdio"|"sse"|"streamable_http"
    command = Column(String, nullable=True)
    args = Column(JSON, default=list)
    env = Column(JSON, default=dict)
    enabled = Column(Boolean, default=True)
    discovered_tools = Column(JSON, default=list)
    last_connected_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_mcp_tenant_name"),
        Index("ix_mcp_tenant_enabled", "tenant_id", "enabled"),
    )


class SkillModel(Base):
    __tablename__ = "skills"
    id = Column(String, primary_key=True)
    tenant_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    version = Column(Integer, default=1)
    version_history = Column(JSON, default=list)
    allowed_tools = Column(JSON, default=list)
    allowed_personas = Column(JSON, default=list)
    tags = Column(JSON, default=list)
    supporting_files = Column(JSON, default=list)
    invocation_count = Column(Integer, default=0)
    last_invoked_at = Column(DateTime(timezone=True), nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_skill_tenant_name"),
        Index("ix_skill_tenant_active", "tenant_id", "active"),
    )


class SkillInvocationModel(Base):
    __tablename__ = "skill_invocations"
    id = Column(String, primary_key=True)
    skill_id = Column(String, ForeignKey("skills.id"), nullable=False, index=True)
    tenant_id = Column(String, nullable=False, index=True)
    execution_id = Column(String, nullable=True)
    workflow_name = Column(String, nullable=True)
    persona_name = Column(String, nullable=False)
    context_summary = Column(Text, default="")
    invoked_at = Column(DateTime(timezone=True), nullable=False)
