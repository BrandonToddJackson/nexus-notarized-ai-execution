"""All ORM models. These map 1:1 to the Pydantic types but are SQLAlchemy models.

Tables: tenants, personas, seals, chains, costs, knowledge_docs
All tables have tenant_id for isolation. Indexes on common query patterns.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean, Text, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase
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
    allowed_tools = Column(JSON, default=list)
    resource_scopes = Column(JSON, default=list)
    intent_patterns = Column(JSON, default=list)
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
    intent = Column(JSON, nullable=False)
    anomaly_result = Column(JSON, nullable=False)
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
    steps = Column(JSON, nullable=False)
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
    content_hash = Column(String, nullable=False)
    access_level = Column(String, default="internal")
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_knowledge_tenant_ns", "tenant_id", "namespace"),)
