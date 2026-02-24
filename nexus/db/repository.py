"""Data access layer. Every query is tenant-scoped.

This is the ONLY layer that talks to the database.
All methods take tenant_id for isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from nexus.db.models import (
    TenantModel, PersonaModel, SealModel, ChainModel, CostModel, KnowledgeDocModel,
    TriggerModel,
)
from nexus.types import (
    AmbiguitySession, Seal, ChainPlan, CostRecord, KnowledgeDocument, TriggerConfig, TriggerType,
)


def _plan_to_json(plan) -> dict | None:
    """Serialize a WorkflowPlan to a JSON-safe dict (datetime → ISO string)."""
    if plan is None:
        return None
    import json
    return json.loads(plan.model_dump_json())


class Repository:
    """All database operations. Every query is tenant-scoped."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── Tenants ──
    async def get_tenant(self, tenant_id: str) -> Optional[TenantModel]:
        """Get tenant by ID."""
        result = await self.session.execute(
            select(TenantModel).where(TenantModel.id == tenant_id)
        )
        return result.scalar_one_or_none()

    async def create_tenant(self, name: str, api_key_hash: str) -> TenantModel:
        """Create a new tenant."""
        tenant = TenantModel(name=name, api_key_hash=api_key_hash)
        self.session.add(tenant)
        await self.session.commit()
        await self.session.refresh(tenant)
        return tenant

    async def get_tenant_by_api_key_hash(self, api_key_hash: str) -> Optional[TenantModel]:
        """Get tenant by API key hash."""
        result = await self.session.execute(
            select(TenantModel).where(TenantModel.api_key_hash == api_key_hash)
        )
        return result.scalar_one_or_none()

    # ── Personas ──
    async def list_personas(self, tenant_id: str) -> list[PersonaModel]:
        """List all personas for a tenant."""
        result = await self.session.execute(
            select(PersonaModel).where(PersonaModel.tenant_id == tenant_id)
        )
        return list(result.scalars().all())

    async def get_persona(self, tenant_id: str, persona_id: str) -> Optional[PersonaModel]:
        """Get persona by ID within tenant."""
        result = await self.session.execute(
            select(PersonaModel).where(
                PersonaModel.tenant_id == tenant_id,
                PersonaModel.id == persona_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_persona_by_name(self, tenant_id: str, name: str) -> Optional[PersonaModel]:
        """Get persona by name within tenant."""
        result = await self.session.execute(
            select(PersonaModel).where(
                PersonaModel.tenant_id == tenant_id,
                PersonaModel.name == name,
            )
        )
        return result.scalar_one_or_none()

    async def upsert_persona(self, tenant_id: str, data: dict) -> PersonaModel:
        """Create or update a persona."""
        existing = await self.get_persona_by_name(tenant_id, data["name"])
        if existing:
            for key, value in data.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            await self.session.commit()
            await self.session.refresh(existing)
            return existing
        persona = PersonaModel(tenant_id=tenant_id, **data)
        self.session.add(persona)
        await self.session.commit()
        await self.session.refresh(persona)
        return persona

    # ── Seals ──
    async def create_seal(self, seal: Seal) -> SealModel:
        """Persist a seal."""
        record = SealModel(
            id=seal.id,
            chain_id=seal.chain_id,
            step_index=seal.step_index,
            tenant_id=seal.tenant_id,
            persona_id=seal.persona_id,
            intent=seal.intent.model_dump(mode="json"),
            anomaly_result=seal.anomaly_result.model_dump(mode="json"),
            tool_name=seal.tool_name,
            tool_params=seal.tool_params,
            tool_result=seal.tool_result,
            status=seal.status.value,
            cot_trace=seal.cot_trace,
            fingerprint=seal.fingerprint,
            parent_fingerprint=seal.parent_fingerprint,
            created_at=seal.created_at,
            completed_at=seal.completed_at,
            error=seal.error,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def update_seal(self, seal_id: str, updates: dict) -> Optional[SealModel]:
        """Update a seal (finalization)."""
        result = await self.session.execute(
            select(SealModel).where(SealModel.id == seal_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            return None
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def list_seals(self, tenant_id: str, limit: int = 100, offset: int = 0) -> list[SealModel]:
        """Paginated seal history for a tenant."""
        result = await self.session.execute(
            select(SealModel)
            .where(SealModel.tenant_id == tenant_id)
            .order_by(SealModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_chain_seals(self, chain_id: str) -> list[SealModel]:
        """Get all seals for a chain, ordered by step_index."""
        result = await self.session.execute(
            select(SealModel)
            .where(SealModel.chain_id == chain_id)
            .order_by(SealModel.step_index)
        )
        return list(result.scalars().all())

    # ── Chains ──
    async def create_chain(self, chain: ChainPlan) -> ChainModel:
        """Persist a chain plan."""
        record = ChainModel(
            id=chain.id,
            tenant_id=chain.tenant_id,
            task=chain.task,
            steps=chain.steps,
            status=chain.status.value,
            seal_ids=chain.seals,
            created_at=chain.created_at,
            completed_at=chain.completed_at,
            error=chain.error,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def update_chain(self, chain_id: str, updates: dict) -> Optional[ChainModel]:
        """Update chain status."""
        result = await self.session.execute(
            select(ChainModel).where(ChainModel.id == chain_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            return None
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def list_chains(self, tenant_id: str, limit: int = 50) -> list[ChainModel]:
        """List chains for a tenant."""
        result = await self.session.execute(
            select(ChainModel)
            .where(ChainModel.tenant_id == tenant_id)
            .order_by(ChainModel.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    # ── Costs ──
    async def add_cost(self, cost: CostRecord) -> CostModel:
        """Record a cost entry."""
        record = CostModel(
            tenant_id=cost.tenant_id,
            chain_id=cost.chain_id,
            seal_id=cost.seal_id,
            model=cost.model,
            input_tokens=cost.input_tokens,
            output_tokens=cost.output_tokens,
            cost_usd=cost.cost_usd,
            timestamp=cost.timestamp,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def get_tenant_cost(self, tenant_id: str, month: str = None) -> float:
        """Get total cost for a tenant, optionally for a specific month.

        Args:
            tenant_id: Tenant identifier.
            month: Optional month filter in "YYYY-MM" format.
        """
        query = select(func.sum(CostModel.cost_usd)).where(
            CostModel.tenant_id == tenant_id
        )
        if month:
            # month format: "YYYY-MM"
            year, mon = month.split("-")
            query = query.where(
                func.extract("year", CostModel.timestamp) == int(year),
                func.extract("month", CostModel.timestamp) == int(mon),
            )
        result = await self.session.execute(query)
        total = result.scalar_one_or_none()
        return total if total is not None else 0.0

    # ── Knowledge ──
    async def add_knowledge_doc(self, doc: KnowledgeDocument) -> KnowledgeDocModel:
        """Record a knowledge document."""
        import hashlib
        content_hash = hashlib.sha256(doc.content.encode()).hexdigest()
        record = KnowledgeDocModel(
            id=doc.id,
            tenant_id=doc.tenant_id,
            namespace=doc.namespace,
            source=doc.source,
            content_hash=content_hash,
            access_level=doc.access_level,
            metadata_=doc.metadata,
            created_at=doc.created_at,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def list_knowledge_docs(self, tenant_id: str, namespace: str = None) -> list[KnowledgeDocModel]:
        """List knowledge documents for a tenant."""
        query = select(KnowledgeDocModel).where(KnowledgeDocModel.tenant_id == tenant_id)
        if namespace:
            query = query.where(KnowledgeDocModel.namespace == namespace)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    # ── Triggers ──
    @staticmethod
    def _model_to_trigger(m: TriggerModel) -> TriggerConfig:
        """Convert a TriggerModel ORM row to a TriggerConfig Pydantic model."""
        return TriggerConfig(
            id=m.id,
            workflow_id=m.workflow_id,
            tenant_id=m.tenant_id,
            trigger_type=TriggerType(m.trigger_type),
            enabled=m.enabled,
            config=m.config or {},
            webhook_path=m.webhook_path,
            last_triggered_at=m.last_triggered_at,
            created_at=m.created_at,
        )

    async def save_trigger(self, trigger: TriggerConfig) -> TriggerConfig:
        """Persist a new trigger."""
        record = TriggerModel(
            id=trigger.id,
            workflow_id=trigger.workflow_id,
            tenant_id=trigger.tenant_id,
            trigger_type=trigger.trigger_type.value,
            enabled=trigger.enabled,
            config=trigger.config,
            webhook_path=trigger.webhook_path,
            last_triggered_at=trigger.last_triggered_at,
            created_at=trigger.created_at,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return self._model_to_trigger(record)

    async def get_trigger(self, tenant_id: str, trigger_id: str) -> Optional[TriggerConfig]:
        """Get trigger by ID within tenant."""
        result = await self.session.execute(
            select(TriggerModel).where(
                TriggerModel.id == trigger_id,
                TriggerModel.tenant_id == tenant_id,
            )
        )
        m = result.scalar_one_or_none()
        return self._model_to_trigger(m) if m is not None else None

    async def get_trigger_by_webhook_path(self, webhook_path: str) -> Optional[TriggerConfig]:
        """Look up a trigger by its unique webhook path."""
        result = await self.session.execute(
            select(TriggerModel).where(TriggerModel.webhook_path == webhook_path)
        )
        m = result.scalar_one_or_none()
        return self._model_to_trigger(m) if m is not None else None

    async def list_triggers(
        self,
        tenant_id: Optional[str],
        workflow_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> list[TriggerConfig]:
        """List triggers with optional filters.

        ``tenant_id=None`` returns triggers across all tenants (used by CronScheduler startup).
        """
        conditions = []
        if tenant_id is not None:
            conditions.append(TriggerModel.tenant_id == tenant_id)
        if workflow_id is not None:
            conditions.append(TriggerModel.workflow_id == workflow_id)
        if enabled is not None:
            conditions.append(TriggerModel.enabled == enabled)

        query = select(TriggerModel)
        if conditions:
            query = query.where(*conditions)
        result = await self.session.execute(query)
        return [self._model_to_trigger(m) for m in result.scalars().all()]

    async def update_trigger(self, trigger: TriggerConfig) -> TriggerConfig:
        """Persist changes to an existing trigger (full replace)."""
        result = await self.session.execute(
            select(TriggerModel).where(TriggerModel.id == trigger.id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            # Fallback: insert if somehow missing
            return await self.save_trigger(trigger)
        record.workflow_id       = trigger.workflow_id
        record.tenant_id         = trigger.tenant_id
        record.trigger_type      = trigger.trigger_type.value
        record.enabled           = trigger.enabled
        record.config            = trigger.config
        record.webhook_path      = trigger.webhook_path
        record.last_triggered_at = trigger.last_triggered_at
        await self.session.commit()
        await self.session.refresh(record)
        return self._model_to_trigger(record)

    async def delete_trigger(self, tenant_id: str, trigger_id: str) -> bool:
        """Delete a trigger.  Returns True if deleted, False if not found."""
        result = await self.session.execute(
            select(TriggerModel).where(
                TriggerModel.id == trigger_id,
                TriggerModel.tenant_id == tenant_id,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            return False
        await self.session.delete(record)
        await self.session.commit()
        return True

    # ── Ambiguity Sessions (Phase 23.1) ──

    async def create_ambiguity_session(self, session_obj) -> AmbiguitySession:
        """Persist a new AmbiguitySession."""
        from nexus.db.models import AmbiguitySessionModel
        model = AmbiguitySessionModel(
            id=session_obj.id,
            tenant_id=session_obj.tenant_id,
            original_description=session_obj.original_description,
            status=session_obj.status.value,
            questions_json=[q.model_dump() for q in session_obj.questions],
            answers_json=[a.model_dump() for a in session_obj.answers],
            current_round=session_obj.current_round,
            max_rounds=session_obj.max_rounds,
            specificity_history_json=session_obj.specificity_history,
            plan_json=_plan_to_json(session_obj.plan),
            created_at=session_obj.created_at,
            updated_at=session_obj.updated_at,
            expires_at=session_obj.expires_at,
        )
        self.session.add(model)
        await self.session.commit()
        return session_obj

    async def get_ambiguity_session(self, session_id: str) -> Optional[AmbiguitySession]:
        """Load an AmbiguitySession by ID. Returns None if not found."""
        from nexus.db.models import AmbiguitySessionModel
        from nexus.types import (
            AmbiguitySession, AmbiguitySessionStatus, ClarifyingAnswer,
            ClarifyingQuestion, WorkflowPlan,
        )
        result = await self.session.execute(
            select(AmbiguitySessionModel).where(AmbiguitySessionModel.id == session_id)
        )
        model = result.scalar_one_or_none()
        if not model:
            return None
        return AmbiguitySession(
            id=model.id,
            tenant_id=model.tenant_id,
            original_description=model.original_description,
            status=AmbiguitySessionStatus(model.status),
            questions=[ClarifyingQuestion(**q) for q in (model.questions_json or [])],
            answers=[ClarifyingAnswer(**a) for a in (model.answers_json or [])],
            current_round=model.current_round,
            max_rounds=model.max_rounds,
            specificity_history=model.specificity_history_json or [],
            plan=WorkflowPlan(**model.plan_json) if model.plan_json else None,
            created_at=model.created_at,
            updated_at=model.updated_at,
            expires_at=model.expires_at,
        )

    async def update_ambiguity_session(
        self, session_id: str, updates: dict
    ) -> Optional[AmbiguitySession]:
        """Apply partial updates to an existing AmbiguitySession."""
        from nexus.db.models import AmbiguitySessionModel
        result = await self.session.execute(
            select(AmbiguitySessionModel).where(AmbiguitySessionModel.id == session_id)
        )
        model = result.scalar_one_or_none()
        if not model:
            raise ValueError(f"AmbiguitySession {session_id!r} not found for update.")

        column_map = {
            "status": "status",
            "questions": "questions_json",
            "answers": "answers_json",
            "current_round": "current_round",
            "specificity_history": "specificity_history_json",
            "plan": "plan_json",
            "updated_at": "updated_at",
        }
        for key, value in updates.items():
            col = column_map.get(key, key)
            if col == "status" and hasattr(value, "value"):
                value = value.value
            elif col == "plan_json" and value is not None and not isinstance(value, dict):
                # value is a WorkflowPlan Pydantic model — serialize datetimes
                import json as _json
                value = _json.loads(value.model_dump_json())
            setattr(model, col, value)

        await self.session.commit()
        return await self.get_ambiguity_session(session_id)

    async def list_ambiguity_sessions(
        self,
        tenant_id: str,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[AmbiguitySession]:
        """List sessions for a tenant, optionally filtered by status. Newest first."""
        from nexus.db.models import AmbiguitySessionModel
        from sqlalchemy import desc
        query = select(AmbiguitySessionModel).where(
            AmbiguitySessionModel.tenant_id == tenant_id
        )
        if status:
            query = query.where(AmbiguitySessionModel.status == status)
        query = (
            query.order_by(desc(AmbiguitySessionModel.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(query)
        models = result.scalars().all()
        sessions = []
        for model in models:
            s = await self.get_ambiguity_session(model.id)
            if s:
                sessions.append(s)
        return sessions

    async def expire_abandoned_sessions(self, cutoff: datetime) -> int:
        """Mark all active sessions with expires_at < cutoff as abandoned. Returns count updated."""
        from nexus.db.models import AmbiguitySessionModel
        from sqlalchemy import update as sql_update
        result = await self.session.execute(
            sql_update(AmbiguitySessionModel)
            .where(
                AmbiguitySessionModel.status == "active",
                AmbiguitySessionModel.expires_at < cutoff,
            )
            .values(status="abandoned", updated_at=datetime.now(timezone.utc))
        )
        await self.session.commit()
        return result.rowcount
