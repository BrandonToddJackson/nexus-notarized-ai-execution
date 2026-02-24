"""Data access layer. Every query is tenant-scoped.

This is the ONLY layer that talks to the database.
All methods take tenant_id for isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
import uuid

from nexus.db.models import (
    TenantModel, PersonaModel, SealModel, ChainModel, CostModel, KnowledgeDocModel,
    TriggerModel, SkillModel, SkillInvocationModel, WorkflowExecutionModel,
    WorkflowModel, CredentialModel, MCPServerModel,
)
from nexus.types import (
    AmbiguitySession, Seal, ChainPlan, CostRecord, KnowledgeDocument, TriggerConfig, TriggerType,
    SkillRecord, SkillInvocation, SkillVersion, SkillFile, WorkflowExecution,
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

    # ── Skills ──

    @staticmethod
    def _model_to_skill(m: SkillModel) -> SkillRecord:
        return SkillRecord(
            id=m.id,
            tenant_id=m.tenant_id,
            name=m.name,
            display_name=m.display_name,
            description=m.description,
            content=m.content,
            version=m.version,
            version_history=[SkillVersion(**v) for v in (m.version_history or [])],
            allowed_tools=m.allowed_tools or [],
            allowed_personas=m.allowed_personas or [],
            tags=m.tags or [],
            supporting_files=[SkillFile(**f) for f in (m.supporting_files or [])],
            invocation_count=m.invocation_count or 0,
            last_invoked_at=m.last_invoked_at,
            active=m.active,
            created_at=m.created_at,
            updated_at=m.updated_at,
        )

    async def create_skill(self, skill: SkillRecord) -> SkillRecord:
        """Persist a new skill."""
        record = SkillModel(
            id=skill.id,
            tenant_id=skill.tenant_id,
            name=skill.name,
            display_name=skill.display_name,
            description=skill.description,
            content=skill.content,
            version=skill.version,
            version_history=[v.model_dump(mode="json") for v in skill.version_history],
            allowed_tools=skill.allowed_tools,
            allowed_personas=skill.allowed_personas,
            tags=skill.tags,
            supporting_files=[f.model_dump() for f in skill.supporting_files],
            invocation_count=skill.invocation_count,
            last_invoked_at=skill.last_invoked_at,
            active=skill.active,
            created_at=skill.created_at,
            updated_at=skill.updated_at,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return self._model_to_skill(record)

    async def get_skill(self, skill_id: str, tenant_id: str) -> Optional[SkillRecord]:
        """Get skill by ID within tenant."""
        result = await self.session.execute(
            select(SkillModel).where(
                SkillModel.id == skill_id,
                SkillModel.tenant_id == tenant_id,
            )
        )
        m = result.scalar_one_or_none()
        return self._model_to_skill(m) if m else None

    async def get_skill_by_name(self, name: str, tenant_id: str) -> Optional[SkillRecord]:
        """Get skill by name within tenant."""
        result = await self.session.execute(
            select(SkillModel).where(
                SkillModel.name == name,
                SkillModel.tenant_id == tenant_id,
            )
        )
        m = result.scalar_one_or_none()
        return self._model_to_skill(m) if m else None

    async def update_skill(self, skill_id: str, updates: dict) -> Optional[SkillRecord]:
        """Update a skill."""
        result = await self.session.execute(
            select(SkillModel).where(SkillModel.id == skill_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            return None
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)
        await self.session.commit()
        await self.session.refresh(record)
        return self._model_to_skill(record)

    async def list_skills(
        self, tenant_id: str, active_only: bool = False, limit: int = 50, offset: int = 0
    ) -> list[SkillRecord]:
        """List skills for a tenant."""
        query = select(SkillModel).where(SkillModel.tenant_id == tenant_id)
        if active_only:
            query = query.where(SkillModel.active == True)  # noqa: E712
        query = query.order_by(SkillModel.created_at.desc()).limit(limit).offset(offset)
        result = await self.session.execute(query)
        return [self._model_to_skill(m) for m in result.scalars().all()]

    async def delete_skill(self, skill_id: str, tenant_id: str) -> bool:
        """Soft delete a skill (set active=False)."""
        result = await self.session.execute(
            select(SkillModel).where(
                SkillModel.id == skill_id,
                SkillModel.tenant_id == tenant_id,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            return False
        record.active = False
        record.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        return True

    async def record_skill_invocation(self, inv: SkillInvocation) -> SkillInvocation:
        """Record a skill invocation."""
        record = SkillInvocationModel(
            id=inv.id,
            skill_id=inv.skill_id,
            tenant_id=inv.tenant_id,
            execution_id=inv.execution_id,
            workflow_name=inv.workflow_name,
            persona_name=inv.persona_name,
            context_summary=inv.context_summary,
            invoked_at=inv.invoked_at,
        )
        self.session.add(record)
        await self.session.commit()
        return inv

    async def list_skill_invocations(
        self, skill_id: str, tenant_id: str, limit: int = 50, offset: int = 0
    ) -> list[SkillInvocation]:
        """List invocations for a skill."""
        result = await self.session.execute(
            select(SkillInvocationModel)
            .where(
                SkillInvocationModel.skill_id == skill_id,
                SkillInvocationModel.tenant_id == tenant_id,
            )
            .order_by(SkillInvocationModel.invoked_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [
            SkillInvocation(
                id=m.id,
                skill_id=m.skill_id,
                tenant_id=m.tenant_id,
                execution_id=m.execution_id,
                workflow_name=m.workflow_name,
                persona_name=m.persona_name,
                context_summary=m.context_summary or "",
                invoked_at=m.invoked_at,
            )
            for m in result.scalars().all()
        ]

    # ── WorkflowExecutions ──

    async def save_execution(self, execution: WorkflowExecution) -> WorkflowExecutionModel:
        """Persist a WorkflowExecution record."""
        import json
        model = WorkflowExecutionModel(
            id=execution.id,
            workflow_id=execution.workflow_id,
            workflow_version=execution.workflow_version,
            tenant_id=execution.tenant_id,
            trigger_type=execution.trigger_type.value if hasattr(execution.trigger_type, "value") else str(execution.trigger_type),
            trigger_data=json.loads(json.dumps(execution.trigger_data, default=str)),
            chain_id=execution.chain_id or "",
            status=execution.status.value if hasattr(execution.status, "value") else str(execution.status),
            started_at=execution.started_at,
            completed_at=getattr(execution, "completed_at", None),
            error=getattr(execution, "error", None),
            step_results=json.loads(json.dumps(getattr(execution, "step_results", {}), default=str)),
        )
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def get_execution(self, tenant_id: str, execution_id: str) -> Optional[WorkflowExecutionModel]:
        """Fetch a WorkflowExecution by ID within a tenant."""
        result = await self.session.execute(
            select(WorkflowExecutionModel).where(
                WorkflowExecutionModel.tenant_id == tenant_id,
                WorkflowExecutionModel.id == execution_id,
            )
        )
        return result.scalar_one_or_none()

    async def update_execution(self, execution_id: str, updates: dict) -> Optional[WorkflowExecutionModel]:
        """Apply partial updates to a WorkflowExecution row."""
        result = await self.session.execute(
            select(WorkflowExecutionModel).where(WorkflowExecutionModel.id == execution_id)
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        for key, value in updates.items():
            if hasattr(model, key):
                setattr(model, key, value)
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def list_executions(
        self,
        tenant_id: str,
        workflow_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[WorkflowExecutionModel]:
        """List executions for a tenant, newest first."""
        from sqlalchemy import desc
        query = select(WorkflowExecutionModel).where(
            WorkflowExecutionModel.tenant_id == tenant_id
        )
        if workflow_id is not None:
            query = query.where(WorkflowExecutionModel.workflow_id == workflow_id)
        query = query.order_by(desc(WorkflowExecutionModel.started_at)).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    # ── Workflows ──

    async def save_workflow(
        self,
        tenant_id: str,
        name: str,
        description: str = "",
        steps: list = None,
        edges: list = None,
        trigger_config: dict = None,
        settings: dict = None,
        tags: list = None,
        created_by: str = "",
        status: str = "draft",
        version: int = 1,
    ) -> WorkflowModel:
        """Create a new workflow."""
        now = datetime.now(timezone.utc)
        record = WorkflowModel(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            name=name,
            description=description,
            version=version,
            status=status,
            trigger_config=trigger_config if trigger_config is not None else {},
            steps=steps if steps is not None else [],
            edges=edges if edges is not None else [],
            created_at=now,
            updated_at=now,
            created_by=created_by,
            tags=tags if tags is not None else [],
            settings=settings if settings is not None else {},
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def get_workflow(
        self,
        tenant_id: str,
        workflow_id: str,
        version: Optional[int] = None,
    ) -> Optional[WorkflowModel]:
        """Get workflow by ID. If version=None, returns the latest version."""
        query = select(WorkflowModel).where(
            WorkflowModel.tenant_id == tenant_id,
            WorkflowModel.id == workflow_id,
        )
        if version is not None:
            query = query.where(WorkflowModel.version == version)
        else:
            query = query.order_by(WorkflowModel.version.desc()).limit(1)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def list_workflows(
        self,
        tenant_id: str,
        status: Optional[str] = None,
        tags: Optional[list] = None,
        limit: int = 50,
        offset: int = 0,
        latest_only: bool = True,
    ) -> list[WorkflowModel]:
        """List workflows for a tenant with optional filters."""
        from sqlalchemy import desc
        query = select(WorkflowModel).where(WorkflowModel.tenant_id == tenant_id)
        if status is not None:
            query = query.where(WorkflowModel.status == status)
        query = query.order_by(desc(WorkflowModel.updated_at))
        result = await self.session.execute(query)
        rows = list(result.scalars().all())

        # Tags filter (Python-side on JSON column)
        if tags:
            rows = [r for r in rows if any(t in (r.tags or []) for t in tags)]

        # Dedup to latest version per name
        if latest_only:
            seen: dict[str, WorkflowModel] = {}
            for row in rows:
                existing = seen.get(row.name)
                if existing is None or row.version > existing.version:
                    seen[row.name] = row
            rows = sorted(seen.values(), key=lambda r: r.updated_at or datetime.min, reverse=True)

        return rows[offset: offset + limit]

    async def update_workflow(
        self, tenant_id: str, workflow_id: str, updates: dict
    ) -> Optional[WorkflowModel]:
        """Apply partial updates to a workflow."""
        allowed = {
            "name", "description", "status", "trigger_config", "steps",
            "edges", "settings", "tags", "created_by", "version",
        }
        bad = set(updates) - allowed
        if bad:
            raise ValueError(f"Unknown workflow update keys: {bad}")
        result = await self.session.execute(
            select(WorkflowModel).where(
                WorkflowModel.tenant_id == tenant_id,
                WorkflowModel.id == workflow_id,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            return None
        for key, value in updates.items():
            setattr(record, key, value)
        record.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def delete_workflow(self, tenant_id: str, workflow_id: str) -> bool:
        """Hard-delete a workflow and cascade to triggers and executions."""
        result = await self.session.execute(
            select(WorkflowModel).where(
                WorkflowModel.tenant_id == tenant_id,
                WorkflowModel.id == workflow_id,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            return False
        # Cascade: delete triggers and executions
        await self.session.execute(
            delete(TriggerModel).where(TriggerModel.workflow_id == workflow_id)
        )
        await self.session.execute(
            delete(WorkflowExecutionModel).where(
                WorkflowExecutionModel.workflow_id == workflow_id
            )
        )
        await self.session.delete(record)
        await self.session.commit()
        return True

    # ── Credentials ──

    async def save_credential(
        self,
        tenant_id: str,
        name: str,
        credential_type: str,
        service_name: str,
        encrypted_data: str,
        scoped_personas: Optional[list] = None,
        expires_at: Optional[datetime] = None,
    ) -> CredentialModel:
        """Create a new credential."""
        now = datetime.now(timezone.utc)
        record = CredentialModel(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            name=name,
            credential_type=credential_type,
            service_name=service_name,
            encrypted_data=encrypted_data,
            scoped_personas=scoped_personas if scoped_personas is not None else [],
            is_active=True,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def get_credential(
        self,
        tenant_id: str,
        credential_id: str,
        include_inactive: bool = False,
    ) -> Optional[CredentialModel]:
        """Get credential by ID within tenant."""
        conditions = [
            CredentialModel.tenant_id == tenant_id,
            CredentialModel.id == credential_id,
        ]
        if not include_inactive:
            conditions.append(CredentialModel.is_active == True)  # noqa: E712
        result = await self.session.execute(
            select(CredentialModel).where(*conditions)
        )
        return result.scalar_one_or_none()

    async def list_credentials(
        self,
        tenant_id: str,
        service_name: Optional[str] = None,
        credential_type: Optional[str] = None,
        persona_name: Optional[str] = None,
        include_inactive: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CredentialModel]:
        """List credentials for a tenant with optional filters."""
        conditions = [CredentialModel.tenant_id == tenant_id]
        if not include_inactive:
            conditions.append(CredentialModel.is_active == True)  # noqa: E712
        if service_name is not None:
            conditions.append(CredentialModel.service_name == service_name)
        if credential_type is not None:
            conditions.append(CredentialModel.credential_type == credential_type)
        result = await self.session.execute(
            select(CredentialModel).where(*conditions)
        )
        rows = list(result.scalars().all())
        # persona_name filter: Python-side on JSON column
        if persona_name is not None:
            rows = [r for r in rows if persona_name in (r.scoped_personas or [])]
        return rows[offset: offset + limit]

    async def update_credential(
        self, tenant_id: str, credential_id: str, updates: dict
    ) -> Optional[CredentialModel]:
        """Apply partial updates to a credential."""
        allowed = {
            "name", "credential_type", "service_name", "encrypted_data",
            "scoped_personas", "expires_at",
        }
        bad = set(updates) - allowed
        if bad:
            raise ValueError(f"Unknown credential update keys: {bad}")
        result = await self.session.execute(
            select(CredentialModel).where(
                CredentialModel.tenant_id == tenant_id,
                CredentialModel.id == credential_id,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            return None
        for key, value in updates.items():
            setattr(record, key, value)
        record.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def delete_credential(self, tenant_id: str, credential_id: str) -> bool:
        """Soft-delete a credential (sets is_active=False, preserves audit trail)."""
        result = await self.session.execute(
            select(CredentialModel).where(
                CredentialModel.tenant_id == tenant_id,
                CredentialModel.id == credential_id,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            return False
        record.is_active = False
        record.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        return True

    # ── MCP Servers ──

    @staticmethod
    def _validate_mcp_transport(transport: str, url: Optional[str], command: Optional[str]) -> None:
        """Raise ValueError if transport/url/command combination is invalid."""
        if transport == "stdio":
            if not command:
                raise ValueError("stdio transport requires 'command'")
        elif transport in ("sse", "streamable_http"):
            if not url:
                raise ValueError(f"{transport} transport requires 'url'")
        else:
            raise ValueError(f"Unknown transport: {transport!r}. Must be stdio, sse, or streamable_http")

    async def save_mcp_server(
        self,
        tenant_id: str,
        name: str,
        transport: str,
        url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[list] = None,
        env: Optional[dict] = None,
        enabled: bool = True,
    ) -> MCPServerModel:
        """Create a new MCP server configuration."""
        self._validate_mcp_transport(transport, url, command)
        record = MCPServerModel(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            name=name,
            transport=transport,
            url=url,
            command=command,
            args=args if args is not None else [],
            env=env if env is not None else {},
            enabled=enabled,
            discovered_tools=[],
            last_connected_at=None,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def get_mcp_server(
        self, tenant_id: str, server_id: str
    ) -> Optional[MCPServerModel]:
        """Get MCP server by ID within tenant."""
        result = await self.session.execute(
            select(MCPServerModel).where(
                MCPServerModel.tenant_id == tenant_id,
                MCPServerModel.id == server_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_mcp_servers(
        self,
        tenant_id: str,
        enabled: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MCPServerModel]:
        """List MCP servers for a tenant, ordered by name."""
        conditions = [MCPServerModel.tenant_id == tenant_id]
        if enabled is not None:
            conditions.append(MCPServerModel.enabled == enabled)
        result = await self.session.execute(
            select(MCPServerModel)
            .where(*conditions)
            .order_by(MCPServerModel.name.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def update_mcp_server(
        self, tenant_id: str, server_id: str, updates: dict
    ) -> Optional[MCPServerModel]:
        """Apply partial updates to an MCP server configuration."""
        allowed = {
            "name", "url", "transport", "command", "args", "env",
            "enabled", "discovered_tools", "last_connected_at",
        }
        bad = set(updates) - allowed
        if bad:
            raise ValueError(f"Unknown MCP server update keys: {bad}")
        result = await self.session.execute(
            select(MCPServerModel).where(
                MCPServerModel.tenant_id == tenant_id,
                MCPServerModel.id == server_id,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            return None
        for key, value in updates.items():
            setattr(record, key, value)
        # Re-validate transport consistency if transport-related fields changed
        transport_keys = {"transport", "url", "command"}
        if transport_keys & set(updates):
            transport = updates.get("transport", record.transport)
            url = updates.get("url", record.url)
            command = updates.get("command", record.command)
            self._validate_mcp_transport(transport, url, command)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def delete_mcp_server(self, tenant_id: str, server_id: str) -> bool:
        """Hard-delete an MCP server configuration."""
        result = await self.session.execute(
            select(MCPServerModel).where(
                MCPServerModel.tenant_id == tenant_id,
                MCPServerModel.id == server_id,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            return False
        await self.session.delete(record)
        await self.session.commit()
        return True
