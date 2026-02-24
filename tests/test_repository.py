"""Smoketest for Phase 5: Persistence — Repository and seed_database.

Uses an in-memory SQLite database (aiosqlite) so no Postgres required.
"""

import hashlib
import pytest
from datetime import datetime

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from nexus.db.models import Base
from nexus.db.repository import Repository
from nexus.db.seed import seed_database, DEFAULT_PERSONAS
from nexus.types import (
    Seal, ChainPlan, CostRecord, KnowledgeDocument,
    IntentDeclaration, AnomalyResult, GateResult,
    ActionStatus, ChainStatus, GateVerdict, RiskLevel,
    TriggerConfig, TriggerType,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
async def session():
    """In-memory SQLite async session with schema created."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


@pytest.fixture
def tenant_id():
    return "test-tenant-001"


def _make_gate(name: str) -> GateResult:
    return GateResult(
        gate_name=name,
        verdict=GateVerdict.PASS,
        score=0.9,
        threshold=0.75,
        details="ok",
    )


def _make_anomaly(persona_id: str) -> AnomalyResult:
    return AnomalyResult(
        gates=[_make_gate(n) for n in ("scope", "intent", "ttl", "drift")],
        overall_verdict=GateVerdict.PASS,
        risk_level=RiskLevel.LOW,
        persona_id=persona_id,
        action_fingerprint="abc123",
    )


def _make_intent() -> IntentDeclaration:
    return IntentDeclaration(
        task_description="test task",
        planned_action="search for info",
        tool_name="knowledge_search",
        tool_params={"query": "nexus"},
        resource_targets=["kb:*"],
        reasoning="need to find info",
        confidence=0.9,
    )


def _make_seal(chain_id: str, tenant_id: str, step: int = 0) -> Seal:
    return Seal(
        chain_id=chain_id,
        step_index=step,
        tenant_id=tenant_id,
        persona_id="persona-001",
        intent=_make_intent(),
        anomaly_result=_make_anomaly("persona-001"),
        tool_name="knowledge_search",
        tool_params={"query": "nexus"},
        status=ActionStatus.EXECUTED,
        fingerprint="fp-001",
        parent_fingerprint="",
    )


def _make_chain(tenant_id: str) -> ChainPlan:
    return ChainPlan(
        tenant_id=tenant_id,
        task="What is NEXUS?",
        steps=[{"tool": "knowledge_search", "params": {"query": "NEXUS"}}],
        status=ChainStatus.EXECUTING,
    )


# ── Tenant tests ─────────────────────────────────────────────────────────────

class TestTenant:
    async def test_create_and_get(self, session):
        repo = Repository(session)
        tenant = await repo.create_tenant("Acme Corp", "hash-xyz")
        assert tenant.id is not None
        assert tenant.name == "Acme Corp"

        fetched = await repo.get_tenant(tenant.id)
        assert fetched is not None
        assert fetched.name == "Acme Corp"
        assert fetched.api_key_hash == "hash-xyz"

    async def test_get_missing_returns_none(self, session):
        repo = Repository(session)
        result = await repo.get_tenant("does-not-exist")
        assert result is None


# ── Persona tests ─────────────────────────────────────────────────────────────

class TestPersona:
    async def test_upsert_creates_new(self, session, tenant_id):
        repo = Repository(session)
        await repo.create_tenant("T", "h")  # need a tenant for FK; use explicit id
        # Use demo tenant to avoid FK issues
        from nexus.db.models import TenantModel
        session.add(TenantModel(id=tenant_id, name="T", api_key_hash="h"))
        await session.commit()

        data = {
            "name": "researcher",
            "description": "Searches stuff",
            "allowed_tools": ["knowledge_search"],
            "resource_scopes": ["kb:*"],
            "intent_patterns": ["find info"],
            "max_ttl_seconds": 60,
            "risk_tolerance": "low",
        }
        persona = await repo.upsert_persona(tenant_id, data)
        assert persona.id is not None
        assert persona.name == "researcher"
        assert persona.tenant_id == tenant_id

    async def test_upsert_updates_existing(self, session, tenant_id):
        repo = Repository(session)
        from nexus.db.models import TenantModel
        session.add(TenantModel(id=tenant_id, name="T", api_key_hash="h"))
        await session.commit()

        data = {"name": "researcher", "description": "v1", "allowed_tools": [],
                "resource_scopes": [], "intent_patterns": [], "max_ttl_seconds": 60, "risk_tolerance": "low"}
        p1 = await repo.upsert_persona(tenant_id, data)

        data["description"] = "v2"
        p2 = await repo.upsert_persona(tenant_id, data)
        assert p1.id == p2.id
        assert p2.description == "v2"

    async def test_list_personas(self, session, tenant_id):
        repo = Repository(session)
        from nexus.db.models import TenantModel
        session.add(TenantModel(id=tenant_id, name="T", api_key_hash="h"))
        await session.commit()

        for name in ("researcher", "analyst"):
            await repo.upsert_persona(tenant_id, {
                "name": name, "description": "", "allowed_tools": [],
                "resource_scopes": [], "intent_patterns": [], "max_ttl_seconds": 60, "risk_tolerance": "low",
            })
        personas = await repo.list_personas(tenant_id)
        assert len(personas) == 2

    async def test_get_persona_by_name_missing(self, session, tenant_id):
        repo = Repository(session)
        result = await repo.get_persona_by_name(tenant_id, "ghost")
        assert result is None

    async def test_tenant_isolation(self, session):
        """Personas from tenant A must not appear for tenant B."""
        repo = Repository(session)
        from nexus.db.models import TenantModel
        for tid in ("tenant-a", "tenant-b"):
            session.add(TenantModel(id=tid, name=tid, api_key_hash="h"))
        await session.commit()

        await repo.upsert_persona("tenant-a", {
            "name": "researcher", "description": "", "allowed_tools": [],
            "resource_scopes": [], "intent_patterns": [], "max_ttl_seconds": 60, "risk_tolerance": "low",
        })
        result = await repo.list_personas("tenant-b")
        assert result == []


# ── Seal tests ───────────────────────────────────────────────────────────────

class TestSeal:
    async def _seed_tenant_chain(self, session, tenant_id):
        from nexus.db.models import TenantModel, ChainModel
        session.add(TenantModel(id=tenant_id, name="T", api_key_hash="h"))
        chain = ChainModel(id="chain-001", tenant_id=tenant_id, task="t", steps=[], status="executing")
        session.add(chain)
        await session.commit()

    async def test_create_seal(self, session, tenant_id):
        await self._seed_tenant_chain(session, tenant_id)
        repo = Repository(session)
        seal = _make_seal("chain-001", tenant_id)
        record = await repo.create_seal(seal)
        assert record.id == seal.id
        assert record.tool_name == "knowledge_search"
        assert record.status == "executed"

    async def test_update_seal(self, session, tenant_id):
        await self._seed_tenant_chain(session, tenant_id)
        repo = Repository(session)
        seal = _make_seal("chain-001", tenant_id)
        record = await repo.create_seal(seal)

        updated = await repo.update_seal(record.id, {"status": "completed", "error": None})
        assert updated.status == "completed"

    async def test_update_seal_missing_returns_none(self, session):
        repo = Repository(session)
        result = await repo.update_seal("no-such-id", {"status": "done"})
        assert result is None

    async def test_list_seals_paginated(self, session, tenant_id):
        await self._seed_tenant_chain(session, tenant_id)
        repo = Repository(session)
        for i in range(5):
            await repo.create_seal(_make_seal("chain-001", tenant_id, step=i))

        page1 = await repo.list_seals(tenant_id, limit=3, offset=0)
        page2 = await repo.list_seals(tenant_id, limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2

    async def test_get_chain_seals_ordered(self, session, tenant_id):
        await self._seed_tenant_chain(session, tenant_id)
        repo = Repository(session)
        for i in [2, 0, 1]:
            await repo.create_seal(_make_seal("chain-001", tenant_id, step=i))

        seals = await repo.get_chain_seals("chain-001")
        assert [s.step_index for s in seals] == [0, 1, 2]


# ── Chain tests ───────────────────────────────────────────────────────────────

class TestChain:
    async def _seed_tenant(self, session, tenant_id):
        from nexus.db.models import TenantModel
        session.add(TenantModel(id=tenant_id, name="T", api_key_hash="h"))
        await session.commit()

    async def test_create_and_list(self, session, tenant_id):
        await self._seed_tenant(session, tenant_id)
        repo = Repository(session)
        chain = _make_chain(tenant_id)
        record = await repo.create_chain(chain)
        assert record.id == chain.id
        assert record.task == "What is NEXUS?"

        chains = await repo.list_chains(tenant_id)
        assert len(chains) == 1

    async def test_update_chain(self, session, tenant_id):
        await self._seed_tenant(session, tenant_id)
        repo = Repository(session)
        chain = _make_chain(tenant_id)
        record = await repo.create_chain(chain)

        updated = await repo.update_chain(record.id, {"status": "completed"})
        assert updated.status == "completed"

    async def test_update_chain_missing_returns_none(self, session):
        repo = Repository(session)
        result = await repo.update_chain("no-such-id", {"status": "done"})
        assert result is None

    async def test_list_chains_limit(self, session, tenant_id):
        await self._seed_tenant(session, tenant_id)
        repo = Repository(session)
        for _ in range(10):
            await repo.create_chain(_make_chain(tenant_id))
        chains = await repo.list_chains(tenant_id, limit=5)
        assert len(chains) == 5


# ── Cost tests ────────────────────────────────────────────────────────────────

class TestCost:
    async def _seed_tenant(self, session, tenant_id):
        from nexus.db.models import TenantModel
        session.add(TenantModel(id=tenant_id, name="T", api_key_hash="h"))
        await session.commit()

    def _make_cost(self, tenant_id, usd=0.05) -> CostRecord:
        return CostRecord(
            tenant_id=tenant_id,
            chain_id="chain-x",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=200,
            cost_usd=usd,
        )

    async def test_add_and_total(self, session, tenant_id):
        await self._seed_tenant(session, tenant_id)
        repo = Repository(session)
        await repo.add_cost(self._make_cost(tenant_id, 0.05))
        await repo.add_cost(self._make_cost(tenant_id, 0.10))
        total = await repo.get_tenant_cost(tenant_id)
        assert abs(total - 0.15) < 1e-9

    async def test_empty_cost_returns_zero(self, session, tenant_id):
        await self._seed_tenant(session, tenant_id)
        repo = Repository(session)
        total = await repo.get_tenant_cost(tenant_id)
        assert total == 0.0

    async def test_month_filter(self, session, tenant_id):
        await self._seed_tenant(session, tenant_id)
        repo = Repository(session)

        cost_jan = CostRecord(
            tenant_id=tenant_id, chain_id="c", model="m",
            input_tokens=0, output_tokens=0, cost_usd=1.0,
            timestamp=datetime(2026, 1, 15),
        )
        cost_feb = CostRecord(
            tenant_id=tenant_id, chain_id="c", model="m",
            input_tokens=0, output_tokens=0, cost_usd=2.0,
            timestamp=datetime(2026, 2, 10),
        )
        await repo.add_cost(cost_jan)
        await repo.add_cost(cost_feb)

        jan_total = await repo.get_tenant_cost(tenant_id, month="2026-01")
        feb_total = await repo.get_tenant_cost(tenant_id, month="2026-02")
        assert abs(jan_total - 1.0) < 1e-9
        assert abs(feb_total - 2.0) < 1e-9


# ── Knowledge tests ───────────────────────────────────────────────────────────

class TestKnowledge:
    async def _seed_tenant(self, session, tenant_id):
        from nexus.db.models import TenantModel
        session.add(TenantModel(id=tenant_id, name="T", api_key_hash="h"))
        await session.commit()

    def _make_doc(self, tenant_id, namespace="product_docs") -> KnowledgeDocument:
        return KnowledgeDocument(
            tenant_id=tenant_id,
            namespace=namespace,
            source="readme.md",
            content="NEXUS is an AI agent framework.",
            access_level="internal",
        )

    async def test_add_and_list(self, session, tenant_id):
        await self._seed_tenant(session, tenant_id)
        repo = Repository(session)
        doc = self._make_doc(tenant_id)
        record = await repo.add_knowledge_doc(doc)
        assert record.id == doc.id
        assert record.namespace == "product_docs"

        docs = await repo.list_knowledge_docs(tenant_id)
        assert len(docs) == 1

    async def test_namespace_filter(self, session, tenant_id):
        await self._seed_tenant(session, tenant_id)
        repo = Repository(session)
        await repo.add_knowledge_doc(self._make_doc(tenant_id, "ns-a"))
        await repo.add_knowledge_doc(self._make_doc(tenant_id, "ns-b"))

        ns_a = await repo.list_knowledge_docs(tenant_id, namespace="ns-a")
        assert len(ns_a) == 1
        assert ns_a[0].namespace == "ns-a"

    async def test_list_no_namespace_returns_all(self, session, tenant_id):
        await self._seed_tenant(session, tenant_id)
        repo = Repository(session)
        await repo.add_knowledge_doc(self._make_doc(tenant_id, "ns-a"))
        await repo.add_knowledge_doc(self._make_doc(tenant_id, "ns-b"))
        all_docs = await repo.list_knowledge_docs(tenant_id)
        assert len(all_docs) == 2


# ── Seed tests ────────────────────────────────────────────────────────────────

class TestSeed:
    async def test_seed_creates_demo_tenant(self, session):
        await seed_database(session)
        from nexus.db.models import TenantModel
        from sqlalchemy import select
        result = await session.execute(select(TenantModel).where(TenantModel.id == "demo"))
        tenant = result.scalar_one_or_none()
        assert tenant is not None
        assert tenant.name == "Demo Tenant"
        expected_hash = hashlib.sha256(b"nxs_demo_key_12345").hexdigest()
        assert tenant.api_key_hash == expected_hash

    async def test_seed_creates_all_5_personas(self, session):
        await seed_database(session)
        from nexus.db.models import PersonaModel
        from sqlalchemy import select
        result = await session.execute(
            select(PersonaModel).where(PersonaModel.tenant_id == "demo")
        )
        personas = result.scalars().all()
        names = {p.name for p in personas}
        expected = {p["name"] for p in DEFAULT_PERSONAS}
        assert names == expected

    async def test_seed_is_idempotent(self, session):
        """Running seed twice must not raise or duplicate data."""
        await seed_database(session)
        await seed_database(session)  # second call — should be a no-op

        from nexus.db.models import PersonaModel
        from sqlalchemy import select
        result = await session.execute(
            select(PersonaModel).where(PersonaModel.tenant_id == "demo")
        )
        personas = result.scalars().all()
        assert len(personas) == len(DEFAULT_PERSONAS)

    async def test_persona_fields_correct(self, session):
        await seed_database(session)
        repo = Repository(session)
        researcher = await repo.get_persona_by_name("demo", "researcher")
        assert researcher is not None
        assert "knowledge_search" in researcher.allowed_tools
        assert researcher.max_ttl_seconds == 60
        assert researcher.risk_tolerance == "low"


# ── Trigger CRUD tests (Gap 9) ────────────────────────────────────────────────

class TestTriggerCRUD:
    """Tests for trigger save/get/list/update/delete. Requires tenant + workflow FKs."""

    async def _seed(self, session) -> tuple[str, str]:
        """Seed a tenant and workflow, return (tenant_id, workflow_id)."""
        from nexus.db.models import TenantModel, WorkflowModel
        tenant_id = "trigger-tenant-001"
        workflow_id = "trigger-workflow-001"
        session.add(TenantModel(id=tenant_id, name="TriggerTest", api_key_hash="h-trigger"))
        session.add(WorkflowModel(
            id=workflow_id,
            tenant_id=tenant_id,
            name="test_workflow",
            description="Workflow for trigger tests",
        ))
        await session.commit()
        return tenant_id, workflow_id

    async def test_save_and_get_trigger(self, session):
        tid, wid = await self._seed(session)
        repo = Repository(session)

        trigger = TriggerConfig(
            workflow_id=wid,
            tenant_id=tid,
            trigger_type=TriggerType.WEBHOOK,
            webhook_path="/hooks/test-abc",
        )
        saved = await repo.save_trigger(trigger)
        assert saved.id is not None
        assert saved.trigger_type == TriggerType.WEBHOOK

        fetched = await repo.get_trigger(tid, saved.id)
        assert fetched is not None
        assert fetched.trigger_type == TriggerType.WEBHOOK
        assert fetched.webhook_path == "/hooks/test-abc"

    async def test_get_trigger_by_webhook_path(self, session):
        tid, wid = await self._seed(session)
        repo = Repository(session)

        trigger = TriggerConfig(
            workflow_id=wid,
            tenant_id=tid,
            trigger_type=TriggerType.WEBHOOK,
            webhook_path="/hooks/unique-path",
        )
        saved = await repo.save_trigger(trigger)

        fetched = await repo.get_trigger_by_webhook_path("/hooks/unique-path")
        assert fetched is not None
        assert fetched.id == saved.id

    async def test_get_trigger_by_webhook_path_missing_returns_none(self, session):
        await self._seed(session)
        repo = Repository(session)
        result = await repo.get_trigger_by_webhook_path("/hooks/does-not-exist")
        assert result is None

    async def test_list_triggers_by_tenant(self, session):
        tid, wid = await self._seed(session)
        repo = Repository(session)

        for i in range(3):
            await repo.save_trigger(TriggerConfig(
                workflow_id=wid,
                tenant_id=tid,
                trigger_type=TriggerType.MANUAL,
            ))

        triggers = await repo.list_triggers(tid)
        assert len(triggers) == 3

    async def test_list_triggers_enabled_filter(self, session):
        tid, wid = await self._seed(session)
        repo = Repository(session)

        enabled_t = TriggerConfig(workflow_id=wid, tenant_id=tid, trigger_type=TriggerType.MANUAL, enabled=True)
        disabled_t = TriggerConfig(workflow_id=wid, tenant_id=tid, trigger_type=TriggerType.MANUAL, enabled=False)
        await repo.save_trigger(enabled_t)
        await repo.save_trigger(disabled_t)

        enabled_only = await repo.list_triggers(tid, enabled=True)
        disabled_only = await repo.list_triggers(tid, enabled=False)
        assert len(enabled_only) == 1
        assert len(disabled_only) == 1

    async def test_update_trigger(self, session):
        tid, wid = await self._seed(session)
        repo = Repository(session)

        trigger = TriggerConfig(
            workflow_id=wid,
            tenant_id=tid,
            trigger_type=TriggerType.WEBHOOK,
            enabled=True,
        )
        saved = await repo.save_trigger(trigger)
        saved = saved.model_copy(update={"enabled": False})
        updated = await repo.update_trigger(saved)

        assert updated.enabled is False

    async def test_delete_trigger_returns_true(self, session):
        tid, wid = await self._seed(session)
        repo = Repository(session)

        trigger = TriggerConfig(workflow_id=wid, tenant_id=tid, trigger_type=TriggerType.MANUAL)
        saved = await repo.save_trigger(trigger)

        deleted = await repo.delete_trigger(tid, saved.id)
        assert deleted is True

        fetched = await repo.get_trigger(tid, saved.id)
        assert fetched is None

    async def test_delete_nonexistent_returns_false(self, session):
        await self._seed(session)
        repo = Repository(session)
        result = await repo.delete_trigger("trigger-tenant-001", "nonexistent-id")
        assert result is False

    async def test_trigger_type_enum_roundtrips(self, session):
        """TriggerType enum value stored as string, read back as TriggerType enum."""
        tid, wid = await self._seed(session)
        repo = Repository(session)

        trigger = TriggerConfig(
            workflow_id=wid,
            tenant_id=tid,
            trigger_type=TriggerType.CRON,
        )
        saved = await repo.save_trigger(trigger)
        fetched = await repo.get_trigger(tid, saved.id)

        assert isinstance(fetched.trigger_type, TriggerType)
        assert fetched.trigger_type == TriggerType.CRON


# ── get_tenant_by_api_key_hash tests (Gap 10) ─────────────────────────────────

class TestGetTenantByApiKeyHash:

    async def test_returns_tenant_for_known_hash(self, session):
        from nexus.db.models import TenantModel
        repo = Repository(session)
        session.add(TenantModel(id="hash-tenant-1", name="HashTest", api_key_hash="known_hash_abc"))
        await session.commit()

        tenant = await repo.get_tenant_by_api_key_hash("known_hash_abc")
        assert tenant is not None
        assert tenant.id == "hash-tenant-1"
        assert tenant.name == "HashTest"

    async def test_returns_none_for_unknown_hash(self, session):
        repo = Repository(session)
        tenant = await repo.get_tenant_by_api_key_hash("nonexistent_hash_xyz")
        assert tenant is None

    async def test_hash_lookup_is_exact_match(self, session):
        """Partial hash match must NOT return a result."""
        from nexus.db.models import TenantModel
        repo = Repository(session)
        session.add(TenantModel(id="exact-tenant", name="Exact", api_key_hash="exact_hash_value"))
        await session.commit()

        result = await repo.get_tenant_by_api_key_hash("exact_hash")  # prefix only
        assert result is None


# ── Seal JSON round-trip (Gap 11) ─────────────────────────────────────────────

class TestSealFullRoundTrip:
    """Verify that Seal.anomaly_result with 4 gate verdicts survives a DB roundtrip."""

    async def _seed_chain(self, session, tenant_id: str, chain_id: str) -> None:
        from nexus.db.models import TenantModel, ChainModel
        session.add(TenantModel(id=tenant_id, name="RT", api_key_hash="h-rt"))
        session.add(ChainModel(
            id=chain_id, tenant_id=tenant_id, task="roundtrip test", steps=[], status="executing",
        ))
        await session.commit()

    async def test_seal_nested_json_survives_roundtrip(self, session):
        """All 4 gate verdicts stored as JSON must be read back correctly."""
        tid = "rt-tenant-001"
        cid = "rt-chain-001"
        await self._seed_chain(session, tid, cid)
        repo = Repository(session)

        # Build a seal with all 4 gates having distinct verdicts
        gate_verdicts = [
            (GateVerdict.PASS, "scope"),
            (GateVerdict.PASS, "intent"),
            (GateVerdict.PASS, "ttl"),
            (GateVerdict.SKIP, "drift"),
        ]
        gates = [
            GateResult(
                gate_name=name,
                verdict=verdict,
                score=0.9 if verdict == GateVerdict.PASS else 0.0,
                threshold=0.75,
                details=f"{name} gate result",
            )
            for verdict, name in gate_verdicts
        ]
        intent = IntentDeclaration(
            task_description="roundtrip task",
            planned_action="search for info",
            tool_name="knowledge_search",
            tool_params={"query": "roundtrip"},
            resource_targets=["kb:roundtrip"],
            reasoning="test reasoning",
            confidence=0.95,
        )
        anomaly = AnomalyResult(
            gates=gates,
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_id="researcher",
            action_fingerprint="fp-roundtrip-001",
        )
        seal = Seal(
            chain_id=cid,
            step_index=0,
            tenant_id=tid,
            persona_id="researcher",
            intent=intent,
            anomaly_result=anomaly,
            tool_name="knowledge_search",
            tool_params={"query": "roundtrip"},
            status=ActionStatus.EXECUTED,
            fingerprint="fp-roundtrip-001",
        )
        await repo.create_seal(seal)

        # Read back via get_chain_seals
        seals = await repo.get_chain_seals(cid)
        assert len(seals) == 1

        record = seals[0]
        # anomaly_result is stored as JSON dict in the DB
        stored_anomaly = record.anomaly_result
        assert isinstance(stored_anomaly, dict)
        stored_gates = stored_anomaly["gates"]
        assert len(stored_gates) == 4

        gate_names_verdicts = {g["gate_name"]: g["verdict"] for g in stored_gates}
        assert gate_names_verdicts["scope"] == "pass"
        assert gate_names_verdicts["intent"] == "pass"
        assert gate_names_verdicts["ttl"] == "pass"
        assert gate_names_verdicts["drift"] == "skip"
