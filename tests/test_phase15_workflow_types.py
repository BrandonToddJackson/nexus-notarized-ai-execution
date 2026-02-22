"""
Phase 15 smoketest — Workflow, Trigger, Credential, MCP foundation.

Coverage:
  Exceptions  — full hierarchy assertions, attribute storage, isinstance catch
  Config      — all 16 new fields present with correct defaults and types
  DB Models   — schema creation, CRUD via SQLite in-memory, index/constraint
                names verified from live metadata
"""

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nexus.db.models import Base


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
async def db():
    """In-memory SQLite with full schema including Phase 15 tables."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase15Exceptions:
    """Exception hierarchy, attribute storage, and isinstance-catch semantics."""

    # ── WorkflowError base ────────────────────────────────────────────────────

    def test_workflow_error_is_nexus_error(self):
        from nexus.exceptions import NexusError, WorkflowError
        assert issubclass(WorkflowError, NexusError)

    def test_workflow_error_raises_and_catches_as_nexus_error(self):
        from nexus.exceptions import NexusError, WorkflowError
        with pytest.raises(NexusError):
            raise WorkflowError("base workflow error")

    # ── WorkflowNotFound ──────────────────────────────────────────────────────

    def test_workflow_not_found_is_workflow_error(self):
        from nexus.exceptions import WorkflowError, WorkflowNotFound
        assert issubclass(WorkflowNotFound, WorkflowError)

    def test_workflow_not_found_catches_as_workflow_error(self):
        from nexus.exceptions import WorkflowError, WorkflowNotFound
        with pytest.raises(WorkflowError):
            raise WorkflowNotFound("not found", workflow_id="wf-abc")

    def test_workflow_not_found_stores_workflow_id(self):
        from nexus.exceptions import WorkflowNotFound
        exc = WorkflowNotFound("missing", workflow_id="wf-xyz")
        assert exc.workflow_id == "wf-xyz"
        assert str(exc) == "missing"

    def test_workflow_not_found_default_workflow_id(self):
        from nexus.exceptions import WorkflowNotFound
        exc = WorkflowNotFound("missing")
        assert exc.workflow_id == ""

    # ── WorkflowValidationError ───────────────────────────────────────────────

    def test_workflow_validation_error_is_workflow_error(self):
        from nexus.exceptions import WorkflowError, WorkflowValidationError
        assert issubclass(WorkflowValidationError, WorkflowError)

    def test_workflow_validation_error_stores_violations(self):
        from nexus.exceptions import WorkflowValidationError
        violations = ["cycle detected", "missing edge"]
        exc = WorkflowValidationError("invalid dag", violations=violations)
        assert exc.violations == violations

    def test_workflow_validation_error_defaults_to_empty_list(self):
        from nexus.exceptions import WorkflowValidationError
        exc = WorkflowValidationError("bad")
        assert exc.violations == []

    # ── TriggerError ──────────────────────────────────────────────────────────

    def test_trigger_error_is_nexus_error(self):
        from nexus.exceptions import NexusError, TriggerError
        assert issubclass(TriggerError, NexusError)

    def test_trigger_error_stores_trigger_type(self):
        from nexus.exceptions import TriggerError
        exc = TriggerError("fire failed", trigger_type="cron")
        assert exc.trigger_type == "cron"

    # ── CredentialError + CredentialNotFound ──────────────────────────────────

    def test_credential_error_stores_credential_id(self):
        from nexus.exceptions import CredentialError
        exc = CredentialError("decrypt failed", credential_id="cred-001")
        assert exc.credential_id == "cred-001"

    def test_credential_not_found_is_credential_error(self):
        from nexus.exceptions import CredentialError, CredentialNotFound
        assert issubclass(CredentialNotFound, CredentialError)

    def test_credential_not_found_catches_as_credential_error(self):
        from nexus.exceptions import CredentialError, CredentialNotFound
        with pytest.raises(CredentialError):
            raise CredentialNotFound("not found", credential_id="cred-999")

    def test_credential_not_found_inherits_credential_id(self):
        from nexus.exceptions import CredentialNotFound
        exc = CredentialNotFound("missing", credential_id="cred-42")
        assert exc.credential_id == "cred-42"

    # ── MCPConnectionError ────────────────────────────────────────────────────

    def test_mcp_connection_error_is_nexus_error(self):
        from nexus.exceptions import MCPConnectionError, NexusError
        assert issubclass(MCPConnectionError, NexusError)

    def test_mcp_connection_error_stores_server_name(self):
        from nexus.exceptions import MCPConnectionError
        exc = MCPConnectionError("timeout", server_name="filesystem")
        assert exc.server_name == "filesystem"

    # ── MCPToolError ──────────────────────────────────────────────────────────

    def test_mcp_tool_error_is_tool_error(self):
        from nexus.exceptions import MCPToolError, ToolError
        assert issubclass(MCPToolError, ToolError)

    def test_mcp_tool_error_catches_as_tool_error(self):
        from nexus.exceptions import MCPToolError, ToolError
        with pytest.raises(ToolError):
            raise MCPToolError("tool failed", tool_name="read_file")

    def test_mcp_tool_error_stores_tool_name(self):
        from nexus.exceptions import MCPToolError
        exc = MCPToolError("exec error", tool_name="write_file")
        assert exc.tool_name == "write_file"

    # ── SandboxError ──────────────────────────────────────────────────────────

    def test_sandbox_error_is_tool_error(self):
        from nexus.exceptions import SandboxError, ToolError
        assert issubclass(SandboxError, ToolError)

    def test_sandbox_error_catches_as_tool_error(self):
        from nexus.exceptions import SandboxError, ToolError
        with pytest.raises(ToolError):
            raise SandboxError("memory limit exceeded", tool_name="code_exec")

    def test_sandbox_error_stores_tool_name(self):
        from nexus.exceptions import SandboxError
        exc = SandboxError("timeout", tool_name="run_python")
        assert exc.tool_name == "run_python"

    # ── Full chain: WorkflowNotFound → WorkflowError → NexusError → Exception ─

    def test_workflow_not_found_is_base_exception(self):
        from nexus.exceptions import WorkflowNotFound
        with pytest.raises(Exception):
            raise WorkflowNotFound("any")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase15Config:
    """All 16 new Phase 15 config fields — defaults, types, values."""

    @pytest.fixture
    def cfg(self):
        from nexus.config import NexusConfig
        return NexusConfig()

    # ── Workflow fields ───────────────────────────────────────────────────────

    def test_max_workflow_steps_default(self, cfg):
        assert cfg.max_workflow_steps == 50

    def test_max_concurrent_workflows_default(self, cfg):
        assert cfg.max_concurrent_workflows == 10

    def test_workflow_execution_timeout_default(self, cfg):
        assert cfg.workflow_execution_timeout == 3600

    # ── Trigger fields ────────────────────────────────────────────────────────

    def test_webhook_base_url_default(self, cfg):
        assert cfg.webhook_base_url == "http://localhost:8000"

    def test_cron_check_interval_default(self, cfg):
        assert cfg.cron_check_interval == 15

    def test_max_triggers_per_workflow_default(self, cfg):
        assert cfg.max_triggers_per_workflow == 5

    # ── Credential fields ─────────────────────────────────────────────────────

    def test_credential_encryption_key_default(self, cfg):
        assert cfg.credential_encryption_key == ""

    def test_credential_max_per_tenant_default(self, cfg):
        assert cfg.credential_max_per_tenant == 100

    # ── MCP fields ────────────────────────────────────────────────────────────

    def test_mcp_connection_timeout_default(self, cfg):
        assert cfg.mcp_connection_timeout == 10

    def test_mcp_tool_timeout_default(self, cfg):
        assert cfg.mcp_tool_timeout == 60

    def test_mcp_max_servers_default(self, cfg):
        assert cfg.mcp_max_servers == 20

    # ── Sandbox fields ────────────────────────────────────────────────────────

    def test_sandbox_max_memory_mb_default(self, cfg):
        assert cfg.sandbox_max_memory_mb == 256

    def test_sandbox_max_execution_seconds_default(self, cfg):
        assert cfg.sandbox_max_execution_seconds == 30

    def test_sandbox_allowed_imports_is_list(self, cfg):
        assert isinstance(cfg.sandbox_allowed_imports, list)
        assert len(cfg.sandbox_allowed_imports) > 0

    def test_sandbox_allowed_imports_safe_stdlib(self, cfg):
        for mod in ("json", "math", "re", "datetime", "hashlib"):
            assert mod in cfg.sandbox_allowed_imports, f"{mod!r} missing from allowed imports"

    def test_sandbox_allowed_imports_excludes_dangerous(self, cfg):
        dangerous = ("os", "sys", "subprocess", "importlib", "shutil")
        for mod in dangerous:
            assert mod not in cfg.sandbox_allowed_imports, f"{mod!r} should not be allowed"

    # ── Background execution fields ───────────────────────────────────────────

    def test_worker_concurrency_default(self, cfg):
        assert cfg.worker_concurrency == 4

    def test_task_queue_url_default(self, cfg):
        assert cfg.task_queue_url == "redis://localhost:6379/1"

    def test_task_queue_url_uses_different_db_than_cache(self, cfg):
        # cache uses /0, task queue uses /1 to avoid key collisions
        assert cfg.task_queue_url.endswith("/1")
        assert cfg.redis_url.endswith("/0")

    # ── Types ─────────────────────────────────────────────────────────────────

    def test_all_int_fields_are_int(self, cfg):
        assert isinstance(cfg.max_workflow_steps, int)
        assert isinstance(cfg.max_concurrent_workflows, int)
        assert isinstance(cfg.workflow_execution_timeout, int)
        assert isinstance(cfg.cron_check_interval, int)
        assert isinstance(cfg.max_triggers_per_workflow, int)
        assert isinstance(cfg.credential_max_per_tenant, int)
        assert isinstance(cfg.mcp_connection_timeout, int)
        assert isinstance(cfg.mcp_tool_timeout, int)
        assert isinstance(cfg.mcp_max_servers, int)
        assert isinstance(cfg.sandbox_max_memory_mb, int)
        assert isinstance(cfg.sandbox_max_execution_seconds, int)
        assert isinstance(cfg.worker_concurrency, int)

    def test_all_str_fields_are_str(self, cfg):
        assert isinstance(cfg.webhook_base_url, str)
        assert isinstance(cfg.credential_encryption_key, str)
        assert isinstance(cfg.task_queue_url, str)


# ─────────────────────────────────────────────────────────────────────────────
# DB MODELS
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase15Models:
    """Schema creation + CRUD + index/constraint metadata for all Phase 15 models."""

    # ── Table names ───────────────────────────────────────────────────────────

    def test_workflow_model_tablename(self):
        from nexus.db.models import WorkflowModel
        assert WorkflowModel.__tablename__ == "workflows"

    def test_workflow_execution_model_tablename(self):
        from nexus.db.models import WorkflowExecutionModel
        assert WorkflowExecutionModel.__tablename__ == "workflow_executions"

    def test_trigger_model_tablename(self):
        from nexus.db.models import TriggerModel
        assert TriggerModel.__tablename__ == "triggers"

    def test_credential_model_tablename(self):
        from nexus.db.models import CredentialModel
        assert CredentialModel.__tablename__ == "credentials"

    def test_mcp_server_model_tablename(self):
        from nexus.db.models import MCPServerModel
        assert MCPServerModel.__tablename__ == "mcp_servers"

    # ── Column presence ───────────────────────────────────────────────────────

    def test_trigger_model_has_webhook_path(self):
        from nexus.db.models import TriggerModel
        cols = {c.key for c in TriggerModel.__table__.columns}
        assert "webhook_path" in cols

    def test_mcp_server_model_has_created_at(self):
        from nexus.db.models import MCPServerModel
        cols = {c.key for c in MCPServerModel.__table__.columns}
        assert "created_at" in cols

    def test_workflow_model_has_required_columns(self):
        from nexus.db.models import WorkflowModel
        cols = {c.key for c in WorkflowModel.__table__.columns}
        for col in ("id", "tenant_id", "name", "version", "status", "steps", "edges", "tags", "settings"):
            assert col in cols, f"WorkflowModel missing column: {col}"

    def test_credential_model_has_required_columns(self):
        from nexus.db.models import CredentialModel
        cols = {c.key for c in CredentialModel.__table__.columns}
        for col in ("id", "tenant_id", "name", "credential_type", "service_name",
                    "encrypted_data", "scoped_personas", "expires_at"):
            assert col in cols, f"CredentialModel missing column: {col}"

    # ── Constraint / index names (inspected from live table metadata) ─────────

    @pytest.mark.asyncio
    async def test_workflow_unique_constraint_name(self, db):
        engine = db.bind
        async with engine.connect() as conn:
            table_info = await conn.run_sync(
                lambda sync_conn: sa_inspect(sync_conn).get_unique_constraints("workflows")
            )
        names = {c["name"] for c in table_info}
        assert "uq_workflow_tenant_name_version" in names

    @pytest.mark.asyncio
    async def test_workflow_index_on_status(self, db):
        engine = db.bind
        async with engine.connect() as conn:
            indexes = await conn.run_sync(
                lambda sync_conn: sa_inspect(sync_conn).get_indexes("workflows")
            )
        index_names = {i["name"] for i in indexes}
        assert "ix_workflow_tenant_status" in index_names

    @pytest.mark.asyncio
    async def test_workflow_execution_index_includes_started_at(self, db):
        engine = db.bind
        async with engine.connect() as conn:
            indexes = await conn.run_sync(
                lambda sync_conn: sa_inspect(sync_conn).get_indexes("workflow_executions")
            )
        name_map = {i["name"]: i["column_names"] for i in indexes}
        idx = name_map.get("ix_wf_exec_tenant_workflow_started", [])
        assert "started_at" in idx

    @pytest.mark.asyncio
    async def test_credential_index_on_service_name(self, db):
        engine = db.bind
        async with engine.connect() as conn:
            indexes = await conn.run_sync(
                lambda sync_conn: sa_inspect(sync_conn).get_indexes("credentials")
            )
        index_names = {i["name"] for i in indexes}
        assert "ix_credential_tenant_service" in index_names

    @pytest.mark.asyncio
    async def test_mcp_server_index_on_enabled(self, db):
        engine = db.bind
        async with engine.connect() as conn:
            indexes = await conn.run_sync(
                lambda sync_conn: sa_inspect(sync_conn).get_indexes("mcp_servers")
            )
        index_names = {i["name"] for i in indexes}
        assert "ix_mcp_tenant_enabled" in index_names

    # ── CRUD: WorkflowModel ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_workflow_insert_and_query(self, db):
        from nexus.db.models import WorkflowModel
        row = WorkflowModel(
            tenant_id="t1", name="my-wf", version=1, status="draft",
            steps=[{"id": "step1"}], edges=[], tags=["test"], settings={},
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        assert row.id is not None
        assert row.name == "my-wf"
        assert row.status == "draft"
        assert row.version == 1

    @pytest.mark.asyncio
    async def test_workflow_unique_constraint_enforced(self, db):
        from sqlalchemy.exc import IntegrityError
        from nexus.db.models import WorkflowModel
        w1 = WorkflowModel(tenant_id="t1", name="dup-wf", version=1, status="draft")
        w2 = WorkflowModel(tenant_id="t1", name="dup-wf", version=1, status="active")
        db.add(w1)
        await db.commit()
        db.add(w2)
        with pytest.raises(IntegrityError):
            await db.commit()

    # ── CRUD: WorkflowExecutionModel ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_workflow_execution_insert_and_query(self, db):
        from nexus.db.models import WorkflowModel, WorkflowExecutionModel
        wf = WorkflowModel(tenant_id="t1", name="exec-wf", version=1)
        db.add(wf)
        await db.commit()
        await db.refresh(wf)

        exe = WorkflowExecutionModel(
            workflow_id=wf.id, workflow_version=1, tenant_id="t1",
            trigger_type="manual", trigger_data={}, status="planning",
        )
        db.add(exe)
        await db.commit()
        await db.refresh(exe)
        assert exe.id is not None
        assert exe.workflow_id == wf.id
        assert exe.trigger_type == "manual"

    # ── CRUD: TriggerModel ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_trigger_insert_with_webhook_path(self, db):
        from nexus.db.models import WorkflowModel, TriggerModel
        wf = WorkflowModel(tenant_id="t1", name="trig-wf", version=1)
        db.add(wf)
        await db.commit()
        await db.refresh(wf)

        trig = TriggerModel(
            workflow_id=wf.id, tenant_id="t1", trigger_type="webhook",
            enabled=True, config={}, webhook_path="/webhooks/abc123",
        )
        db.add(trig)
        await db.commit()
        await db.refresh(trig)
        assert trig.id is not None
        assert trig.webhook_path == "/webhooks/abc123"
        assert trig.enabled is True

    @pytest.mark.asyncio
    async def test_trigger_webhook_path_unique_enforced(self, db):
        from sqlalchemy.exc import IntegrityError
        from nexus.db.models import WorkflowModel, TriggerModel
        wf = WorkflowModel(tenant_id="t1", name="trig-wf2", version=1)
        db.add(wf)
        await db.commit()
        await db.refresh(wf)

        t1 = TriggerModel(workflow_id=wf.id, tenant_id="t1", trigger_type="webhook",
                          webhook_path="/hooks/same")
        t2 = TriggerModel(workflow_id=wf.id, tenant_id="t1", trigger_type="webhook",
                          webhook_path="/hooks/same")
        db.add(t1)
        await db.commit()
        db.add(t2)
        with pytest.raises(IntegrityError):
            await db.commit()

    # ── CRUD: CredentialModel ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_credential_insert_and_query(self, db):
        from nexus.db.models import CredentialModel
        cred = CredentialModel(
            tenant_id="t1", name="github-token", credential_type="api_key",
            service_name="github", encrypted_data="ENCRYPTED_BLOB",
            scoped_personas=["researcher"],
        )
        db.add(cred)
        await db.commit()
        await db.refresh(cred)
        assert cred.id is not None
        assert cred.service_name == "github"
        assert cred.encrypted_data == "ENCRYPTED_BLOB"
        assert cred.scoped_personas == ["researcher"]

    # ── CRUD: MCPServerModel ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_mcp_server_insert_and_query(self, db):
        from nexus.db.models import MCPServerModel
        srv = MCPServerModel(
            tenant_id="t1", name="filesystem", url="http://localhost:9000",
            transport="sse", enabled=True, discovered_tools=[{"name": "read_file"}],
        )
        db.add(srv)
        await db.commit()
        await db.refresh(srv)
        assert srv.id is not None
        assert srv.name == "filesystem"
        assert srv.created_at is not None
        assert srv.discovered_tools == [{"name": "read_file"}]

    @pytest.mark.asyncio
    async def test_mcp_server_enabled_defaults_true(self, db):
        from nexus.db.models import MCPServerModel
        srv = MCPServerModel(
            tenant_id="t1", name="memory-mcp", url="stdio://", transport="stdio",
        )
        db.add(srv)
        await db.commit()
        await db.refresh(srv)
        assert srv.enabled is True
