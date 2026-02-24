"""Phase 28 — Persistence v2: smoke tests for 15 new repository methods.

Tests use an in-memory SQLite DB via pytest-asyncio — no real Postgres needed.
Pattern matches existing Phase tests (e.g. test_phase22_triggers.py).
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from nexus.db.models import Base
from nexus.db.repository import Repository


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def session():
    """In-memory SQLite async session with all tables created."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as s:
        # Seed a demo tenant so FK constraints pass
        from nexus.db.models import TenantModel
        tenant = TenantModel(id="t1", name="Test", api_key_hash="abc")
        s.add(tenant)
        await s.commit()
        yield s

    await engine.dispose()


@pytest_asyncio.fixture
async def repo(session):
    return Repository(session)


# ── Workflow CRUD ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_and_get_workflow(repo):
    wf = await repo.save_workflow(
        tenant_id="t1",
        name="My Flow",
        description="Test workflow",
        steps=[{"id": "s1"}],
        edges=[],
        status="draft",
        version=1,
    )
    assert wf.id is not None
    assert wf.name == "My Flow"

    fetched = await repo.get_workflow("t1", wf.id)
    assert fetched is not None
    assert fetched.id == wf.id
    assert fetched.description == "Test workflow"


@pytest.mark.asyncio
async def test_get_workflow_tenant_isolation(repo):
    wf = await repo.save_workflow(tenant_id="t1", name="Flow A")
    result = await repo.get_workflow("other-tenant", wf.id)
    assert result is None


@pytest.mark.asyncio
async def test_list_workflows_basic(repo):
    await repo.save_workflow(tenant_id="t1", name="Flow1", status="active")
    await repo.save_workflow(tenant_id="t1", name="Flow2", status="draft")

    all_flows = await repo.list_workflows("t1")
    assert len(all_flows) == 2

    active = await repo.list_workflows("t1", status="active")
    assert len(active) == 1
    assert active[0].name == "Flow1"


@pytest.mark.asyncio
async def test_list_workflows_tags_filter(repo):
    await repo.save_workflow(tenant_id="t1", name="TaggedFlow", tags=["email", "automation"])
    await repo.save_workflow(tenant_id="t1", name="PlainFlow", tags=[])

    email_flows = await repo.list_workflows("t1", tags=["email"])
    assert len(email_flows) == 1
    assert email_flows[0].name == "TaggedFlow"


@pytest.mark.asyncio
async def test_list_workflows_latest_only(repo):
    """latest_only=True deduplicates by name, keeping highest version."""
    await repo.save_workflow(tenant_id="t1", name="Versioned", version=1)
    await repo.save_workflow(tenant_id="t1", name="Versioned", version=2)

    latest = await repo.list_workflows("t1", latest_only=True)
    versioned = [w for w in latest if w.name == "Versioned"]
    assert len(versioned) == 1
    assert versioned[0].version == 2


@pytest.mark.asyncio
async def test_update_workflow(repo):
    wf = await repo.save_workflow(tenant_id="t1", name="Flow", status="draft")
    updated = await repo.update_workflow("t1", wf.id, {"status": "active", "description": "Updated"})
    assert updated is not None
    assert updated.status == "active"
    assert updated.description == "Updated"


@pytest.mark.asyncio
async def test_update_workflow_unknown_key_raises(repo):
    wf = await repo.save_workflow(tenant_id="t1", name="Flow")
    with pytest.raises(ValueError, match="Unknown workflow update keys"):
        await repo.update_workflow("t1", wf.id, {"nonexistent_field": "x"})


@pytest.mark.asyncio
async def test_update_workflow_not_found(repo):
    result = await repo.update_workflow("t1", "nonexistent-id", {"status": "active"})
    assert result is None


@pytest.mark.asyncio
async def test_delete_workflow(repo):
    wf = await repo.save_workflow(tenant_id="t1", name="ToDelete")
    deleted = await repo.delete_workflow("t1", wf.id)
    assert deleted is True

    fetched = await repo.get_workflow("t1", wf.id)
    assert fetched is None


@pytest.mark.asyncio
async def test_delete_workflow_not_found(repo):
    result = await repo.delete_workflow("t1", "no-such-id")
    assert result is False


@pytest.mark.asyncio
async def test_delete_workflow_cascades_triggers(repo, session):
    """Deleting a workflow removes its triggers."""
    from nexus.db.models import TriggerModel
    from sqlalchemy import select

    wf = await repo.save_workflow(tenant_id="t1", name="WithTrigger")
    trigger = TriggerModel(
        id="trig-1",
        workflow_id=wf.id,
        tenant_id="t1",
        trigger_type="webhook",
        webhook_path="/test/path",
    )
    session.add(trigger)
    await session.commit()

    await repo.delete_workflow("t1", wf.id)

    result = await session.execute(
        select(TriggerModel).where(TriggerModel.id == "trig-1")
    )
    assert result.scalar_one_or_none() is None


# ── Credential CRUD ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_and_get_credential(repo):
    cred = await repo.save_credential(
        tenant_id="t1",
        name="My API Key",
        credential_type="api_key",
        service_name="openai",
        encrypted_data="enc-abc123",
        scoped_personas=["researcher"],
    )
    assert cred.id is not None
    assert cred.is_active is True

    fetched = await repo.get_credential("t1", cred.id)
    assert fetched is not None
    assert fetched.name == "My API Key"
    assert fetched.service_name == "openai"


@pytest.mark.asyncio
async def test_get_credential_tenant_isolation(repo):
    cred = await repo.save_credential(
        tenant_id="t1",
        name="Cred",
        credential_type="api_key",
        service_name="github",
        encrypted_data="enc",
    )
    result = await repo.get_credential("other-tenant", cred.id)
    assert result is None


@pytest.mark.asyncio
async def test_get_credential_excludes_inactive(repo):
    cred = await repo.save_credential(
        tenant_id="t1", name="InactiveCred",
        credential_type="api_key", service_name="aws", encrypted_data="enc"
    )
    await repo.delete_credential("t1", cred.id)  # soft delete

    # Default: exclude inactive
    result = await repo.get_credential("t1", cred.id)
    assert result is None

    # include_inactive=True: still accessible
    result = await repo.get_credential("t1", cred.id, include_inactive=True)
    assert result is not None
    assert result.is_active is False


@pytest.mark.asyncio
async def test_list_credentials_service_filter(repo):
    await repo.save_credential("t1", "Key1", "api_key", "openai", "enc1")
    await repo.save_credential("t1", "Key2", "api_key", "anthropic", "enc2")

    openai_creds = await repo.list_credentials("t1", service_name="openai")
    assert len(openai_creds) == 1
    assert openai_creds[0].name == "Key1"


@pytest.mark.asyncio
async def test_list_credentials_persona_filter(repo):
    await repo.save_credential(
        "t1", "Key1", "api_key", "openai", "enc1",
        scoped_personas=["researcher", "analyst"]
    )
    await repo.save_credential(
        "t1", "Key2", "bearer_token", "slack", "enc2",
        scoped_personas=["communicator"]
    )

    researcher_creds = await repo.list_credentials("t1", persona_name="researcher")
    assert len(researcher_creds) == 1
    assert researcher_creds[0].name == "Key1"


@pytest.mark.asyncio
async def test_update_credential(repo):
    cred = await repo.save_credential(
        "t1", "OldName", "api_key", "openai", "enc"
    )
    updated = await repo.update_credential("t1", cred.id, {"name": "NewName"})
    assert updated is not None
    assert updated.name == "NewName"


@pytest.mark.asyncio
async def test_update_credential_unknown_key_raises(repo):
    cred = await repo.save_credential("t1", "Cred", "api_key", "aws", "enc")
    with pytest.raises(ValueError, match="Unknown credential update keys"):
        await repo.update_credential("t1", cred.id, {"bad_field": "x"})


@pytest.mark.asyncio
async def test_delete_credential_soft_delete(repo):
    """Soft-delete sets is_active=False, record still exists."""
    cred = await repo.save_credential("t1", "SoftDel", "api_key", "openai", "enc")
    result = await repo.delete_credential("t1", cred.id)
    assert result is True

    # Record still exists with is_active=False
    fetched = await repo.get_credential("t1", cred.id, include_inactive=True)
    assert fetched is not None
    assert fetched.is_active is False


@pytest.mark.asyncio
async def test_delete_credential_not_found(repo):
    result = await repo.delete_credential("t1", "no-such-cred")
    assert result is False


# ── MCP Server CRUD ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_and_get_mcp_server_sse(repo):
    server = await repo.save_mcp_server(
        tenant_id="t1",
        name="My MCP",
        transport="sse",
        url="http://localhost:3000/sse",
    )
    assert server.id is not None
    assert server.transport == "sse"
    assert server.url == "http://localhost:3000/sse"

    fetched = await repo.get_mcp_server("t1", server.id)
    assert fetched is not None
    assert fetched.name == "My MCP"


@pytest.mark.asyncio
async def test_save_mcp_server_stdio(repo):
    server = await repo.save_mcp_server(
        tenant_id="t1",
        name="Stdio MCP",
        transport="stdio",
        command="/usr/bin/my-mcp",
        args=["--verbose"],
    )
    assert server.command == "/usr/bin/my-mcp"
    assert server.url is None


@pytest.mark.asyncio
async def test_save_mcp_server_invalid_transport_raises(repo):
    with pytest.raises(ValueError, match="stdio transport requires"):
        await repo.save_mcp_server(
            tenant_id="t1", name="Bad", transport="stdio"  # no command
        )


@pytest.mark.asyncio
async def test_save_mcp_server_sse_without_url_raises(repo):
    with pytest.raises(ValueError, match="sse transport requires"):
        await repo.save_mcp_server(
            tenant_id="t1", name="Bad", transport="sse"  # no url
        )


@pytest.mark.asyncio
async def test_get_mcp_server_tenant_isolation(repo):
    server = await repo.save_mcp_server(
        "t1", "Server", transport="sse", url="http://x.y"
    )
    result = await repo.get_mcp_server("other-tenant", server.id)
    assert result is None


@pytest.mark.asyncio
async def test_list_mcp_servers_enabled_filter(repo):
    await repo.save_mcp_server("t1", "Active", transport="sse", url="http://a.b", enabled=True)
    await repo.save_mcp_server("t1", "Inactive", transport="sse", url="http://c.d", enabled=False)

    active = await repo.list_mcp_servers("t1", enabled=True)
    assert len(active) == 1
    assert active[0].name == "Active"

    all_servers = await repo.list_mcp_servers("t1")
    assert len(all_servers) == 2


@pytest.mark.asyncio
async def test_list_mcp_servers_ordered_by_name(repo):
    await repo.save_mcp_server("t1", "Zebra", transport="sse", url="http://z")
    await repo.save_mcp_server("t1", "Alpha", transport="sse", url="http://a")

    servers = await repo.list_mcp_servers("t1")
    assert servers[0].name == "Alpha"
    assert servers[1].name == "Zebra"


@pytest.mark.asyncio
async def test_update_mcp_server(repo):
    server = await repo.save_mcp_server(
        "t1", "OldName", transport="sse", url="http://old"
    )
    updated = await repo.update_mcp_server(
        "t1", server.id, {"name": "NewName", "url": "http://new"}
    )
    assert updated is not None
    assert updated.name == "NewName"
    assert updated.url == "http://new"


@pytest.mark.asyncio
async def test_update_mcp_server_unknown_key_raises(repo):
    server = await repo.save_mcp_server("t1", "S", transport="sse", url="http://x")
    with pytest.raises(ValueError, match="Unknown MCP server update keys"):
        await repo.update_mcp_server("t1", server.id, {"bad": "x"})


@pytest.mark.asyncio
async def test_update_mcp_server_transport_revalidation(repo):
    """Changing transport to stdio without providing command should fail."""
    server = await repo.save_mcp_server("t1", "S", transport="sse", url="http://x")
    with pytest.raises(ValueError, match="stdio transport requires"):
        await repo.update_mcp_server("t1", server.id, {"transport": "stdio"})


@pytest.mark.asyncio
async def test_delete_mcp_server_hard_delete(repo):
    """MCP servers are hard-deleted (no soft-delete)."""
    server = await repo.save_mcp_server("t1", "ToDelete", transport="sse", url="http://x")
    result = await repo.delete_mcp_server("t1", server.id)
    assert result is True

    fetched = await repo.get_mcp_server("t1", server.id)
    assert fetched is None


@pytest.mark.asyncio
async def test_delete_mcp_server_not_found(repo):
    result = await repo.delete_mcp_server("t1", "no-such-server")
    assert result is False


# ── Repository method presence check ─────────────────────────────────────────

def test_all_15_methods_present():
    methods = [
        "save_workflow", "get_workflow", "list_workflows", "update_workflow", "delete_workflow",
        "save_credential", "get_credential", "list_credentials", "update_credential", "delete_credential",
        "save_mcp_server", "get_mcp_server", "list_mcp_servers", "update_mcp_server", "delete_mcp_server",
    ]
    missing = [m for m in methods if not hasattr(Repository, m)]
    assert not missing, f"Missing repository methods: {missing}"
