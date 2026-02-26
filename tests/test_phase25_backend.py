"""Phase 25: Skills, Credentials, MCP, Executions — backend test suite.

All tests use mocks — no LLM calls, no DB, no I/O.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from nexus.config import NexusConfig
from nexus.exceptions import SkillNotFound, SkillValidationError
from nexus.types import SkillRecord


# ── Helpers ──────────────────────────────────────────────────────────────────

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)

def make_skill(**overrides) -> SkillRecord:
    defaults = dict(
        id="sk-1",
        tenant_id="t1",
        name="my-skill",
        display_name="My Skill",
        description="A test skill",
        content="# Hello\nSome content here.",
        version=1,
        version_history=[],
        allowed_tools=["knowledge_search"],
        allowed_personas=["researcher"],
        tags=["test"],
        supporting_files=[],
        invocation_count=0,
        last_invoked_at=None,
        active=True,
        created_at=NOW,
        updated_at=NOW,
    )
    defaults.update(overrides)
    return SkillRecord(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# SkillManager — unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkillManager:

    @pytest.fixture
    def manager(self):
        from nexus.skills.manager import SkillManager
        return SkillManager(repository=None, embedding_service=None, config=NexusConfig())

    @pytest.mark.asyncio
    async def test_create_sets_id(self, manager):
        skill = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        assert skill.id
        assert len(skill.id) == 36  # UUID

    @pytest.mark.asyncio
    async def test_create_sets_version_1(self, manager):
        skill = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        assert skill.version == 1

    @pytest.mark.asyncio
    async def test_create_sets_created_at(self, manager):
        skill = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        assert skill.created_at is not None
        assert skill.created_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_create_rejects_spaces_in_name(self, manager):
        with pytest.raises(SkillValidationError):
            await manager.create("t1", "my skill", "My Skill", "desc", "content")

    @pytest.mark.asyncio
    async def test_create_rejects_uppercase_name(self, manager):
        with pytest.raises(SkillValidationError):
            await manager.create("t1", "MySkill", "My Skill", "desc", "content")

    @pytest.mark.asyncio
    async def test_create_rejects_too_long_name(self, manager):
        with pytest.raises(SkillValidationError):
            await manager.create("t1", "a" * 65, "My Skill", "desc", "content")

    @pytest.mark.asyncio
    async def test_create_rejects_empty_display_name(self, manager):
        with pytest.raises(SkillValidationError):
            await manager.create("t1", "my-skill", "", "desc", "content")

    @pytest.mark.asyncio
    async def test_create_rejects_empty_content(self, manager):
        with pytest.raises(SkillValidationError):
            await manager.create("t1", "my-skill", "My Skill", "desc", "")

    @pytest.mark.asyncio
    async def test_get_raises_for_unknown_id(self, manager):
        with pytest.raises(SkillNotFound):
            await manager.get("nonexistent", "t1")

    @pytest.mark.asyncio
    async def test_get_returns_created_skill(self, manager):
        created = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        fetched = await manager.get(created.id, "t1")
        assert fetched.id == created.id
        assert fetched.name == "my-skill"

    @pytest.mark.asyncio
    async def test_get_tenant_isolation(self, manager):
        created = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        with pytest.raises(SkillNotFound):
            await manager.get(created.id, "t2")

    @pytest.mark.asyncio
    async def test_update_increments_version(self, manager):
        created = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        updated = await manager.update(created.id, "t1", change_note="update", description="new desc")
        assert updated.version == 2

    @pytest.mark.asyncio
    async def test_update_appends_version_history(self, manager):
        created = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        updated = await manager.update(created.id, "t1", change_note="v2 note", content="new content")
        assert len(updated.version_history) == 1
        assert updated.version_history[0].version == 1
        assert updated.version_history[0].change_note == "v2 note"

    @pytest.mark.asyncio
    async def test_update_preserves_old_content_in_history(self, manager):
        created = await manager.create("t1", "my-skill", "My Skill", "desc", "original content")
        await manager.update(created.id, "t1", change_note="changed", content="new content")
        updated = await manager.get(created.id, "t1")
        assert updated.version_history[0].content == "original content"
        assert updated.content == "new content"

    @pytest.mark.asyncio
    async def test_list_filters_active_only(self, manager):
        s1 = await manager.create("t1", "skill-a", "Skill A", "desc", "content")
        await manager.create("t1", "skill-b", "Skill B", "desc", "content")
        await manager.delete(s1.id, "t1")
        active = await manager.list("t1", active_only=True)
        assert len(active) == 1
        assert active[0].name == "skill-b"

    @pytest.mark.asyncio
    async def test_list_tenant_isolation(self, manager):
        await manager.create("t1", "skill-a", "Skill A", "desc", "content")
        await manager.create("t2", "skill-b", "Skill B", "desc", "content")
        t1_skills = await manager.list("t1")
        assert len(t1_skills) == 1
        assert t1_skills[0].name == "skill-a"

    @pytest.mark.asyncio
    async def test_delete_soft_deletes(self, manager):
        created = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        result = await manager.delete(created.id, "t1")
        assert result is True
        # Active-only list should not include it
        active = await manager.list("t1", active_only=True)
        assert len(active) == 0
        # Non-active list should still include it
        all_skills = await manager.list("t1", active_only=False)
        assert len(all_skills) == 1
        assert all_skills[0].active is False

    @pytest.mark.asyncio
    async def test_duplicate_new_id(self, manager):
        original = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        copy = await manager.duplicate(original.id, "t1")
        assert copy.id != original.id

    @pytest.mark.asyncio
    async def test_duplicate_display_name_has_copy(self, manager):
        original = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        copy = await manager.duplicate(original.id, "t1")
        assert "(copy)" in copy.display_name

    @pytest.mark.asyncio
    async def test_duplicate_same_content(self, manager):
        original = await manager.create("t1", "my-skill", "My Skill", "desc", "content body")
        copy = await manager.duplicate(original.id, "t1")
        assert copy.content == original.content

    @pytest.mark.asyncio
    async def test_export_json_returns_valid_json(self, manager):
        skill = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        exported = manager.export_json(skill)
        data = json.loads(exported)
        assert data["name"] == "my-skill"

    @pytest.mark.asyncio
    async def test_import_json_round_trip(self, manager):
        original = await manager.create("t1", "my-skill", "My Skill", "A test skill", "content")
        exported = manager.export_json(original)
        imported = await manager.import_json(exported, "t1")
        assert imported.name == original.name
        assert imported.content == original.content

    @pytest.mark.asyncio
    async def test_import_frontmatter(self, manager):
        md = "---\nname: my-skill\ndisplay_name: My Skill\ndescription: A test\n---\n# Content here"
        skill = await manager.import_json(md, "t1")
        assert skill.name == "my-skill"
        assert skill.display_name == "My Skill"
        assert "Content here" in skill.content

    @pytest.mark.asyncio
    async def test_semantic_match_substring_fallback(self, manager):
        await manager.create("t1", "search-tool", "Search Tool", "searches for data", "search content")
        await manager.create("t1", "writer-tool", "Writer Tool", "writes documents", "write content")
        results = await manager.semantic_match("search data", "t1")
        assert len(results) >= 1
        assert results[0].name == "search-tool"

    @pytest.mark.asyncio
    async def test_semantic_match_no_results(self, manager):
        await manager.create("t1", "search-tool", "Search Tool", "searches for data", "search content")
        results = await manager.semantic_match("zzzzz", "t1")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_record_invocation_increments_count(self, manager):
        skill = await manager.create("t1", "my-skill", "My Skill", "desc", "content")
        inv = await manager.record_invocation(skill.id, "t1", persona_name="researcher")
        assert inv.skill_id == skill.id
        updated = await manager.get(skill.id, "t1")
        assert updated.invocation_count == 1

    @pytest.mark.asyncio
    async def test_get_active_for_persona(self, manager):
        await manager.create("t1", "skill-a", "A", "desc", "c", allowed_personas=["researcher"])
        await manager.create("t1", "skill-b", "B", "desc", "c", allowed_personas=["developer"])
        await manager.create("t1", "skill-c", "C", "desc", "c", allowed_personas=[])
        results = await manager.get_active_for_persona("t1", "researcher")
        names = [s.name for s in results]
        assert "skill-a" in names
        assert "skill-c" in names  # empty allowed_personas = available to all
        assert "skill-b" not in names


# ═══════════════════════════════════════════════════════════════════════════════
# Route tests — build a minimal FastAPI app with mocked dependencies
# ═══════════════════════════════════════════════════════════════════════════════

def _build_test_app():
    """Build a minimal FastAPI app with the Phase 25 routes and auth bypass."""
    from fastapi import FastAPI, Request
    from starlette.middleware.base import BaseHTTPMiddleware

    app = FastAPI()

    # Bypass auth: inject tenant_id into request.state
    class FakeTenantMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request.state.tenant_id = "test-tenant"
            return await call_next(request)

    app.add_middleware(FakeTenantMiddleware)

    from nexus.api.routes import skills, credentials, mcp_servers, executions
    app.include_router(skills.router, prefix="/v2")
    app.include_router(credentials.router, prefix="/v2")
    app.include_router(mcp_servers.router, prefix="/v2")
    app.include_router(executions.router, prefix="/v2")

    return app


# ── Skills routes ──────────────────────────────────────────────────────────────

class TestSkillsRoutes:

    @pytest.fixture
    def app(self):
        app = _build_test_app()
        from nexus.skills.manager import SkillManager
        mgr = SkillManager(repository=None, embedding_service=None, config=NexusConfig())
        app.state.skill_manager = mgr
        return app

    @pytest.fixture
    def client(self, app):
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_list_skills_empty(self, client):
        resp = await client.get("/v2/skills")
        assert resp.status_code == 200
        data = resp.json()
        assert data["skills"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_create_skill_201(self, client):
        resp = await client.post("/v2/skills", json={
            "name": "my-skill",
            "display_name": "My Skill",
            "content": "some content",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "my-skill"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_create_invalid_name_422(self, client):
        resp = await client.post("/v2/skills", json={
            "name": "INVALID NAME",
            "display_name": "Test",
            "content": "c",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_skill(self, client):
        create_resp = await client.post("/v2/skills", json={
            "name": "my-skill", "display_name": "My Skill", "content": "c",
        })
        skill_id = create_resp.json()["id"]
        resp = await client.get(f"/v2/skills/{skill_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == skill_id

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self, client):
        resp = await client.get("/v2/skills/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_skill(self, client):
        create_resp = await client.post("/v2/skills", json={
            "name": "my-skill", "display_name": "My Skill", "content": "c",
        })
        skill_id = create_resp.json()["id"]
        resp = await client.patch(f"/v2/skills/{skill_id}", json={
            "change_note": "updated", "description": "new desc",
        })
        assert resp.status_code == 200
        assert resp.json()["version"] == 2

    @pytest.mark.asyncio
    async def test_delete_skill(self, client):
        create_resp = await client.post("/v2/skills", json={
            "name": "my-skill", "display_name": "My Skill", "content": "c",
        })
        skill_id = create_resp.json()["id"]
        resp = await client.delete(f"/v2/skills/{skill_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_duplicate_skill(self, client):
        create_resp = await client.post("/v2/skills", json={
            "name": "my-skill", "display_name": "My Skill", "content": "c",
        })
        skill_id = create_resp.json()["id"]
        resp = await client.post(f"/v2/skills/{skill_id}/duplicate")
        assert resp.status_code == 201
        assert "(copy)" in resp.json()["display_name"]

    @pytest.mark.asyncio
    async def test_list_skills_after_create(self, client):
        await client.post("/v2/skills", json={
            "name": "my-skill", "display_name": "My Skill", "content": "c",
        })
        resp = await client.get("/v2/skills")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    @pytest.mark.asyncio
    async def test_export_skill(self, client):
        create_resp = await client.post("/v2/skills", json={
            "name": "my-skill", "display_name": "My Skill", "content": "c",
        })
        skill_id = create_resp.json()["id"]
        resp = await client.get(f"/v2/skills/{skill_id}/export")
        assert resp.status_code == 200
        data = json.loads(resp.json()["data"])
        assert data["name"] == "my-skill"

    @pytest.mark.asyncio
    async def test_import_skill(self, client):
        payload = json.dumps({
            "name": "imported-skill",
            "display_name": "Imported",
            "content": "imported content",
        })
        resp = await client.post("/v2/skills/import", json={"data": payload})
        assert resp.status_code == 201
        assert resp.json()["name"] == "imported-skill"

    @pytest.mark.asyncio
    async def test_invocations_empty(self, client):
        create_resp = await client.post("/v2/skills", json={
            "name": "my-skill", "display_name": "My Skill", "content": "c",
        })
        skill_id = create_resp.json()["id"]
        resp = await client.get(f"/v2/skills/{skill_id}/invocations")
        assert resp.status_code == 200
        assert resp.json()["invocations"] == []

    @pytest.mark.asyncio
    async def test_diff_new_skill(self, client):
        create_resp = await client.post("/v2/skills", json={
            "name": "my-skill", "display_name": "My Skill", "content": "c",
        })
        skill_id = create_resp.json()["id"]
        resp = await client.get(f"/v2/skills/{skill_id}/diff?from_version=1")
        assert resp.status_code == 200
        assert resp.json()["skill_id"] == skill_id

    @pytest.mark.asyncio
    async def test_search_filter(self, client):
        await client.post("/v2/skills", json={
            "name": "alpha-skill", "display_name": "Alpha", "content": "c",
        })
        await client.post("/v2/skills", json={
            "name": "beta-skill", "display_name": "Beta", "content": "c",
        })
        resp = await client.get("/v2/skills?search=alpha")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


# ── Credentials routes ─────────────────────────────────────────────────────────

class TestCredentialsRoutes:

    @pytest.fixture
    def app(self):
        app = _build_test_app()
        from nexus.credentials.encryption import CredentialEncryption
        from nexus.credentials.vault import CredentialVault
        enc = CredentialEncryption()
        vault = CredentialVault(encryption=enc)
        app.state.vault = vault
        return app

    @pytest.fixture
    def client(self, app):
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_list_credentials_empty(self, client):
        resp = await client.get("/v2/credentials")
        assert resp.status_code == 200
        assert resp.json()["credentials"] == []

    @pytest.mark.asyncio
    async def test_create_credential(self, client):
        resp = await client.post("/v2/credentials", json={
            "name": "test-cred",
            "credential_type": "api_key",
            "service_name": "test-service",
            "data": {"api_key": "sk-test-1234567890"},
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test-cred"
        # encrypted_data should be empty in response
        assert data.get("encrypted_data", "") == ""

    @pytest.mark.asyncio
    async def test_create_credential_invalid_type(self, client):
        resp = await client.post("/v2/credentials", json={
            "name": "test-cred",
            "credential_type": "invalid_type",
            "service_name": "svc",
            "data": {"key": "val"},
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_list_credential_types(self, client):
        resp = await client.get("/v2/credentials/types")
        assert resp.status_code == 200
        types = resp.json()["types"]
        values = [t["value"] for t in types]
        assert "api_key" in values
        assert "oauth2" in values

    @pytest.mark.asyncio
    async def test_peek_credential(self, client):
        create_resp = await client.post("/v2/credentials", json={
            "name": "test-cred",
            "credential_type": "api_key",
            "service_name": "test-service",
            "data": {"api_key": "sk-test-1234567890"},
        })
        cred_id = create_resp.json()["id"]
        resp = await client.post(f"/v2/credentials/{cred_id}/peek")
        assert resp.status_code == 200
        data = resp.json()
        assert data["hint"].startswith("...")
        assert len(data["hint"]) == 7  # "..." + 4 chars

    @pytest.mark.asyncio
    async def test_peek_not_found(self, client):
        resp = await client.post("/v2/credentials/nonexistent/peek")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_credential(self, client):
        create_resp = await client.post("/v2/credentials", json={
            "name": "test-cred",
            "credential_type": "api_key",
            "service_name": "svc",
            "data": {"api_key": "val"},
        })
        cred_id = create_resp.json()["id"]
        resp = await client.delete(f"/v2/credentials/{cred_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_credential_not_found(self, client):
        resp = await client.delete("/v2/credentials/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_test_credential(self, client):
        resp = await client.post("/v2/credentials/test", json={
            "credential_type": "api_key",
            "service_name": "svc",
            "data": {"api_key": "val"},
        })
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    @pytest.mark.asyncio
    async def test_test_credential_invalid_type(self, client):
        resp = await client.post("/v2/credentials/test", json={
            "credential_type": "nope",
            "service_name": "svc",
            "data": {},
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rotate_credential(self, client):
        create_resp = await client.post("/v2/credentials", json={
            "name": "test-cred",
            "credential_type": "api_key",
            "service_name": "svc",
            "data": {"api_key": "old-key"},
        })
        cred_id = create_resp.json()["id"]
        resp = await client.post(f"/v2/credentials/{cred_id}/rotate", json={
            "data": {"api_key": "new-key"},
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_usage_count(self, client):
        create_resp = await client.post("/v2/credentials", json={
            "name": "test-cred",
            "credential_type": "api_key",
            "service_name": "svc",
            "data": {"api_key": "val"},
        })
        cred_id = create_resp.json()["id"]
        resp = await client.get(f"/v2/credentials/{cred_id}/usage")
        assert resp.status_code == 200
        assert resp.json()["usage_count"] == 0

    @pytest.mark.asyncio
    async def test_oauth_authorize_501(self, client):
        resp = await client.get("/v2/oauth/authorize")
        assert resp.status_code == 501

    @pytest.mark.asyncio
    async def test_oauth_callback_501(self, client):
        resp = await client.get("/v2/oauth/callback")
        assert resp.status_code == 501

    @pytest.mark.asyncio
    async def test_list_after_create(self, client):
        await client.post("/v2/credentials", json={
            "name": "test-cred",
            "credential_type": "api_key",
            "service_name": "svc",
            "data": {"api_key": "val"},
        })
        resp = await client.get("/v2/credentials")
        assert resp.status_code == 200
        assert len(resp.json()["credentials"]) == 1


# ── MCP routes ─────────────────────────────────────────────────────────────────

class TestMCPRoutes:

    @pytest.fixture
    def mock_adapter(self):
        adapter = MagicMock()
        adapter.list_servers.return_value = []
        adapter.register_server = AsyncMock(return_value=[])
        adapter.unregister_server = AsyncMock()
        adapter.get_server.return_value = None
        adapter._server_tools = {}
        return adapter

    @pytest.fixture
    def app(self, mock_adapter):
        app = _build_test_app()
        app.state.mcp_adapter = mock_adapter
        return app

    @pytest.fixture
    def client(self, app):
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_list_servers_empty(self, client):
        resp = await client.get("/v2/mcp/servers")
        assert resp.status_code == 200
        assert resp.json()["servers"] == []

    @pytest.mark.asyncio
    async def test_add_server(self, client, mock_adapter):
        resp = await client.post("/v2/mcp/servers", json={
            "name": "test-server",
            "transport": "stdio",
            "command": "echo",
        })
        assert resp.status_code == 201
        mock_adapter.register_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_server_not_found(self, client):
        resp = await client.delete("/v2/mcp/servers/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_server(self, client, mock_adapter):
        from nexus.types import MCPServerConfig
        server = MCPServerConfig(
            tenant_id="test-tenant", name="test", url="", transport="stdio", command="echo",
        )
        mock_adapter.get_server.return_value = server
        resp = await client.delete(f"/v2/mcp/servers/{server.id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_get_server_tools(self, client, mock_adapter):
        from nexus.types import MCPServerConfig
        server = MCPServerConfig(
            tenant_id="test-tenant", name="test", url="", transport="stdio", command="echo",
        )
        mock_adapter.get_server.return_value = server
        mock_adapter._server_tools[server.id] = ["tool-a", "tool-b"]
        resp = await client.get(f"/v2/mcp/servers/{server.id}/tools")
        assert resp.status_code == 200
        assert resp.json()["tools"] == ["tool-a", "tool-b"]

    @pytest.mark.asyncio
    async def test_get_server_tools_not_found(self, client):
        resp = await client.get("/v2/mcp/servers/nonexistent/tools")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_reconnect_server(self, client, mock_adapter):
        from nexus.types import MCPServerConfig
        server = MCPServerConfig(
            tenant_id="test-tenant", name="test", url="", transport="stdio", command="echo",
        )
        mock_adapter.get_server.return_value = server
        resp = await client.post(f"/v2/mcp/servers/{server.id}/reconnect")
        assert resp.status_code == 200
        assert resp.json()["reconnected"] is True

    @pytest.mark.asyncio
    async def test_reconnect_not_found(self, client):
        resp = await client.post("/v2/mcp/servers/nonexistent/reconnect")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_add_server_returns_tool_count(self, client, mock_adapter):
        mock_adapter.register_server = AsyncMock(return_value=["t1", "t2", "t3"])
        resp = await client.post("/v2/mcp/servers", json={
            "name": "test-server", "transport": "stdio", "command": "echo",
        })
        assert resp.status_code == 201
        assert resp.json()["tools_registered"] == 3

    @pytest.mark.asyncio
    async def test_list_servers_returns_mocked(self, mock_adapter):
        from nexus.types import MCPServerConfig
        s = MCPServerConfig(tenant_id="test-tenant", name="srv", url="", transport="stdio", command="echo")
        mock_adapter.list_servers.return_value = [s]
        app = _build_test_app()
        app.state.mcp_adapter = mock_adapter
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v2/mcp/servers")
        assert resp.status_code == 200
        assert len(resp.json()["servers"]) == 1


# ── Executions routes ──────────────────────────────────────────────────────────

class TestExecutionsRoutes:

    @pytest.fixture
    def app(self):
        app = _build_test_app()
        return app

    @pytest.fixture
    def client(self, app):
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_list_executions_no_ledger(self, client):
        resp = await client.get("/v2/executions")
        assert resp.status_code == 200
        assert resp.json()["executions"] == []

    @pytest.mark.asyncio
    async def test_get_execution_not_found(self, client):
        resp = await client.get("/v2/executions/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_add_pin(self, client):
        resp = await client.post("/v2/executions/exec-1/pins", json={
            "step_id": "step-1",
            "output_data": {"result": "hello"},
        })
        assert resp.status_code == 200
        assert resp.json()["pinned"] is True

    @pytest.mark.asyncio
    async def test_get_pins(self, client):
        await client.post("/v2/executions/exec-1/pins", json={
            "step_id": "step-1",
            "output_data": {"result": "hello"},
        })
        resp = await client.get("/v2/executions/exec-1/pins")
        assert resp.status_code == 200
        pins = resp.json()["pins"]
        assert "step-1" in pins

    @pytest.mark.asyncio
    async def test_remove_pin(self, client):
        await client.post("/v2/executions/exec-2/pins", json={
            "step_id": "step-1",
            "output_data": {"result": "hello"},
        })
        resp = await client.delete("/v2/executions/exec-2/pins/step-1")
        assert resp.status_code == 200
        assert resp.json()["unpinned"] is True
        # Verify pin was removed
        get_resp = await client.get("/v2/executions/exec-2/pins")
        assert "step-1" not in get_resp.json()["pins"]

    @pytest.mark.asyncio
    async def test_get_pins_empty(self, client):
        resp = await client.get("/v2/executions/nonexistent/pins")
        assert resp.status_code == 200
        assert resp.json()["pins"] == {}

    @pytest.mark.asyncio
    async def test_retry_no_engine_503(self, client):
        resp = await client.post("/v2/executions/exec-1/retry")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_multiple_pins(self, client):
        await client.post("/v2/executions/exec-3/pins", json={
            "step_id": "step-1", "output_data": {"a": 1},
        })
        await client.post("/v2/executions/exec-3/pins", json={
            "step_id": "step-2", "output_data": {"b": 2},
        })
        resp = await client.get("/v2/executions/exec-3/pins")
        pins = resp.json()["pins"]
        assert "step-1" in pins
        assert "step-2" in pins

    @pytest.mark.asyncio
    async def test_list_with_ledger(self, client, app):
        mock_ledger = MagicMock()
        mock_ledger.list_chains.return_value = []
        app.state.ledger = mock_ledger
        resp = await client.get("/v2/executions")
        assert resp.status_code == 200
        assert resp.json()["executions"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# WorkflowGenerator skill context — unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkflowGeneratorSkillContext:

    @pytest.mark.asyncio
    async def test_load_skill_context_no_manager(self):
        from nexus.workflows.generator import WorkflowGenerator
        gen = WorkflowGenerator(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            persona_manager=MagicMock(),
            workflow_manager=MagicMock(),
            config=NexusConfig(),
            skill_manager=None,
        )
        result = await gen._load_skill_context("task", "researcher", "t1")
        assert result == []

    @pytest.mark.asyncio
    async def test_load_skill_context_with_manager(self):
        from nexus.workflows.generator import WorkflowGenerator
        mock_mgr = AsyncMock()
        skill = make_skill()
        mock_mgr.semantic_match = AsyncMock(return_value=[skill])
        gen = WorkflowGenerator(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            persona_manager=MagicMock(),
            workflow_manager=MagicMock(),
            config=NexusConfig(),
            skill_manager=mock_mgr,
        )
        result = await gen._load_skill_context("search query", "researcher", "t1")
        assert len(result) == 1
        assert result[0].name == "my-skill"

    @pytest.mark.asyncio
    async def test_load_skill_context_error_returns_empty(self):
        from nexus.workflows.generator import WorkflowGenerator
        mock_mgr = AsyncMock()
        mock_mgr.semantic_match = AsyncMock(side_effect=Exception("boom"))
        gen = WorkflowGenerator(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            persona_manager=MagicMock(),
            workflow_manager=MagicMock(),
            config=NexusConfig(),
            skill_manager=mock_mgr,
        )
        result = await gen._load_skill_context("q", "r", "t1")
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_includes_skill_context(self):
        """Verify generate() includes skills in the prompt when available."""
        from nexus.workflows.generator import WorkflowGenerator

        skill = make_skill(name="test-skill", description="A test skill for searching")

        mock_skill_mgr = AsyncMock()
        mock_skill_mgr.semantic_match = AsyncMock(return_value=[skill])

        mock_tool_reg = MagicMock()
        mock_tool_reg.list_tools.return_value = []

        mock_persona_mgr = MagicMock()
        mock_persona_mgr.list_personas.return_value = []

        mock_wf_mgr = MagicMock()

        mock_llm = MagicMock()
        # Return valid workflow JSON from the LLM
        valid_json = json.dumps({
            "name": "test-wf",
            "description": "test",
            "steps": [{
                "id": "s1", "workflow_id": "", "step_type": "action",
                "name": "Do", "tool_name": "knowledge_search", "persona_name": "researcher",
            }],
            "edges": [],
            "trigger": {"type": "manual", "config": {}},
        })
        mock_llm.complete = AsyncMock(return_value={"content": valid_json})

        gen = WorkflowGenerator(
            llm_client=mock_llm,
            tool_registry=mock_tool_reg,
            persona_manager=mock_persona_mgr,
            workflow_manager=mock_wf_mgr,
            config=NexusConfig(),
            skill_manager=mock_skill_mgr,
        )
        mock_wf_mgr.create = AsyncMock(side_effect=lambda **kw: MagicMock(id="wf-1"))

        await gen.generate("search for something", "t1")
        # The LLM was called with messages that contain skill context
        mock_llm.complete.assert_called_once()
        call_args = mock_llm.complete.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1].get("messages", [])
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert "AVAILABLE SKILLS" in user_msg or "test-skill" in user_msg
