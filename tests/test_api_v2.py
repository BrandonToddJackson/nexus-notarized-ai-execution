"""Phase 30: API v2 integration tests.

Tests exercise the v2 route handlers with minimal FastAPI test apps,
mocked state dependencies, and no database or external services.
"""

import asyncio
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from cryptography.fernet import Fernet

from nexus.types import TriggerConfig, TriggerType, MCPServerConfig, CredentialType

TENANT_A = "tenant-alpha-001"
TENANT_B = "tenant-beta-002"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_test_app(router, state_attrs=None, tenant_id=TENANT_A):
    """Create minimal test app with given router and injected tenant."""
    app = FastAPI()
    app.include_router(router)

    if state_attrs:
        for key, value in state_attrs.items():
            setattr(app.state, key, value)

    @app.middleware("http")
    async def inject_tenant(request, call_next):
        request.state.tenant_id = tenant_id
        return await call_next(request)

    return app


# ── Workflow CRUD ─────────────────────────────────────────────────────────────

class TestWorkflowCRUD:
    """Tests for /v2/workflows CRUD endpoints."""

    def setup_method(self):
        from nexus.api.routes import workflows as wf_routes
        from nexus.workflows.manager import WorkflowManager

        self.manager = WorkflowManager()
        app = make_test_app(
            wf_routes.router,
            {"workflow_manager": self.manager, "workflow_generator": None},
        )
        self.client = TestClient(app)

    def test_list_workflows_empty(self):
        resp = self.client.get("/v2/workflows")
        assert resp.status_code == 200
        data = resp.json()
        assert data["workflows"] == []

    def test_create_workflow(self):
        resp = self.client.post("/v2/workflows", json={
            "name": "Test Workflow",
            "description": "A test workflow",
            "steps": [],
            "edges": [],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Workflow"
        assert data["tenant_id"] == TENANT_A
        assert "id" in data

    def test_get_workflow(self):
        create = self.client.post("/v2/workflows", json={"name": "Fetch Me"})
        wf_id = create.json()["id"]

        resp = self.client.get(f"/v2/workflows/{wf_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Fetch Me"

    def test_get_workflow_not_found(self):
        resp = self.client.get("/v2/workflows/nonexistent")
        assert resp.status_code == 404

    def test_update_workflow(self):
        create = self.client.post("/v2/workflows", json={"name": "Original"})
        wf_id = create.json()["id"]

        resp = self.client.put(f"/v2/workflows/{wf_id}", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    def test_activate_workflow(self):
        create = self.client.post("/v2/workflows", json={"name": "Activate Me"})
        wf_id = create.json()["id"]

        resp = self.client.post(f"/v2/workflows/{wf_id}/activate")
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_pause_workflow(self):
        create = self.client.post("/v2/workflows", json={"name": "Pause Me"})
        wf_id = create.json()["id"]
        # Must activate first
        self.client.post(f"/v2/workflows/{wf_id}/activate")

        resp = self.client.post(f"/v2/workflows/{wf_id}/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    def test_status_patch(self):
        create = self.client.post("/v2/workflows", json={"name": "Status Test"})
        wf_id = create.json()["id"]

        resp = self.client.patch(f"/v2/workflows/{wf_id}/status", json={"status": "active"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_status_patch_invalid(self):
        create = self.client.post("/v2/workflows", json={"name": "Bad Status"})
        wf_id = create.json()["id"]

        resp = self.client.patch(f"/v2/workflows/{wf_id}/status", json={"status": "bogus"})
        assert resp.status_code == 400

    def test_get_version_history(self):
        create = self.client.post("/v2/workflows", json={"name": "Versioned"})
        wf_id = create.json()["id"]

        resp = self.client.get(f"/v2/workflows/{wf_id}/versions")
        assert resp.status_code == 200
        versions = resp.json()["versions"]
        # A newly created workflow has at least one version record (v1)
        assert len(versions) >= 1
        # The version records must contain a version number field
        assert all("version" in v for v in versions)
        assert versions[0]["version"] == 1

    def test_duplicate_workflow(self):
        create = self.client.post("/v2/workflows", json={"name": "Duplicatable"})
        wf_id = create.json()["id"]

        resp = self.client.post(f"/v2/workflows/{wf_id}/duplicate")
        assert resp.status_code == 201
        dup = resp.json()
        assert dup["id"] != wf_id
        assert dup["name"] == "Duplicatable (copy)"

    def test_export_import(self):
        create = self.client.post("/v2/workflows", json={
            "name": "Export Test",
            "description": "For export",
        })
        wf_id = create.json()["id"]

        export_resp = self.client.get(f"/v2/workflows/{wf_id}/export")
        assert export_resp.status_code == 200
        exported = export_resp.json()["data"]

        import_resp = self.client.post("/v2/workflows/import", json={"data": exported})
        assert import_resp.status_code == 201
        imported = import_resp.json()
        assert imported["name"] == "Export Test"
        # Imported workflow gets a new ID
        assert imported["id"] != wf_id

    def test_import_missing_data(self):
        resp = self.client.post("/v2/workflows/import", json={})
        assert resp.status_code == 422

    def test_list_templates(self):
        # GET /v2/workflows/templates is registered before the parameterized
        # /{workflow_id} route, so it returns the templates list (200).
        resp = self.client.get("/v2/workflows/templates")
        assert resp.status_code == 200
        assert "templates" in resp.json()


# ── Trigger CRUD ──────────────────────────────────────────────────────────────

class TestTriggerCRUD:
    """Tests for /v2/triggers endpoints."""

    def _make_trigger_config(self, **overrides):
        defaults = {
            "id": "trig-001",
            "workflow_id": "wf-001",
            "tenant_id": TENANT_A,
            "trigger_type": TriggerType.WEBHOOK,
            "enabled": True,
            "config": {},
            "webhook_path": "/webhooks/tenant-alpha-001/abc",
        }
        defaults.update(overrides)
        return TriggerConfig(**defaults)

    def setup_method(self):
        from nexus.api.routes import triggers as trig_routes

        self.mock_manager = MagicMock()
        self.mock_manager.list = AsyncMock(return_value=[])
        self.mock_manager.create_trigger = AsyncMock(
            return_value=self._make_trigger_config()
        )
        self.mock_manager.enable = AsyncMock(
            return_value=self._make_trigger_config(enabled=True)
        )
        self.mock_manager.disable = AsyncMock(
            return_value=self._make_trigger_config(enabled=False)
        )
        self.mock_manager.delete = AsyncMock(return_value=None)

        app = make_test_app(
            trig_routes.router,
            {"trigger_manager": self.mock_manager},
        )
        self.client = TestClient(app)

    def test_list_triggers_empty(self):
        resp = self.client.get("/triggers")
        assert resp.status_code == 200
        assert resp.json()["triggers"] == []

    def test_create_webhook_trigger(self):
        resp = self.client.post("/triggers", json={
            "workflow_id": "wf-001",
            "type": "webhook",
            "config": {},
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["trigger_type"] == "webhook"
        assert "webhook_url" in data

    def test_create_cron_trigger(self):
        self.mock_manager.create_trigger = AsyncMock(
            return_value=self._make_trigger_config(
                trigger_type=TriggerType.CRON,
                webhook_path=None,
                config={"cron_expression": "0 * * * *"},
            )
        )
        resp = self.client.post("/triggers", json={
            "workflow_id": "wf-001",
            "type": "cron",
            "config": {"cron_expression": "0 * * * *"},
        })
        assert resp.status_code == 201
        assert resp.json()["trigger_type"] == "cron"

    def test_create_trigger_invalid_type(self):
        resp = self.client.post("/triggers", json={
            "workflow_id": "wf-001",
            "type": "invalid_type",
            "config": {},
        })
        assert resp.status_code == 422

    def test_get_trigger(self):
        tc = self._make_trigger_config()
        self.mock_manager.list = AsyncMock(return_value=[tc])

        resp = self.client.get(f"/triggers/{tc.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == tc.id

    def test_get_trigger_not_found(self):
        resp = self.client.get("/triggers/nonexistent")
        assert resp.status_code == 404

    def test_enable_trigger(self):
        resp = self.client.post("/triggers/trig-001/enable")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is True

    def test_disable_trigger(self):
        resp = self.client.post("/triggers/trig-001/disable")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_delete_trigger(self):
        resp = self.client.delete("/triggers/trig-001")
        assert resp.status_code == 204

    def test_trigger_tenant_isolation(self):
        """Triggers list is filtered by tenant via manager.list(tenant_id)."""
        from nexus.api.routes import triggers as trig_routes

        mock_mgr = MagicMock()
        mock_mgr.list = AsyncMock(return_value=[])

        app_b = make_test_app(
            trig_routes.router,
            {"trigger_manager": mock_mgr},
            tenant_id=TENANT_B,
        )
        client_b = TestClient(app_b)
        client_b.get("/triggers")
        mock_mgr.list.assert_called_once()
        call_args = mock_mgr.list.call_args
        assert call_args[0][0] == TENANT_B or call_args[1].get("tenant_id") == TENANT_B


# ── Webhook Receiver ──────────────────────────────────────────────────────────

class TestWebhookReceiver:
    """Tests for /v2/webhooks/{path} catch-all endpoint."""

    def setup_method(self):
        from nexus.api.routes import webhooks as wh_routes

        self.mock_handler = MagicMock()
        self.mock_handler.handle = AsyncMock(return_value={"status": "accepted"})

        app = make_test_app(wh_routes.router, {"webhook_handler": self.mock_handler})
        self.client = TestClient(app)

    def test_webhook_fires_post(self):
        resp = self.client.post(
            "/v2/webhooks/tenant-alpha-001/abc",
            json={"event": "push"},
        )
        assert resp.status_code == 200
        self.mock_handler.handle.assert_called_once()
        call_kw = self.mock_handler.handle.call_args[1]
        assert call_kw["webhook_path"] == "/webhooks/tenant-alpha-001/abc"
        assert call_kw["method"] == "POST"

    def test_webhook_get_method(self):
        resp = self.client.get("/v2/webhooks/tenant-alpha-001/abc")
        assert resp.status_code == 200
        call_kw = self.mock_handler.handle.call_args[1]
        assert call_kw["method"] == "GET"

    def test_webhook_put_method(self):
        resp = self.client.put(
            "/v2/webhooks/test/path",
            json={"data": "value"},
        )
        assert resp.status_code == 200
        call_kw = self.mock_handler.handle.call_args[1]
        assert call_kw["method"] == "PUT"
        assert call_kw["webhook_path"] == "/webhooks/test/path"

    def test_webhook_no_handler_503(self):
        from nexus.api.routes import webhooks as wh_routes

        app = make_test_app(wh_routes.router, {})
        client = TestClient(app)
        resp = client.post("/v2/webhooks/any/path", json={})
        assert resp.status_code == 503

    def test_webhook_trigger_not_found(self):
        from nexus.exceptions import TriggerError

        self.mock_handler.handle = AsyncMock(
            side_effect=TriggerError("unknown webhook path")
        )
        resp = self.client.post("/v2/webhooks/unknown/path", json={})
        assert resp.status_code == 404


# ── Credential CRUD ───────────────────────────────────────────────────────────

class TestCredentialCRUD:
    """Tests for /v2/credentials endpoints."""

    def setup_method(self):
        from nexus.api.routes import credentials as cred_routes
        from nexus.credentials.vault import CredentialVault
        from nexus.credentials.encryption import CredentialEncryption

        enc = CredentialEncryption(key=Fernet.generate_key().decode())
        self.vault = CredentialVault(encryption=enc)

        app = make_test_app(cred_routes.router, {"vault": self.vault})
        self.client = TestClient(app)

    def _create_credential(self, name="GitHub PAT"):
        return self.client.post("/credentials", json={
            "name": name,
            "credential_type": "api_key",
            "service_name": "github",
            "data": {"token": "ghp_test1234567890abcdef"},
        })

    def test_create_credential(self):
        resp = self._create_credential()
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "GitHub PAT"
        assert data["service_name"] == "github"
        # Encrypted data should not contain the raw token
        assert "ghp_test1234567890abcdef" not in data.get("encrypted_data", "")

    def test_list_credentials(self):
        self._create_credential("Cred A")
        self._create_credential("Cred B")

        resp = self.client.get("/credentials")
        assert resp.status_code == 200
        creds = resp.json()["credentials"]
        assert len(creds) == 2

    def test_delete_credential(self):
        create = self._create_credential()
        cred_id = create.json()["id"]

        resp = self.client.delete(f"/credentials/{cred_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # Verify deleted
        list_resp = self.client.get("/credentials")
        assert len(list_resp.json()["credentials"]) == 0

    def test_delete_credential_not_found(self):
        resp = self.client.delete("/credentials/nonexistent")
        assert resp.status_code == 404

    def test_peek_credential(self):
        create = self._create_credential()
        cred_id = create.json()["id"]

        resp = self.client.post(f"/credentials/{cred_id}/peek")
        assert resp.status_code == 200
        data = resp.json()
        assert data["credential_id"] == cred_id
        # Hint should be last 4 chars
        assert data["hint"].startswith("...")
        assert len(data["hint"]) == 7  # "..." + 4 chars

    def test_peek_credential_not_found(self):
        resp = self.client.post("/credentials/nonexistent/peek")
        assert resp.status_code == 404

    def test_create_credential_invalid_type(self):
        resp = self.client.post("/credentials", json={
            "name": "Bad Type",
            "credential_type": "invalid",
            "service_name": "test",
            "data": {"key": "value"},
        })
        assert resp.status_code == 422

    def test_list_credential_types(self):
        resp = self.client.get("/credentials/types")
        assert resp.status_code == 200
        types = resp.json()["types"]
        values = [t["value"] for t in types]
        assert "api_key" in values
        assert "oauth2" in values

    def test_credential_tenant_isolation(self):
        """Credentials stored under TENANT_A are not visible to TENANT_B."""
        from nexus.api.routes import credentials as cred_routes

        # Store under TENANT_A
        self._create_credential("Isolated Cred")

        # Create app for TENANT_B using same vault
        app_b = make_test_app(cred_routes.router, {"vault": self.vault}, tenant_id=TENANT_B)
        client_b = TestClient(app_b)

        resp = client_b.get("/credentials")
        assert resp.status_code == 200
        assert len(resp.json()["credentials"]) == 0


# ── MCP Server CRUD ───────────────────────────────────────────────────────────

class TestMCPServerCRUD:
    """Tests for /v2/mcp/servers endpoints."""

    def _make_server_config(self, **overrides):
        defaults = {
            "tenant_id": TENANT_A,
            "name": "test-server",
            "url": "http://localhost:3000",
            "transport": "sse",
        }
        defaults.update(overrides)
        return MCPServerConfig(**defaults)

    def setup_method(self):
        from nexus.api.routes import mcp_servers as mcp_routes

        self.server = self._make_server_config()

        self.mock_adapter = MagicMock()
        self.mock_adapter.list_servers = MagicMock(return_value=[self.server])
        self.mock_adapter.register_server = AsyncMock(return_value=["tool_a", "tool_b"])
        self.mock_adapter.unregister_server = AsyncMock(return_value=None)
        self.mock_adapter.get_server = MagicMock(return_value=self.server)
        self.mock_adapter._server_tools = {self.server.id: ["tool_a", "tool_b"]}

        app = make_test_app(mcp_routes.router, {"mcp_adapter": self.mock_adapter})
        self.client = TestClient(app)

    def test_list_servers(self):
        resp = self.client.get("/mcp/servers")
        assert resp.status_code == 200
        servers = resp.json()["servers"]
        assert len(servers) == 1
        assert servers[0]["name"] == "test-server"

    def test_register_server(self):
        resp = self.client.post("/mcp/servers", json={
            "name": "new-server",
            "url": "http://localhost:4000",
            "transport": "sse",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["tools_registered"] == 2
        assert data["server"]["name"] == "new-server"

    def test_delete_server(self):
        resp = self.client.delete(f"/mcp/servers/{self.server.id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        self.mock_adapter.unregister_server.assert_called_once_with(self.server.id)

    def test_delete_server_not_found(self):
        self.mock_adapter.get_server = MagicMock(return_value=None)
        resp = self.client.delete("/mcp/servers/nonexistent")
        assert resp.status_code == 404

    def test_refresh_server(self):
        resp = self.client.post(f"/mcp/servers/{self.server.id}/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["refreshed"] is True
        assert data["tools_registered"] == 2

    def test_refresh_server_not_found(self):
        self.mock_adapter.get_server = MagicMock(return_value=None)
        resp = self.client.post("/mcp/servers/nonexistent/refresh")
        assert resp.status_code == 404

    def test_list_server_tools(self):
        resp = self.client.get(f"/mcp/servers/{self.server.id}/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tools"] == ["tool_a", "tool_b"]

    def test_reconnect_server(self):
        resp = self.client.post(f"/mcp/servers/{self.server.id}/reconnect")
        assert resp.status_code == 200
        assert resp.json()["reconnected"] is True

    def test_server_tenant_isolation(self):
        """Servers belonging to another tenant are 404 on delete."""
        other_server = self._make_server_config(tenant_id=TENANT_B)
        self.mock_adapter.get_server = MagicMock(return_value=other_server)

        resp = self.client.delete(f"/mcp/servers/{other_server.id}")
        assert resp.status_code == 404


# ── Job Status ────────────────────────────────────────────────────────────────

class TestJobStatus:
    """Tests for /v2/jobs/{job_id} endpoint."""

    def setup_method(self):
        from nexus.api.routes import jobs as job_routes

        self.mock_dispatcher = MagicMock()
        self.mock_dispatcher.get_job_status = AsyncMock(
            return_value={"status": "queued", "job_id": "job-001"}
        )

        app = make_test_app(job_routes.router, {"dispatcher": self.mock_dispatcher})
        self.client = TestClient(app)

    def test_job_pending(self):
        resp = self.client.get("/job-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["job_id"] == "job-001"

    def test_job_complete(self):
        self.mock_dispatcher.get_job_status = AsyncMock(
            return_value={"status": "complete", "job_id": "job-002", "result": {"ok": True}}
        )
        resp = self.client.get("/job-002")
        assert resp.status_code == 200
        assert resp.json()["status"] == "complete"

    def test_job_result(self):
        self.mock_dispatcher.get_job_result = AsyncMock(
            return_value={"status": "complete", "job_id": "job-003", "output": "done"}
        )
        resp = self.client.get("/job-003/result")
        assert resp.status_code == 200
        assert resp.json()["output"] == "done"

    def test_job_result_fallback_to_status(self):
        """If dispatcher has no get_job_result, falls back to get_job_status."""
        del self.mock_dispatcher.get_job_result
        resp = self.client.get("/job-004/result")
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"


# ── Execution Endpoints ───────────────────────────────────────────────────────

class TestExecutionEndpoints:
    """Tests for /v2/executions endpoints."""

    def setup_method(self):
        from nexus.api.routes import executions as exec_routes
        # Reset in-memory pins between tests
        exec_routes._pins.clear()

        app = make_test_app(exec_routes.router, {"ledger": None})
        self.client = TestClient(app)

    def test_list_executions_no_ledger(self):
        resp = self.client.get("/executions")
        assert resp.status_code == 200
        assert resp.json()["executions"] == []

    def test_pin_and_get_pins(self):
        # Add a pin
        resp = self.client.post("/executions/exec-001/pins", json={
            "step_id": "step-a",
            "output_data": {"value": 42},
        })
        assert resp.status_code == 200
        assert resp.json()["pinned"] is True

        # Get pins
        resp = self.client.get("/executions/exec-001/pins")
        assert resp.status_code == 200
        pins = resp.json()["pins"]
        assert pins["step-a"]["value"] == 42

    def test_remove_pin(self):
        self.client.post("/executions/exec-001/pins", json={
            "step_id": "step-a",
            "output_data": {"value": 42},
        })

        resp = self.client.delete("/executions/exec-001/pins/step-a")
        assert resp.status_code == 200
        assert resp.json()["unpinned"] is True

        # Verify removed
        resp = self.client.get("/executions/exec-001/pins")
        assert resp.json()["pins"] == {}

    def test_delete_execution_no_ledger(self):
        resp = self.client.delete("/executions/exec-001")
        assert resp.status_code == 404


# ── Workflow Run ──────────────────────────────────────────────────────────────

class TestWorkflowRun:
    """Tests for POST /v2/workflows/{id}/run dispatching."""

    def setup_method(self):
        from nexus.api.routes import workflows as wf_routes
        from nexus.workflows.manager import WorkflowManager

        self.manager = WorkflowManager()
        self.mock_dispatcher = MagicMock()
        self.mock_dispatcher.dispatch = AsyncMock(
            return_value={"job_id": "job-123", "status": "queued"}
        )

        app = make_test_app(
            wf_routes.router,
            {
                "workflow_manager": self.manager,
                "workflow_generator": None,
                "dispatcher": self.mock_dispatcher,
            },
        )
        self.client = TestClient(app)

    def test_run_workflow(self):
        create = self.client.post("/v2/workflows", json={"name": "Runnable"})
        wf_id = create.json()["id"]

        resp = self.client.post(f"/v2/workflows/{wf_id}/run", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-123"

    def test_run_workflow_no_dispatcher(self):
        from nexus.api.routes import workflows as wf_routes
        from nexus.workflows.manager import WorkflowManager

        manager = WorkflowManager()
        app = make_test_app(
            wf_routes.router,
            {"workflow_manager": manager, "workflow_generator": None},
        )
        client = TestClient(app)

        create = client.post("/v2/workflows", json={"name": "No Dispatcher"})
        wf_id = create.json()["id"]

        resp = client.post(f"/v2/workflows/{wf_id}/run", json={})
        assert resp.status_code == 503


# ── Generate Workflow ─────────────────────────────────────────────────────────

class TestWorkflowGenerate:
    """Tests for POST /v2/workflows/generate."""

    def setup_method(self):
        from nexus.api.routes import workflows as wf_routes
        from nexus.workflows.manager import WorkflowManager
        from nexus.types import WorkflowDefinition

        self.mock_generator = MagicMock()
        mock_wf = WorkflowDefinition(
            id="gen-wf-001",
            tenant_id=TENANT_A,
            name="Generated Workflow",
        )
        self.mock_generator.generate = AsyncMock(return_value=mock_wf)

        app = make_test_app(
            wf_routes.router,
            {
                "workflow_manager": WorkflowManager(),
                "workflow_generator": self.mock_generator,
            },
        )
        self.client = TestClient(app)

    def test_generate_workflow(self):
        resp = self.client.post("/v2/workflows/generate", json={
            "description": "Create a workflow that fetches data from an API and stores it",
        })
        assert resp.status_code == 200
        assert resp.json()["name"] == "Generated Workflow"

    def test_generate_workflow_description_too_short(self):
        resp = self.client.post("/v2/workflows/generate", json={
            "description": "short",
        })
        assert resp.status_code == 422


# ── Auth Middleware (real JWT path) ───────────────────────────────────────────

class TestAuthMiddleware:
    """Tests for AuthMiddleware JWT verification and PUBLIC_PREFIXES bypass.

    These use the real AuthMiddleware with a real JWT — no inject_tenant shortcut.
    """

    def setup_method(self):
        from nexus.auth.middleware import AuthMiddleware
        from nexus.auth.jwt import JWTManager

        self.jwt_manager = JWTManager()
        # Create a valid token synchronously for use in tests
        self.valid_token = asyncio.run(self.jwt_manager.create_token(TENANT_A))

        protected_app = FastAPI()

        @protected_app.get("/protected")
        async def protected(request: Request):
            return {"tenant_id": request.state.tenant_id}

        @protected_app.get("/v2/webhooks/path/test")
        async def webhook_endpoint(request: Request):
            # tenant_id may not be set — that's fine for public paths
            return {"received": True}

        protected_app.add_middleware(AuthMiddleware)
        self.client = TestClient(protected_app, raise_server_exceptions=False)

    def test_missing_auth_returns_401(self):
        """Request with no Authorization header is rejected."""
        resp = self.client.get("/protected")
        assert resp.status_code == 401
        assert "Missing" in resp.json()["detail"]

    def test_invalid_jwt_returns_401(self):
        """Request with a garbage Bearer token is rejected."""
        resp = self.client.get("/protected", headers={"Authorization": "Bearer not.a.real.token"})
        assert resp.status_code == 401

    def test_valid_jwt_sets_tenant_id(self):
        """Valid JWT is decoded and tenant_id injected into request.state."""
        resp = self.client.get(
            "/protected",
            headers={"Authorization": f"Bearer {self.valid_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == TENANT_A

    def test_webhook_path_bypasses_auth(self):
        """Paths matching PUBLIC_PREFIXES ('/v2/webhooks') skip JWT entirely."""
        resp = self.client.get("/v2/webhooks/path/test")  # no auth header
        assert resp.status_code == 200
        assert resp.json()["received"] is True
