"""Phase 29 — API v2 Complete Engineering

Tests for:
- Trigger endpoints (7): list, create, get, update, enable, disable, delete
- Webhook catch-all (3 scenarios): 200, 503, no-auth
- Marketplace endpoints (4): search, installed, install, uninstall
- New workflow endpoints (6): activate, pause, versions, rollback, refine, explain
- Execution: delete, stream
- Job: result endpoint
- MCP: refresh endpoint
- Auth middleware: webhook path bypass
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with all Phase 29 routes — no lifespan."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # Auth middleware with a bypass that sets tenant_id = "test-tenant"
    from starlette.middleware.base import BaseHTTPMiddleware
    class FakeAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # Allow webhooks without auth
            if request.url.path.startswith("/v2/webhooks"):
                return await call_next(request)
            request.state.tenant_id = "test-tenant"
            return await call_next(request)
    app.add_middleware(FakeAuthMiddleware)

    from nexus.api.routes import triggers, webhooks, marketplace
    from nexus.api.routes import workflows, executions, mcp_servers
    from nexus.api.routes.jobs import router as jobs_router

    app.include_router(workflows.router)
    app.include_router(executions.router, prefix="/v2")
    app.include_router(mcp_servers.router, prefix="/v2")
    app.include_router(jobs_router, prefix="/v2/jobs", tags=["jobs"])
    app.include_router(triggers.router, prefix="/v2")
    app.include_router(webhooks.router)
    app.include_router(marketplace.router, prefix="/v2")

    return app


def _make_trigger(trigger_id: str = "trig-1", workflow_id: str = "wf-1") -> MagicMock:
    t = MagicMock()
    t.id = trigger_id
    t.workflow_id = workflow_id
    t.tenant_id = "test-tenant"
    t.enabled = True
    t.webhook_path = None
    t.trigger_type = MagicMock()
    t.trigger_type.value = "cron"
    t.model_dump.return_value = {
        "id": trigger_id,
        "workflow_id": workflow_id,
        "tenant_id": "test-tenant",
        "enabled": True,
        "trigger_type": "cron",
        "config": {},
    }
    return t


def _make_workflow(workflow_id: str = "wf-1") -> MagicMock:
    w = MagicMock()
    w.id = workflow_id
    w.model_dump.return_value = {
        "id": workflow_id,
        "name": "Test Workflow",
        "status": "active",
        "version": 1,
    }
    return w


def _make_plugin_manifest(name: str = "test-plugin") -> MagicMock:
    m = MagicMock()
    m.name = name
    m.version = "1.0.0"
    m.model_dump.return_value = {"name": name, "version": "1.0.0"}
    return m


# ─────────────────────────────────────────────────────────────────────────────
# ── TRIGGER TESTS ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestListTriggers:
    def setup_method(self):
        self.app = _make_app()
        self.trigger = _make_trigger()
        mgr = MagicMock()
        mgr.list = AsyncMock(return_value=[self.trigger])
        self.app.state.trigger_manager = mgr

    def test_list_returns_triggers(self):
        with TestClient(self.app) as client:
            resp = client.get("/v2/triggers")
        assert resp.status_code == 200
        data = resp.json()
        assert "triggers" in data
        assert len(data["triggers"]) == 1
        assert data["triggers"][0]["id"] == "trig-1"

    def test_list_filter_by_workflow_id(self):
        with TestClient(self.app) as client:
            resp = client.get("/v2/triggers?workflow_id=wf-1")
        assert resp.status_code == 200

    def test_list_filter_by_enabled(self):
        with TestClient(self.app) as client:
            resp = client.get("/v2/triggers?enabled=true")
        assert resp.status_code == 200

    def test_list_no_manager_returns_503(self):
        app = _make_app()
        # no trigger_manager in state
        with TestClient(app) as client:
            resp = client.get("/v2/triggers")
        assert resp.status_code == 503

    def test_list_empty(self):
        mgr = MagicMock()
        mgr.list = AsyncMock(return_value=[])
        self.app.state.trigger_manager = mgr
        with TestClient(self.app) as client:
            resp = client.get("/v2/triggers")
        assert resp.status_code == 200
        assert resp.json()["triggers"] == []


class TestCreateTrigger:
    def setup_method(self):
        self.app = _make_app()
        self.trigger = _make_trigger()
        mgr = MagicMock()
        mgr.create_trigger = AsyncMock(return_value=self.trigger)
        self.app.state.trigger_manager = mgr

    def test_create_cron_trigger(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/triggers", json={
                "workflow_id": "wf-1",
                "type": "cron",
                "config": {"expression": "0 * * * *"},
            })
        assert resp.status_code == 201
        assert resp.json()["id"] == "trig-1"

    def test_create_webhook_trigger_has_url(self):
        trigger = _make_trigger()
        trigger.webhook_path = "/webhooks/abc123"
        trigger.trigger_type.value = "webhook"
        from nexus.types import TriggerType
        trigger.trigger_type = TriggerType.WEBHOOK
        mgr = MagicMock()
        mgr.create_trigger = AsyncMock(return_value=trigger)
        self.app.state.trigger_manager = mgr

        with TestClient(self.app) as client:
            resp = client.post("/v2/triggers", json={
                "workflow_id": "wf-1",
                "type": "webhook",
                "config": {},
            })
        assert resp.status_code == 201
        body = resp.json()
        assert "webhook_url" in body
        # webhook_base_url comes from nexus_config (module-level singleton)
        # default is "http://localhost:8000"; path is appended
        assert "/webhooks/abc123" in body["webhook_url"]

    def test_create_invalid_type(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/triggers", json={
                "workflow_id": "wf-1",
                "type": "invalid_type",
                "config": {},
            })
        assert resp.status_code == 422

    def test_create_workflow_not_found(self):
        from nexus.exceptions import WorkflowNotFound
        mgr = MagicMock()
        mgr.create_trigger = AsyncMock(side_effect=WorkflowNotFound("wf-missing", workflow_id="wf-missing"))
        self.app.state.trigger_manager = mgr
        with TestClient(self.app) as client:
            resp = client.post("/v2/triggers", json={
                "workflow_id": "wf-missing",
                "type": "cron",
                "config": {"expression": "* * * * *"},
            })
        assert resp.status_code == 404

    def test_create_trigger_error(self):
        from nexus.exceptions import TriggerError
        mgr = MagicMock()
        mgr.create_trigger = AsyncMock(side_effect=TriggerError("CRON trigger requires config.expression"))
        self.app.state.trigger_manager = mgr
        with TestClient(self.app) as client:
            resp = client.post("/v2/triggers", json={
                "workflow_id": "wf-1",
                "type": "cron",
                "config": {},
            })
        assert resp.status_code == 422


class TestGetTrigger:
    def setup_method(self):
        self.app = _make_app()
        self.trigger = _make_trigger(trigger_id="trig-1")
        mgr = MagicMock()
        mgr.list = AsyncMock(return_value=[self.trigger])
        self.app.state.trigger_manager = mgr

    def test_get_existing_trigger(self):
        with TestClient(self.app) as client:
            resp = client.get("/v2/triggers/trig-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "trig-1"

    def test_get_nonexistent_trigger(self):
        with TestClient(self.app) as client:
            resp = client.get("/v2/triggers/no-such-trigger")
        assert resp.status_code == 404


class TestUpdateTrigger:
    def setup_method(self):
        self.app = _make_app()
        self.trigger = _make_trigger()
        mgr = MagicMock()
        mgr.list = AsyncMock(return_value=[self.trigger])
        mgr.enable = AsyncMock(return_value=self.trigger)
        mgr.disable = AsyncMock(return_value=self.trigger)
        self.app.state.trigger_manager = mgr

    def test_update_enable(self):
        with TestClient(self.app) as client:
            resp = client.put("/v2/triggers/trig-1", json={"enabled": True})
        assert resp.status_code == 200
        self.app.state.trigger_manager.enable.assert_called_once()

    def test_update_disable(self):
        with TestClient(self.app) as client:
            resp = client.put("/v2/triggers/trig-1", json={"enabled": False})
        assert resp.status_code == 200
        self.app.state.trigger_manager.disable.assert_called_once()

    def test_update_no_change(self):
        with TestClient(self.app) as client:
            resp = client.put("/v2/triggers/trig-1", json={})
        assert resp.status_code == 200

    def test_update_nonexistent(self):
        mgr = MagicMock()
        mgr.list = AsyncMock(return_value=[])
        self.app.state.trigger_manager = mgr
        with TestClient(self.app) as client:
            resp = client.put("/v2/triggers/no-such", json={"enabled": True})
        assert resp.status_code == 404


class TestEnableDisableTrigger:
    def setup_method(self):
        self.app = _make_app()
        self.trigger = _make_trigger()
        mgr = MagicMock()
        mgr.enable = AsyncMock(return_value=self.trigger)
        mgr.disable = AsyncMock(return_value=self.trigger)
        self.app.state.trigger_manager = mgr

    def test_enable(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/triggers/trig-1/enable")
        assert resp.status_code == 200
        assert resp.json()["id"] == "trig-1"

    def test_disable(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/triggers/trig-1/disable")
        assert resp.status_code == 200

    def test_enable_not_found(self):
        from nexus.exceptions import TriggerError
        mgr = MagicMock()
        mgr.enable = AsyncMock(side_effect=TriggerError("Trigger 'x' not found"))
        self.app.state.trigger_manager = mgr
        with TestClient(self.app) as client:
            resp = client.post("/v2/triggers/x/enable")
        assert resp.status_code == 404

    def test_disable_not_found(self):
        from nexus.exceptions import TriggerError
        mgr = MagicMock()
        mgr.disable = AsyncMock(side_effect=TriggerError("Trigger 'x' not found"))
        self.app.state.trigger_manager = mgr
        with TestClient(self.app) as client:
            resp = client.post("/v2/triggers/x/disable")
        assert resp.status_code == 404


class TestDeleteTrigger:
    def setup_method(self):
        self.app = _make_app()
        mgr = MagicMock()
        mgr.delete = AsyncMock(return_value=True)
        self.app.state.trigger_manager = mgr

    def test_delete_returns_204(self):
        with TestClient(self.app) as client:
            resp = client.delete("/v2/triggers/trig-1")
        assert resp.status_code == 204

    def test_delete_not_found(self):
        from nexus.exceptions import TriggerError
        mgr = MagicMock()
        mgr.delete = AsyncMock(side_effect=TriggerError("Trigger 'x' not found"))
        self.app.state.trigger_manager = mgr
        with TestClient(self.app) as client:
            resp = client.delete("/v2/triggers/x")
        assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# ── WEBHOOK TESTS ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestWebhookCatchAll:
    def setup_method(self):
        self.app = _make_app()

    def test_webhook_post_no_auth_required(self):
        """Webhook endpoint must not require authentication.

        handler.handle returns a dict — route serialises it as JSONResponse.
        """
        handler = MagicMock()
        handler.handle = AsyncMock(return_value={"execution_id": "ex-1", "status": "queued"})
        self.app.state.webhook_handler = handler
        with TestClient(self.app) as client:
            # Deliberately no Authorization header
            resp = client.post("/v2/webhooks/test-path", json={"event": "test"})
        assert resp.status_code == 200
        assert resp.json()["execution_id"] == "ex-1"

    def test_webhook_passes_correct_args_to_handler(self):
        """Route must unpack request into positional handler kwargs."""
        handler = MagicMock()
        handler.handle = AsyncMock(return_value={"status": "accepted"})
        self.app.state.webhook_handler = handler
        with TestClient(self.app) as client:
            client.post("/v2/webhooks/abc/xyz", json={"k": "v"})
        call_kwargs = handler.handle.call_args.kwargs
        assert call_kwargs["webhook_path"] == "/webhooks/abc/xyz"
        assert call_kwargs["method"] == "POST"
        assert "content-type" in call_kwargs["headers"]
        assert isinstance(call_kwargs["body"], dict)  # parsed JSON

    def test_webhook_get(self):
        handler = MagicMock()
        handler.handle = AsyncMock(return_value={"ok": True})
        self.app.state.webhook_handler = handler
        with TestClient(self.app) as client:
            resp = client.get("/v2/webhooks/some/nested/path")
        assert resp.status_code == 200
        # Verify webhook_path reconstruction for nested paths
        call_kwargs = handler.handle.call_args.kwargs
        assert call_kwargs["webhook_path"] == "/webhooks/some/nested/path"
        assert call_kwargs["method"] == "GET"

    def test_webhook_no_handler_returns_503(self):
        # No webhook_handler on state
        with TestClient(self.app) as client:
            resp = client.post("/v2/webhooks/test")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["error"]

    def test_webhook_unknown_path_returns_404(self):
        """TriggerError('Unknown webhook path') should map to 404."""
        from nexus.exceptions import TriggerError
        handler = MagicMock()
        handler.handle = AsyncMock(side_effect=TriggerError("Unknown webhook path"))
        self.app.state.webhook_handler = handler
        with TestClient(self.app) as client:
            resp = client.post("/v2/webhooks/no-such-path")
        assert resp.status_code == 404

    def test_webhook_handler_exception_returns_500(self):
        handler = MagicMock()
        handler.handle = AsyncMock(side_effect=RuntimeError("boom"))
        self.app.state.webhook_handler = handler
        with TestClient(self.app, raise_server_exceptions=False) as client:
            resp = client.post("/v2/webhooks/error-path")
        assert resp.status_code == 500

    def test_webhook_delete_method(self):
        handler = MagicMock()
        handler.handle = AsyncMock(return_value={"deleted": True})
        self.app.state.webhook_handler = handler
        with TestClient(self.app) as client:
            resp = client.delete("/v2/webhooks/resource/123")
        assert resp.status_code == 200

    def test_webhook_model_dump_result(self):
        """Handler returning a Pydantic-like object is serialised via model_dump."""
        result = MagicMock()
        result.model_dump = MagicMock(return_value={"id": "ex-42", "status": "running"})
        handler = MagicMock()
        handler.handle = AsyncMock(return_value=result)
        self.app.state.webhook_handler = handler
        with TestClient(self.app) as client:
            resp = client.post("/v2/webhooks/trigger-path")
        assert resp.status_code == 200
        assert resp.json()["id"] == "ex-42"


# ─────────────────────────────────────────────────────────────────────────────
# ── AUTH MIDDLEWARE WEBHOOK BYPASS ────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthMiddlewareWebhookBypass:
    def test_webhook_path_in_public_prefixes(self):
        from nexus.auth.middleware import AuthMiddleware
        assert hasattr(AuthMiddleware, "PUBLIC_PREFIXES")
        assert any("/v2/webhooks" in p for p in AuthMiddleware.PUBLIC_PREFIXES)

    def test_public_paths_unchanged(self):
        from nexus.auth.middleware import AuthMiddleware
        assert "/v1/health" in AuthMiddleware.PUBLIC_PATHS
        assert "/v1/auth/token" in AuthMiddleware.PUBLIC_PATHS


# ─────────────────────────────────────────────────────────────────────────────
# ── MARKETPLACE TESTS ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketplaceSearch:
    def setup_method(self):
        self.app = _make_app()
        registry = MagicMock()
        registry.search = AsyncMock(return_value=[
            {"name": "nexus-plugin-slack", "plugin_name": "slack", "version": "1.0.0"},
        ])
        self.app.state.plugin_registry = registry

    def test_search_returns_results(self):
        with TestClient(self.app) as client:
            resp = client.get("/v2/marketplace/search?q=slack")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert data["count"] == 1
        assert data["query"] == "slack"

    def test_search_empty_query(self):
        registry = MagicMock()
        registry.search = AsyncMock(return_value=[])
        self.app.state.plugin_registry = registry
        with TestClient(self.app) as client:
            resp = client.get("/v2/marketplace/search")
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_search_no_registry_503(self):
        app = _make_app()
        with TestClient(app) as client:
            resp = client.get("/v2/marketplace/search?q=test")
        assert resp.status_code == 503


class TestMarketplaceInstalled:
    def setup_method(self):
        self.app = _make_app()
        manifest = _make_plugin_manifest("test-plugin")
        registry = MagicMock()
        registry.list_installed = MagicMock(return_value=[manifest])
        self.app.state.plugin_registry = registry

    def test_list_installed(self):
        with TestClient(self.app) as client:
            resp = client.get("/v2/marketplace/installed")
        assert resp.status_code == 200
        data = resp.json()
        assert "plugins" in data
        assert data["count"] == 1
        assert data["plugins"][0]["name"] == "test-plugin"

    def test_list_installed_empty(self):
        registry = MagicMock()
        registry.list_installed = MagicMock(return_value=[])
        self.app.state.plugin_registry = registry
        with TestClient(self.app) as client:
            resp = client.get("/v2/marketplace/installed")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


class TestMarketplaceInstall:
    def setup_method(self):
        self.app = _make_app()
        manifest = _make_plugin_manifest("my-plugin")
        registry = MagicMock()
        registry.install = AsyncMock(return_value=manifest)
        self.app.state.plugin_registry = registry

    def test_install_success(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/marketplace/install", json={"package_name": "my-plugin"})
        assert resp.status_code == 201
        assert resp.json()["name"] == "my-plugin"

    def test_install_with_version(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/marketplace/install", json={
                "package_name": "my-plugin",
                "version": "2.0.0",
            })
        assert resp.status_code == 201

    def test_install_force(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/marketplace/install", json={
                "package_name": "my-plugin",
                "force": True,
            })
        assert resp.status_code == 201
        _, kwargs = self.app.state.plugin_registry.install.call_args
        assert kwargs.get("force") is True

    def test_install_manifest_error_returns_422(self):
        from nexus.marketplace.plugin_sdk import PluginManifestError
        registry = MagicMock()
        registry.install = AsyncMock(side_effect=PluginManifestError("bad manifest"))
        self.app.state.plugin_registry = registry
        with TestClient(self.app) as client:
            resp = client.post("/v2/marketplace/install", json={"package_name": "bad-plugin"})
        assert resp.status_code == 422

    def test_install_install_error_returns_422(self):
        from nexus.marketplace.plugin_sdk import PluginInstallError
        registry = MagicMock()
        registry.install = AsyncMock(side_effect=PluginInstallError("pkg", "pip failed"))
        self.app.state.plugin_registry = registry
        with TestClient(self.app) as client:
            resp = client.post("/v2/marketplace/install", json={"package_name": "broken"})
        assert resp.status_code == 422


class TestMarketplaceUninstall:
    def setup_method(self):
        self.app = _make_app()
        registry = MagicMock()
        registry.uninstall = AsyncMock(return_value=True)
        self.app.state.plugin_registry = registry

    def test_uninstall_returns_204(self):
        with TestClient(self.app) as client:
            resp = client.delete("/v2/marketplace/my-plugin")
        assert resp.status_code == 204

    def test_uninstall_not_found(self):
        from nexus.marketplace.plugin_sdk import PluginNotFoundError
        registry = MagicMock()
        registry.uninstall = AsyncMock(side_effect=PluginNotFoundError("no-plugin"))
        self.app.state.plugin_registry = registry
        with TestClient(self.app) as client:
            resp = client.delete("/v2/marketplace/no-plugin")
        assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# ── NEW WORKFLOW ENDPOINT TESTS ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkflowActivatePause:
    def setup_method(self):
        self.app = _make_app()
        self.workflow = _make_workflow()
        mgr = MagicMock()
        mgr.activate = AsyncMock(return_value=self.workflow)
        mgr.pause = AsyncMock(return_value=self.workflow)
        self.app.state.workflow_manager = mgr

    def test_activate_success(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/wf-1/activate")
        assert resp.status_code == 200
        assert resp.json()["id"] == "wf-1"

    def test_activate_not_found(self):
        from nexus.exceptions import WorkflowNotFound
        mgr = MagicMock()
        mgr.activate = AsyncMock(side_effect=WorkflowNotFound("wf-x", workflow_id="wf-x"))
        self.app.state.workflow_manager = mgr
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/wf-x/activate")
        assert resp.status_code == 404

    def test_pause_success(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/wf-1/pause")
        assert resp.status_code == 200

    def test_pause_not_found(self):
        from nexus.exceptions import WorkflowNotFound
        mgr = MagicMock()
        mgr.pause = AsyncMock(side_effect=WorkflowNotFound("wf-x", workflow_id="wf-x"))
        self.app.state.workflow_manager = mgr
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/wf-x/pause")
        assert resp.status_code == 404

    def test_activate_validation_error_returns_422(self):
        """Activate raises WorkflowValidationError for invalid graphs → 422."""
        from nexus.exceptions import WorkflowValidationError
        mgr = MagicMock()
        mgr.activate = AsyncMock(side_effect=WorkflowValidationError("Graph has no start node"))
        self.app.state.workflow_manager = mgr
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/wf-1/activate")
        assert resp.status_code == 422


class TestWorkflowVersions:
    def setup_method(self):
        self.app = _make_app()
        self.workflow = _make_workflow()
        mgr = MagicMock()
        mgr.get_version_history = AsyncMock(return_value=[self.workflow, self.workflow])
        self.app.state.workflow_manager = mgr

    def test_get_versions(self):
        with TestClient(self.app) as client:
            resp = client.get("/v2/workflows/wf-1/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert "versions" in data
        assert len(data["versions"]) == 2

    def test_get_versions_not_found(self):
        from nexus.exceptions import WorkflowNotFound
        mgr = MagicMock()
        mgr.get_version_history = AsyncMock(side_effect=WorkflowNotFound("x", workflow_id="x"))
        self.app.state.workflow_manager = mgr
        with TestClient(self.app) as client:
            resp = client.get("/v2/workflows/x/versions")
        assert resp.status_code == 404

    def test_get_versions_empty(self):
        mgr = MagicMock()
        mgr.get_version_history = AsyncMock(return_value=[])
        self.app.state.workflow_manager = mgr
        with TestClient(self.app) as client:
            resp = client.get("/v2/workflows/wf-1/versions")
        assert resp.status_code == 200
        assert resp.json()["versions"] == []


class TestWorkflowRollback:
    def setup_method(self):
        self.app = _make_app()
        self.workflow = _make_workflow()
        mgr = MagicMock()
        mgr.rollback = AsyncMock(return_value=self.workflow)
        self.app.state.workflow_manager = mgr

    def test_rollback_success(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/wf-1/rollback/2")
        assert resp.status_code == 200
        assert resp.json()["id"] == "wf-1"

    def test_rollback_version_not_found(self):
        from nexus.exceptions import WorkflowNotFound
        mgr = MagicMock()
        mgr.rollback = AsyncMock(side_effect=WorkflowNotFound("v99 not found", workflow_id="wf-1"))
        self.app.state.workflow_manager = mgr
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/wf-1/rollback/99")
        assert resp.status_code == 404

    def test_rollback_calls_correct_version(self):
        with TestClient(self.app) as client:
            client.post("/v2/workflows/wf-1/rollback/5")
        call_kwargs = self.app.state.workflow_manager.rollback.call_args
        assert call_kwargs.kwargs.get("target_version") == 5


class TestWorkflowRefine:
    def setup_method(self):
        self.app = _make_app()
        self.workflow = _make_workflow()
        gen = MagicMock()
        gen.refine = AsyncMock(return_value=self.workflow)
        self.app.state.workflow_generator = gen

    def test_refine_success(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/wf-1/refine", json={"feedback": "Add error handling"})
        assert resp.status_code == 200
        assert resp.json()["id"] == "wf-1"

    def test_refine_not_found(self):
        from nexus.exceptions import WorkflowNotFound
        gen = MagicMock()
        gen.refine = AsyncMock(side_effect=WorkflowNotFound("x", workflow_id="x"))
        self.app.state.workflow_generator = gen
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/x/refine", json={"feedback": "fix it"})
        assert resp.status_code == 404

    def test_refine_generation_error(self):
        from nexus.exceptions import WorkflowGenerationError
        gen = MagicMock()
        gen.refine = AsyncMock(side_effect=WorkflowGenerationError("parse failed"))
        self.app.state.workflow_generator = gen
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/wf-1/refine", json={"feedback": "bad request"})
        assert resp.status_code == 422

    def test_refine_no_generator_503(self):
        app = _make_app()
        with TestClient(app) as client:
            resp = client.post("/v2/workflows/wf-1/refine", json={"feedback": "x"})
        assert resp.status_code == 503

    def test_refine_passes_feedback_to_generator(self):
        with TestClient(self.app) as client:
            client.post("/v2/workflows/wf-1/refine", json={"feedback": "Add logging"})
        args = self.app.state.workflow_generator.refine.call_args
        assert "Add logging" in str(args)


class TestWorkflowExplain:
    def setup_method(self):
        self.app = _make_app()
        gen = MagicMock()
        gen.explain = AsyncMock(return_value="This workflow does X and Y.")
        self.app.state.workflow_generator = gen

    def test_explain_success(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/wf-1/explain", json={"audience": "non-technical"})
        assert resp.status_code == 200
        data = resp.json()
        assert "explanation" in data
        assert "This workflow" in data["explanation"]
        assert data["audience"] == "non-technical"

    def test_explain_default_audience(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/wf-1/explain", json={})
        assert resp.status_code == 200
        assert resp.json()["audience"] == "technical"

    def test_explain_not_found(self):
        from nexus.exceptions import WorkflowNotFound
        gen = MagicMock()
        gen.explain = AsyncMock(side_effect=WorkflowNotFound("x", workflow_id="x"))
        self.app.state.workflow_generator = gen
        with TestClient(self.app) as client:
            resp = client.post("/v2/workflows/x/explain", json={})
        assert resp.status_code == 404

    def test_explain_no_generator_503(self):
        app = _make_app()
        with TestClient(app) as client:
            resp = client.post("/v2/workflows/wf-1/explain", json={})
        assert resp.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# ── EXECUTION DELETE + STREAM TESTS ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteExecution:
    def setup_method(self):
        self.app = _make_app()
        chain = MagicMock()
        chain.id = "exec-1"
        chain.status = "completed"
        ledger = MagicMock()
        ledger.get_chain = MagicMock(return_value=chain)
        ledger.delete_chain = MagicMock()
        self.app.state.ledger = ledger

    def test_delete_completed_returns_204(self):
        with TestClient(self.app) as client:
            resp = client.delete("/v2/executions/exec-1")
        assert resp.status_code == 204

    def test_delete_running_returns_409(self):
        chain = MagicMock()
        chain.status = "running"
        ledger = MagicMock()
        ledger.get_chain = MagicMock(return_value=chain)
        self.app.state.ledger = ledger
        with TestClient(self.app) as client:
            resp = client.delete("/v2/executions/exec-1")
        assert resp.status_code == 409

    def test_delete_not_found_returns_404(self):
        ledger = MagicMock()
        ledger.get_chain = MagicMock(return_value=None)
        self.app.state.ledger = ledger
        with TestClient(self.app) as client:
            resp = client.delete("/v2/executions/no-such")
        assert resp.status_code == 404

    def test_delete_no_ledger_returns_404(self):
        app = _make_app()
        with TestClient(app) as client:
            resp = client.delete("/v2/executions/exec-1")
        assert resp.status_code == 404


class TestStreamExecution:
    def setup_method(self):
        self.app = _make_app()

    def test_stream_returns_sse_headers(self):
        chain = MagicMock()
        chain.status = "completed"
        ledger = MagicMock()
        ledger.get_chain = MagicMock(return_value=chain)
        ledger.get_seals_for_chain = MagicMock(return_value=[])
        self.app.state.ledger = ledger
        with TestClient(self.app) as client:
            resp = client.get("/v2/executions/exec-1/stream")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_replays_completed_seals(self):
        seal = MagicMock()
        seal.model_dump = MagicMock(return_value={"action": "search", "status": "allowed"})
        chain = MagicMock()
        chain.status = "completed"
        ledger = MagicMock()
        ledger.get_chain = MagicMock(return_value=chain)
        ledger.get_seals_for_chain = MagicMock(return_value=[seal])
        self.app.state.ledger = ledger
        with TestClient(self.app) as client:
            resp = client.get("/v2/executions/exec-1/stream")
        assert resp.status_code == 200
        body = resp.text
        assert "seal" in body


# ─────────────────────────────────────────────────────────────────────────────
# ── JOB RESULT TESTS ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestJobResult:
    def setup_method(self):
        self.app = _make_app()

    def test_get_result_uses_get_job_result_if_exists(self):
        dispatcher = MagicMock()
        dispatcher.get_job_result = AsyncMock(return_value={"result": "done", "job_id": "j1"})
        self.app.state.dispatcher = dispatcher
        with TestClient(self.app) as client:
            resp = client.get("/v2/jobs/j1/result")
        assert resp.status_code == 200
        assert resp.json()["result"] == "done"
        dispatcher.get_job_result.assert_called_once_with("j1")

    def test_get_result_fallback_to_get_status(self):
        dispatcher = MagicMock(spec=[])  # no get_job_result attribute
        dispatcher.get_job_status = AsyncMock(return_value={"status": "complete", "job_id": "j2"})
        # spec=[] means hasattr(dispatcher, 'get_job_result') is False
        self.app.state.dispatcher = dispatcher
        with TestClient(self.app) as client:
            resp = client.get("/v2/jobs/j2/result")
        assert resp.status_code == 200
        dispatcher.get_job_status.assert_called_once_with("j2")

    def test_get_status_still_works(self):
        dispatcher = MagicMock()
        dispatcher.get_job_status = AsyncMock(return_value={"status": "pending"})
        self.app.state.dispatcher = dispatcher
        with TestClient(self.app) as client:
            resp = client.get("/v2/jobs/j3")
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# ── MCP REFRESH TESTS ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPRefresh:
    def setup_method(self):
        self.app = _make_app()

    def _make_adapter(self, tools_count: int = 3):
        server = MagicMock()
        server.tenant_id = "test-tenant"
        server.model_dump = MagicMock(return_value={"id": "srv-1", "name": "test-server"})
        adapter = MagicMock()
        adapter.get_server = MagicMock(return_value=server)
        adapter.unregister_server = AsyncMock()
        adapter.register_server = AsyncMock(return_value=["tool1", "tool2", "tool3"][:tools_count])
        return adapter

    def test_refresh_success(self):
        self.app.state.mcp_adapter = self._make_adapter(tools_count=2)
        with TestClient(self.app) as client:
            resp = client.post("/v2/mcp/servers/srv-1/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["refreshed"] is True
        assert data["server_id"] == "srv-1"
        assert data["tools_registered"] == 2

    def test_refresh_server_not_found(self):
        adapter = MagicMock()
        adapter.get_server = MagicMock(return_value=None)
        self.app.state.mcp_adapter = adapter
        with TestClient(self.app) as client:
            resp = client.post("/v2/mcp/servers/no-such/refresh")
        assert resp.status_code == 404

    def test_refresh_wrong_tenant(self):
        server = MagicMock()
        server.tenant_id = "other-tenant"  # different from test-tenant
        adapter = MagicMock()
        adapter.get_server = MagicMock(return_value=server)
        self.app.state.mcp_adapter = adapter
        with TestClient(self.app) as client:
            resp = client.post("/v2/mcp/servers/srv-1/refresh")
        assert resp.status_code == 404

    def test_refresh_connection_error(self):
        from nexus.exceptions import MCPConnectionError
        server = MagicMock()
        server.tenant_id = "test-tenant"
        adapter = MagicMock()
        adapter.get_server = MagicMock(return_value=server)
        adapter.unregister_server = AsyncMock()
        adapter.register_server = AsyncMock(side_effect=MCPConnectionError("connection refused"))
        self.app.state.mcp_adapter = adapter
        with TestClient(self.app) as client:
            resp = client.post("/v2/mcp/servers/srv-1/refresh")
        assert resp.status_code == 502

    def test_refresh_no_adapter_503(self):
        with TestClient(self.app) as client:
            resp = client.post("/v2/mcp/servers/srv-1/refresh")
        assert resp.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# ── ROUTE EXISTENCE SMOKE TESTS ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteRegistration:
    """Verify all new routes appear in the app's route table."""

    def setup_method(self):
        self.app = _make_app()

    def _routes(self) -> set[tuple[str, str]]:
        """Return set of (method, path) tuples from the app's route table."""
        result = set()
        for route in self.app.routes:
            if hasattr(route, "methods") and hasattr(route, "path"):
                for method in (route.methods or []):
                    result.add((method.upper(), route.path))
        return result

    def test_trigger_routes_registered(self):
        routes = self._routes()
        assert ("GET", "/v2/triggers") in routes
        assert ("POST", "/v2/triggers") in routes
        assert ("GET", "/v2/triggers/{trigger_id}") in routes
        assert ("PUT", "/v2/triggers/{trigger_id}") in routes
        assert ("POST", "/v2/triggers/{trigger_id}/enable") in routes
        assert ("POST", "/v2/triggers/{trigger_id}/disable") in routes
        assert ("DELETE", "/v2/triggers/{trigger_id}") in routes

    def test_webhook_route_registered(self):
        routes = self._routes()
        # Catch-all is registered as /v2/webhooks/{path:path}
        paths = {p for _, p in routes}
        assert any("webhooks" in p for p in paths)

    def test_marketplace_routes_registered(self):
        routes = self._routes()
        assert ("GET", "/v2/marketplace/search") in routes
        assert ("GET", "/v2/marketplace/installed") in routes
        assert ("POST", "/v2/marketplace/install") in routes
        assert ("DELETE", "/v2/marketplace/{plugin_name}") in routes

    def test_workflow_lifecycle_routes_registered(self):
        routes = self._routes()
        assert ("POST", "/v2/workflows/{workflow_id}/activate") in routes
        assert ("POST", "/v2/workflows/{workflow_id}/pause") in routes
        assert ("GET", "/v2/workflows/{workflow_id}/versions") in routes
        assert ("POST", "/v2/workflows/{workflow_id}/rollback/{version}") in routes
        assert ("POST", "/v2/workflows/{workflow_id}/refine") in routes
        assert ("POST", "/v2/workflows/{workflow_id}/explain") in routes

    def test_execution_new_routes_registered(self):
        routes = self._routes()
        assert ("DELETE", "/v2/executions/{execution_id}") in routes
        assert ("GET", "/v2/executions/{execution_id}/stream") in routes

    def test_job_result_route_registered(self):
        routes = self._routes()
        assert ("GET", "/v2/jobs/{job_id}/result") in routes

    def test_mcp_refresh_route_registered(self):
        routes = self._routes()
        assert ("POST", "/v2/mcp/servers/{server_id}/refresh") in routes
