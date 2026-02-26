"""Phase 26 — Background Execution tests.

Covers:
- WorkflowDispatcher.dispatch() decision tree (all 4 paths)
- WorkflowDispatcher.get_job_status() for all states
- execute_workflow_task() with mocked engine and repository
- refresh_mcp_connections_task() with mocked adapter
- Repository execution CRUD (4 methods)
- POST /v2/workflows/{id}/run endpoint
- GET /v2/jobs/{job_id} endpoint
- TriggerManager.fire() routes through dispatcher when present
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_execution(**kwargs):
    """Minimal WorkflowExecution-like object."""
    from nexus.types import WorkflowExecution, TriggerType, ChainStatus
    defaults = dict(
        workflow_id=str(uuid.uuid4()),
        workflow_version=1,
        tenant_id="demo",
        trigger_type=TriggerType.MANUAL,
        chain_id=str(uuid.uuid4()),
        status=ChainStatus.COMPLETED,
    )
    defaults.update(kwargs)
    return WorkflowExecution(**defaults)


def _make_workflow(num_steps: int = 2):
    """Minimal WorkflowDefinition-like object with `steps` list."""
    obj = MagicMock()
    obj.steps = [MagicMock() for _ in range(num_steps)]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# WorkflowDispatcher — dispatch() decision tree
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatcherDecisionTree:
    """All 4 dispatch paths: force_bg, source trigger, step count, inline."""

    def _make_dispatcher(self):
        from nexus.workers.dispatcher import WorkflowDispatcher
        engine = AsyncMock()
        redis_pool = AsyncMock()
        config = MagicMock()
        return WorkflowDispatcher(engine, redis_pool, config), engine, redis_pool

    @pytest.mark.asyncio
    async def test_force_background_enqueues(self):
        dispatcher, engine, redis_pool = self._make_dispatcher()
        job = MagicMock()
        job.job_id = "job-abc"
        redis_pool.enqueue_job = AsyncMock(return_value=job)

        result = await dispatcher.dispatch(
            workflow_id="wf1",
            tenant_id="t1",
            trigger_data={"_source": "manual"},
            force_background=True,
        )

        assert result["status"] == "queued"
        assert result["mode"] == "background"
        assert result["job_id"] == "job-abc"
        redis_pool.enqueue_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_source_enqueues(self):
        dispatcher, engine, redis_pool = self._make_dispatcher()
        job = MagicMock(); job.job_id = "job-wh"
        redis_pool.enqueue_job = AsyncMock(return_value=job)

        result = await dispatcher.dispatch(
            workflow_id="wf1",
            tenant_id="t1",
            trigger_data={"_source": "webhook"},
        )

        assert result["mode"] == "background"
        redis_pool.enqueue_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_cron_source_enqueues(self):
        dispatcher, engine, redis_pool = self._make_dispatcher()
        job = MagicMock(); job.job_id = "job-cr"
        redis_pool.enqueue_job = AsyncMock(return_value=job)

        result = await dispatcher.dispatch(
            workflow_id="wf1",
            tenant_id="t1",
            trigger_data={"_source": "cron"},
        )

        assert result["mode"] == "background"

    @pytest.mark.asyncio
    async def test_event_source_enqueues(self):
        dispatcher, engine, redis_pool = self._make_dispatcher()
        job = MagicMock(); job.job_id = "job-ev"
        redis_pool.enqueue_job = AsyncMock(return_value=job)

        result = await dispatcher.dispatch(
            workflow_id="wf1",
            tenant_id="t1",
            trigger_data={"_source": "event"},
        )

        assert result["mode"] == "background"

    @pytest.mark.asyncio
    async def test_schedule_source_enqueues(self):
        dispatcher, engine, redis_pool = self._make_dispatcher()
        job = MagicMock(); job.job_id = "job-sc"
        redis_pool.enqueue_job = AsyncMock(return_value=job)

        result = await dispatcher.dispatch(
            workflow_id="wf1",
            tenant_id="t1",
            trigger_data={"_source": "schedule"},
        )

        assert result["mode"] == "background"

    @pytest.mark.asyncio
    async def test_large_workflow_enqueues(self):
        dispatcher, engine, redis_pool = self._make_dispatcher()
        job = MagicMock(); job.job_id = "job-lg"
        redis_pool.enqueue_job = AsyncMock(return_value=job)

        workflow_manager = AsyncMock()
        workflow_manager.get = AsyncMock(return_value=_make_workflow(num_steps=6))

        result = await dispatcher.dispatch(
            workflow_id="wf1",
            tenant_id="t1",
            trigger_data={"_source": "manual"},
            workflow_manager=workflow_manager,
        )

        assert result["mode"] == "background"
        redis_pool.enqueue_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_small_manual_workflow_runs_inline(self):
        dispatcher, engine, redis_pool = self._make_dispatcher()
        execution = _make_execution()
        engine.run_workflow = AsyncMock(return_value=execution)

        workflow_manager = AsyncMock()
        workflow_manager.get = AsyncMock(return_value=_make_workflow(num_steps=3))

        result = await dispatcher.dispatch(
            workflow_id="wf1",
            tenant_id="t1",
            trigger_data={"_source": "manual"},
            workflow_manager=workflow_manager,
        )

        assert result["mode"] == "inline"
        assert result["execution_id"] == execution.id
        redis_pool.enqueue_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_exactly_5_steps_runs_inline(self):
        """Boundary: exactly 5 steps → inline."""
        dispatcher, engine, redis_pool = self._make_dispatcher()
        execution = _make_execution()
        engine.run_workflow = AsyncMock(return_value=execution)

        workflow_manager = AsyncMock()
        workflow_manager.get = AsyncMock(return_value=_make_workflow(num_steps=5))

        result = await dispatcher.dispatch(
            workflow_id="wf1",
            tenant_id="t1",
            trigger_data={"_source": "manual"},
            workflow_manager=workflow_manager,
        )

        assert result["mode"] == "inline"

    @pytest.mark.asyncio
    async def test_no_workflow_manager_falls_through_to_inline(self):
        """Without workflow_manager, step count check is skipped → inline."""
        dispatcher, engine, redis_pool = self._make_dispatcher()
        execution = _make_execution()
        engine.run_workflow = AsyncMock(return_value=execution)

        result = await dispatcher.dispatch(
            workflow_id="wf1",
            tenant_id="t1",
            trigger_data={"_source": "manual"},
        )

        assert result["mode"] == "inline"

    @pytest.mark.asyncio
    async def test_inline_includes_duration_ms(self):
        dispatcher, engine, redis_pool = self._make_dispatcher()
        execution = _make_execution()
        engine.run_workflow = AsyncMock(return_value=execution)

        result = await dispatcher.dispatch("wf1", "t1", {"_source": "manual"})

        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], int)

    @pytest.mark.asyncio
    async def test_inline_saves_execution_via_repository(self):
        dispatcher, engine, redis_pool = self._make_dispatcher()
        execution = _make_execution()
        engine.run_workflow = AsyncMock(return_value=execution)

        repo = AsyncMock()
        repo.save_execution = AsyncMock()
        repo.get_chain_seals = AsyncMock(return_value=[MagicMock(), MagicMock()])

        result = await dispatcher.dispatch(
            "wf1", "t1", {"_source": "manual"}, repository=repo
        )

        repo.save_execution.assert_called_once_with(execution)
        assert result["seal_count"] == 2

    @pytest.mark.asyncio
    async def test_workflow_manager_lookup_failure_falls_to_inline(self):
        """If workflow lookup raises, fall through to inline."""
        dispatcher, engine, redis_pool = self._make_dispatcher()
        execution = _make_execution()
        engine.run_workflow = AsyncMock(return_value=execution)

        workflow_manager = AsyncMock()
        workflow_manager.get = AsyncMock(side_effect=Exception("not found"))

        result = await dispatcher.dispatch(
            "wf1", "t1", {"_source": "manual"},
            workflow_manager=workflow_manager,
        )

        assert result["mode"] == "inline"


# ─────────────────────────────────────────────────────────────────────────────
# WorkflowDispatcher — get_job_status()
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatcherGetJobStatus:

    def _make_dispatcher(self):
        from nexus.workers.dispatcher import WorkflowDispatcher
        return WorkflowDispatcher(AsyncMock(), AsyncMock(), MagicMock())

    @pytest.mark.asyncio
    async def test_queued_status(self):
        dispatcher = self._make_dispatcher()

        mock_job = AsyncMock()
        mock_status = MagicMock()
        mock_status.value = "queued"
        mock_job.status = AsyncMock(return_value=mock_status)

        with patch("nexus.workers.dispatcher.Job", return_value=mock_job):
            result = await dispatcher.get_job_status("job-123")

        assert result["job_id"] == "job-123"
        assert result["status"] == "queued"
        assert result["result"] is None

    @pytest.mark.asyncio
    async def test_in_progress_status(self):
        dispatcher = self._make_dispatcher()

        mock_job = AsyncMock()
        mock_status = MagicMock()
        mock_status.value = "in_progress"
        mock_job.status = AsyncMock(return_value=mock_status)

        with patch("nexus.workers.dispatcher.Job", return_value=mock_job):
            result = await dispatcher.get_job_status("job-456")

        assert result["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_complete_status_returns_result(self):
        dispatcher = self._make_dispatcher()
        job_result = {"execution_id": "exec-1", "status": "completed", "seal_count": 3, "duration_ms": 100}

        mock_job = AsyncMock()
        mock_status = MagicMock()
        mock_status.value = "complete"
        mock_job.status = AsyncMock(return_value=mock_status)
        mock_job.result = AsyncMock(return_value=job_result)

        with patch("nexus.workers.dispatcher.Job", return_value=mock_job):
            result = await dispatcher.get_job_status("job-789")

        assert result["status"] == "complete"
        assert result["result"] == job_result

    @pytest.mark.asyncio
    async def test_not_found_status(self):
        dispatcher = self._make_dispatcher()

        mock_job = AsyncMock()
        mock_status = MagicMock()
        mock_status.value = "not_found"
        mock_job.status = AsyncMock(return_value=mock_status)

        with patch("nexus.workers.dispatcher.Job", return_value=mock_job):
            result = await dispatcher.get_job_status("job-missing")

        assert result["status"] == "not_found"
        assert result["result"] is None

    @pytest.mark.asyncio
    async def test_failed_job_returns_error(self):
        dispatcher = self._make_dispatcher()

        mock_job = AsyncMock()
        mock_status = MagicMock()
        mock_status.value = "failed"
        mock_job.status = AsyncMock(return_value=mock_status)

        with patch("nexus.workers.dispatcher.Job", return_value=mock_job):
            result = await dispatcher.get_job_status("job-failed")

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_arq_not_installed_returns_not_found(self):
        dispatcher = self._make_dispatcher()

        with patch("nexus.workers.dispatcher.Job", side_effect=ImportError("arq not installed")):
            result = await dispatcher.get_job_status("job-xyz")

        assert result["job_id"] == "job-xyz"
        assert result["status"] == "not_found"
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_redis_error_returns_not_found(self):
        dispatcher = self._make_dispatcher()

        mock_job = AsyncMock()
        mock_job.status = AsyncMock(side_effect=ConnectionError("redis down"))

        with patch("nexus.workers.dispatcher.Job", return_value=mock_job):
            result = await dispatcher.get_job_status("job-err")

        assert result["status"] == "not_found"
        assert "redis down" in result["error"]


# ─────────────────────────────────────────────────────────────────────────────
# execute_workflow_task()
# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteWorkflowTask:

    @pytest.mark.asyncio
    async def test_success_returns_execution_metadata(self):
        from nexus.workers.queue import execute_workflow_task

        execution = _make_execution()
        engine = AsyncMock()
        engine.run_workflow = AsyncMock(return_value=execution)

        repo = AsyncMock()
        repo.save_execution = AsyncMock()
        repo.get_chain_seals = AsyncMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        ctx = {"engine": engine, "repository": repo}
        result = await execute_workflow_task(ctx, "wf1", "demo", {"_source": "manual"})

        assert result["execution_id"] == execution.id
        assert result["status"] == "completed"
        assert result["seal_count"] == 3
        assert "duration_ms" in result

    @pytest.mark.asyncio
    async def test_engine_failure_returns_failed_status(self):
        from nexus.workers.queue import execute_workflow_task

        engine = AsyncMock()
        engine.run_workflow = AsyncMock(side_effect=RuntimeError("engine exploded"))

        repo = AsyncMock()
        ctx = {"engine": engine, "repository": repo}

        result = await execute_workflow_task(ctx, "wf1", "demo", {})

        assert result["status"] == "failed"
        assert result["execution_id"] is None
        assert "engine exploded" in result["error"]

    @pytest.mark.asyncio
    async def test_saves_execution_to_repo(self):
        from nexus.workers.queue import execute_workflow_task

        execution = _make_execution()
        engine = AsyncMock()
        engine.run_workflow = AsyncMock(return_value=execution)

        repo = AsyncMock()
        repo.save_execution = AsyncMock()
        repo.get_chain_seals = AsyncMock(return_value=[])

        ctx = {"engine": engine, "repository": repo}
        await execute_workflow_task(ctx, "wf1", "demo", {})

        repo.save_execution.assert_called_once_with(execution)

    @pytest.mark.asyncio
    async def test_empty_chain_id_returns_zero_seals(self):
        from nexus.workers.queue import execute_workflow_task

        execution = _make_execution(chain_id="")
        engine = AsyncMock()
        engine.run_workflow = AsyncMock(return_value=execution)

        repo = AsyncMock()
        repo.save_execution = AsyncMock()
        repo.get_chain_seals = AsyncMock(return_value=[])

        ctx = {"engine": engine, "repository": repo}
        result = await execute_workflow_task(ctx, "wf1", "demo", {})

        repo.get_chain_seals.assert_not_called()
        assert result["seal_count"] == 0

    @pytest.mark.asyncio
    async def test_returns_duration_ms(self):
        from nexus.workers.queue import execute_workflow_task

        execution = _make_execution()
        engine = AsyncMock()
        engine.run_workflow = AsyncMock(return_value=execution)
        repo = AsyncMock()
        repo.save_execution = AsyncMock()
        repo.get_chain_seals = AsyncMock(return_value=[])

        ctx = {"engine": engine, "repository": repo}
        result = await execute_workflow_task(ctx, "wf1", "demo", {})

        assert isinstance(result["duration_ms"], int)
        assert result["duration_ms"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# refresh_mcp_connections_task()
# ─────────────────────────────────────────────────────────────────────────────

class TestRefreshMCPConnectionsTask:

    @pytest.mark.asyncio
    async def test_success(self):
        from nexus.workers.queue import refresh_mcp_connections_task

        mcp_adapter = AsyncMock()
        mcp_adapter.reconnect_all = AsyncMock(return_value={
            "reconnected": ["server1", "server2"],
            "failed": [],
            "errors": {},
        })
        ctx = {"mcp_adapter": mcp_adapter}

        result = await refresh_mcp_connections_task(ctx, "demo")

        assert result["tenant_id"] == "demo"
        assert result["reconnected"] == ["server1", "server2"]
        assert result["failed"] == []

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        from nexus.workers.queue import refresh_mcp_connections_task

        mcp_adapter = AsyncMock()
        mcp_adapter.reconnect_all = AsyncMock(return_value={
            "reconnected": ["server1"],
            "failed": ["server2"],
            "errors": {"server2": "timeout"},
        })
        ctx = {"mcp_adapter": mcp_adapter}

        result = await refresh_mcp_connections_task(ctx, "demo")

        assert "server2" in result["failed"]
        assert "server2" in result["errors"]

    @pytest.mark.asyncio
    async def test_adapter_exception_returns_error_dict(self):
        from nexus.workers.queue import refresh_mcp_connections_task

        mcp_adapter = AsyncMock()
        mcp_adapter.reconnect_all = AsyncMock(side_effect=ConnectionError("redis gone"))
        ctx = {"mcp_adapter": mcp_adapter}

        result = await refresh_mcp_connections_task(ctx, "demo")

        assert result["reconnected"] == []
        assert "_task" in result["errors"]


# ─────────────────────────────────────────────────────────────────────────────
# Repository — execution CRUD
# ─────────────────────────────────────────────────────────────────────────────

class TestRepositoryExecutionCRUD:

    def _make_mock_session(self):
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_save_execution(self):
        from nexus.db.repository import Repository
        execution = _make_execution()
        session = self._make_mock_session()

        # Mock execute to return empty for the refresh
        mock_result = MagicMock()
        session.execute = AsyncMock(return_value=mock_result)

        repo = Repository(session)
        with patch("nexus.db.repository.WorkflowExecutionModel") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance

            result = await repo.save_execution(execution)

            session.add.assert_called_once_with(mock_instance)
            session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_execution_returns_none_when_missing(self):
        from nexus.db.repository import Repository
        session = self._make_mock_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        session.execute = AsyncMock(return_value=mock_result)

        repo = Repository(session)
        result = await repo.get_execution("demo", "nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_execution_returns_model(self):
        from nexus.db.repository import Repository
        session = self._make_mock_session()

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_model)
        session.execute = AsyncMock(return_value=mock_result)

        repo = Repository(session)
        result = await repo.get_execution("demo", "exec-id")
        assert result is mock_model

    @pytest.mark.asyncio
    async def test_update_execution_returns_none_when_missing(self):
        from nexus.db.repository import Repository
        session = self._make_mock_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        session.execute = AsyncMock(return_value=mock_result)

        repo = Repository(session)
        result = await repo.update_execution("nonexistent", {"status": "completed"})
        assert result is None

    @pytest.mark.asyncio
    async def test_update_execution_applies_updates(self):
        from nexus.db.repository import Repository
        session = self._make_mock_session()

        mock_model = MagicMock()
        mock_model.status = "planning"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_model)
        session.execute = AsyncMock(return_value=mock_result)

        repo = Repository(session)
        result = await repo.update_execution("exec-1", {"status": "completed"})

        assert mock_model.status == "completed"
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_executions_returns_list(self):
        from nexus.db.repository import Repository
        session = self._make_mock_session()

        mock_models = [MagicMock(), MagicMock()]
        mock_scalars = MagicMock()
        mock_scalars.all = MagicMock(return_value=mock_models)
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=mock_scalars)
        session.execute = AsyncMock(return_value=mock_result)

        repo = Repository(session)
        result = await repo.list_executions("demo")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_executions_with_workflow_filter(self):
        from nexus.db.repository import Repository
        session = self._make_mock_session()

        mock_scalars = MagicMock()
        mock_scalars.all = MagicMock(return_value=[])
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=mock_scalars)
        session.execute = AsyncMock(return_value=mock_result)

        repo = Repository(session)
        result = await repo.list_executions("demo", workflow_id="wf-specific")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# API endpoint: POST /v2/workflows/{id}/run
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWorkflowEndpoint:

    def _make_app(self, dispatcher=None, workflow_manager=None):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from nexus.api.routes.workflows import router

        app = FastAPI()
        app.include_router(router)

        # Minimal state
        app.state.dispatcher = dispatcher or AsyncMock()
        app.state.workflow_manager = workflow_manager or AsyncMock()
        app.state.async_session = AsyncMock()

        # Inject tenant via middleware substitute
        from starlette.middleware.base import BaseHTTPMiddleware
        class FakeTenantMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.tenant_id = "demo"
                return await call_next(request)

        app.add_middleware(FakeTenantMiddleware)
        return TestClient(app, raise_server_exceptions=False)

    def test_run_workflow_inline_returns_200(self):
        dispatcher = AsyncMock()
        dispatcher.dispatch = AsyncMock(return_value={
            "execution_id": "exec-1",
            "status": "completed",
            "seal_count": 2,
            "duration_ms": 150,
            "mode": "inline",
        })

        workflow_manager = AsyncMock()
        workflow_manager.list = AsyncMock(return_value=[])

        client = self._make_app(dispatcher=dispatcher, workflow_manager=workflow_manager)
        resp = client.post("/v2/workflows/wf-1/run", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "inline"
        assert data["execution_id"] == "exec-1"

    def test_run_workflow_background_returns_queued(self):
        dispatcher = AsyncMock()
        dispatcher.dispatch = AsyncMock(return_value={
            "job_id": "job-xyz",
            "status": "queued",
            "mode": "background",
        })

        workflow_manager = AsyncMock()
        workflow_manager.list = AsyncMock(return_value=[])

        client = self._make_app(dispatcher=dispatcher, workflow_manager=workflow_manager)
        resp = client.post("/v2/workflows/wf-1/run", json={"force_background": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "background"
        assert data["job_id"] == "job-xyz"

    def test_run_workflow_no_dispatcher_returns_503(self):
        client = self._make_app(dispatcher=None)
        # Override state.dispatcher to None AFTER construction
        # Actually, the dispatcher is set before TestClient starts
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from nexus.api.routes.workflows import router
        from starlette.middleware.base import BaseHTTPMiddleware

        app = FastAPI()
        app.include_router(router)
        app.state.dispatcher = None
        app.state.workflow_manager = AsyncMock()
        app.state.async_session = AsyncMock()

        class FakeTenantMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.tenant_id = "demo"
                return await call_next(request)

        app.add_middleware(FakeTenantMiddleware)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v2/workflows/wf-1/run", json={})
        assert resp.status_code == 503

    def test_run_workflow_passes_context(self):
        dispatcher = AsyncMock()
        dispatcher.dispatch = AsyncMock(return_value={"mode": "inline", "status": "completed"})
        workflow_manager = AsyncMock()
        workflow_manager.list = AsyncMock(return_value=[])

        client = self._make_app(dispatcher=dispatcher, workflow_manager=workflow_manager)
        client.post("/v2/workflows/wf-1/run", json={"context": {"key": "value"}})

        call_kwargs = dispatcher.dispatch.call_args
        assert call_kwargs is not None
        # trigger_data is passed as a keyword arg
        trigger_data = call_kwargs.kwargs.get("trigger_data", {})
        assert trigger_data.get("_source") == "manual"
        assert trigger_data.get("key") == "value"


# ─────────────────────────────────────────────────────────────────────────────
# API endpoint: GET /v2/jobs/{job_id}
# ─────────────────────────────────────────────────────────────────────────────

class TestGetJobEndpoint:

    def _make_client(self, dispatcher=None):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from nexus.api.routes.jobs import router
        from starlette.middleware.base import BaseHTTPMiddleware

        app = FastAPI()
        app.include_router(router)
        app.state.dispatcher = dispatcher or AsyncMock()

        class FakeTenantMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.tenant_id = "demo"
                return await call_next(request)

        app.add_middleware(FakeTenantMiddleware)
        return TestClient(app, raise_server_exceptions=False)

    def test_get_job_queued(self):
        dispatcher = AsyncMock()
        dispatcher.get_job_status = AsyncMock(return_value={
            "job_id": "job-123",
            "status": "queued",
            "result": None,
            "error": None,
        })
        client = self._make_client(dispatcher)
        resp = client.get("/job-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["job_id"] == "job-123"

    def test_get_job_complete(self):
        dispatcher = AsyncMock()
        dispatcher.get_job_status = AsyncMock(return_value={
            "job_id": "job-done",
            "status": "complete",
            "result": {"execution_id": "exec-1", "seal_count": 5},
            "error": None,
        })
        client = self._make_client(dispatcher)
        resp = client.get("/job-done")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "complete"
        assert data["result"]["seal_count"] == 5

    def test_get_job_not_found(self):
        dispatcher = AsyncMock()
        dispatcher.get_job_status = AsyncMock(return_value={
            "job_id": "job-nope",
            "status": "not_found",
            "result": None,
            "error": "Job not found",
        })
        client = self._make_client(dispatcher)
        resp = client.get("/job-nope")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_found"

    def test_get_job_failed(self):
        dispatcher = AsyncMock()
        dispatcher.get_job_status = AsyncMock(return_value={
            "job_id": "job-fail",
            "status": "failed",
            "result": None,
            "error": "Engine crashed",
        })
        client = self._make_client(dispatcher)
        resp = client.get("/job-fail")
        data = resp.json()
        assert data["status"] == "failed"


# ─────────────────────────────────────────────────────────────────────────────
# TriggerManager.fire() routes through dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class TestTriggerManagerDispatcherRouting:

    def _make_trigger(self, workflow_id="wf-1", tenant_id="demo"):
        from nexus.types import TriggerConfig, TriggerType
        return TriggerConfig(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            trigger_type=TriggerType.WEBHOOK,
            config={},
            enabled=True,
        )

    @pytest.mark.asyncio
    async def test_fire_uses_dispatcher_when_present(self):
        from nexus.triggers.manager import TriggerManager

        engine = AsyncMock()
        workflow_manager = AsyncMock()
        repository = AsyncMock()
        event_bus = AsyncMock()
        config = MagicMock()
        dispatcher = AsyncMock()

        # Repository update_trigger must return the trigger
        trigger = self._make_trigger()
        repository.update_trigger = AsyncMock(return_value=trigger)

        dispatcher.dispatch = AsyncMock(return_value={"mode": "background", "job_id": "j1"})

        manager = TriggerManager(engine, workflow_manager, repository, event_bus, config, dispatcher=dispatcher)
        result = await manager.fire(trigger)

        dispatcher.dispatch.assert_called_once()
        engine.run_workflow.assert_not_called()

    @pytest.mark.asyncio
    async def test_fire_uses_engine_when_no_dispatcher(self):
        from nexus.triggers.manager import TriggerManager
        from nexus.types import WorkflowExecution, TriggerType, ChainStatus

        engine = AsyncMock()
        execution = _make_execution()
        engine.run_workflow = AsyncMock(return_value=execution)

        workflow_manager = AsyncMock()
        repository = AsyncMock()
        event_bus = AsyncMock()
        config = MagicMock()

        trigger = self._make_trigger()
        repository.update_trigger = AsyncMock(return_value=trigger)

        manager = TriggerManager(engine, workflow_manager, repository, event_bus, config)
        result = await manager.fire(trigger)

        engine.run_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_fire_enriches_trigger_data(self):
        from nexus.triggers.manager import TriggerManager

        engine = AsyncMock()
        execution = _make_execution()
        engine.run_workflow = AsyncMock(return_value=execution)

        workflow_manager = AsyncMock()
        repository = AsyncMock()
        event_bus = AsyncMock()
        config = MagicMock()

        trigger = self._make_trigger()
        repository.update_trigger = AsyncMock(return_value=trigger)

        manager = TriggerManager(engine, workflow_manager, repository, event_bus, config)
        await manager.fire(trigger, {"custom": "value"})

        call_kwargs = engine.run_workflow.call_args
        trigger_data = call_kwargs.kwargs.get("trigger_data", {})
        assert "_trigger_id" in trigger_data
        assert "_trigger_type" in trigger_data
        assert trigger_data.get("custom") == "value"

    @pytest.mark.asyncio
    async def test_fire_disabled_trigger_raises(self):
        from nexus.triggers.manager import TriggerManager
        from nexus.exceptions import TriggerError
        from nexus.types import TriggerConfig, TriggerType

        engine = AsyncMock()
        trigger = TriggerConfig(
            workflow_id="wf-1",
            tenant_id="demo",
            trigger_type=TriggerType.WEBHOOK,
            config={},
            enabled=False,
        )

        manager = TriggerManager(engine, AsyncMock(), AsyncMock(), AsyncMock(), MagicMock())
        with pytest.raises(TriggerError):
            await manager.fire(trigger)

    @pytest.mark.asyncio
    async def test_trigger_manager_accepts_dispatcher_kwarg(self):
        from nexus.triggers.manager import TriggerManager

        dispatcher = MagicMock()
        manager = TriggerManager(
            AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock(), MagicMock(),
            dispatcher=dispatcher,
        )
        assert manager._dispatcher is dispatcher

    def test_set_dispatcher_injects_late(self):
        """set_dispatcher() mirrors set_cron_scheduler() for post-construction injection."""
        from nexus.triggers.manager import TriggerManager

        manager = TriggerManager(
            AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock(), MagicMock()
        )
        assert manager._dispatcher is None

        dispatcher = MagicMock()
        manager.set_dispatcher(dispatcher)
        assert manager._dispatcher is dispatcher


# ─────────────────────────────────────────────────────────────────────────────
# WorkerSettings import guard
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkerSettings:

    def test_worker_settings_class_exists_or_none(self):
        """WorkerSettings is either a class (arq installed) or None (not installed)."""
        from nexus.workers.queue import WorkerSettings
        # Either arq is installed and WorkerSettings is a proper class,
        # or it's None due to ImportError
        assert WorkerSettings is None or isinstance(WorkerSettings, type)

    def test_task_functions_importable(self):
        from nexus.workers.queue import execute_workflow_task, refresh_mcp_connections_task
        assert callable(execute_workflow_task)
        assert callable(refresh_mcp_connections_task)

    def test_dispatcher_importable(self):
        from nexus.workers.dispatcher import WorkflowDispatcher
        assert WorkflowDispatcher is not None

    def test_jobs_router_importable(self):
        from nexus.api.routes.jobs import router
        assert router is not None
