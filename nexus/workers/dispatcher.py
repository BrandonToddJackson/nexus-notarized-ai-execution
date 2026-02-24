"""WorkflowDispatcher — routes workflow executions inline or to background queue."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Sources that always go to background
_BACKGROUND_SOURCES = {"webhook", "cron", "event", "schedule"}

# Import arq Job at module level so tests can patch nexus.workers.dispatcher.Job
try:
    from arq.jobs import Job
except ImportError:
    Job = None  # type: ignore[assignment,misc]


class WorkflowDispatcher:
    """Routes workflow execution inline (small/manual) or to ARQ background queue."""

    def __init__(self, engine, redis_pool, config) -> None:
        self._engine = engine
        self._redis = redis_pool
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    async def dispatch(
        self,
        workflow_id: str,
        tenant_id: str,
        trigger_data: dict[str, Any] | None = None,
        force_background: bool = False,
        workflow_manager=None,
        repository=None,
    ) -> dict:
        """Decide inline vs. background and execute accordingly.

        Decision tree:
        1. force_background=True → _enqueue
        2. trigger_data["_source"] in BACKGROUND_SOURCES → _enqueue
        3. workflow has >5 steps → _enqueue
        4. else → _run_inline
        """
        trigger_data = trigger_data or {}

        if force_background:
            return await self._enqueue(workflow_id, tenant_id, trigger_data)

        source = trigger_data.get("_source", "")
        if source in _BACKGROUND_SOURCES:
            return await self._enqueue(workflow_id, tenant_id, trigger_data)

        # Check step count if we have a workflow_manager
        if workflow_manager is not None:
            try:
                workflow = await workflow_manager.get(workflow_id, tenant_id)
                if len(workflow.steps) > 5:
                    return await self._enqueue(workflow_id, tenant_id, trigger_data)
            except Exception:
                pass  # If lookup fails, fall through to inline

        return await self._run_inline(workflow_id, tenant_id, trigger_data, repository=repository)

    async def get_job_status(self, job_id: str) -> dict:
        """Query ARQ for job status.

        Returns: {job_id, status, result, error}
        """
        try:
            if Job is None:
                raise ImportError("arq not installed")
            job = Job(job_id, redis=self._redis)
            status = await job.status()
            status_str = status.value if hasattr(status, "value") else str(status)

            if status_str in ("complete", "not_found"):
                result = None
                error = None
                if status_str == "complete":
                    try:
                        result = await job.result(timeout=0)
                    except Exception as exc:
                        error = str(exc)
                return {"job_id": job_id, "status": status_str, "result": result, "error": error}

            return {"job_id": job_id, "status": status_str, "result": None, "error": None}

        except Exception as exc:
            logger.warning("get_job_status failed for job_id=%s: %s", job_id, exc)
            return {"job_id": job_id, "status": "not_found", "result": None, "error": str(exc)}

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _enqueue(
        self,
        workflow_id: str,
        tenant_id: str,
        trigger_data: dict[str, Any],
    ) -> dict:
        """Enqueue the workflow task in ARQ and return job metadata."""
        job = await self._redis.enqueue_job(
            "execute_workflow_task",
            workflow_id,
            tenant_id,
            trigger_data,
        )
        job_id = job.job_id if job else None
        logger.info("Enqueued workflow=%s tenant=%s job_id=%s", workflow_id, tenant_id, job_id)
        return {"job_id": job_id, "status": "queued", "mode": "background"}

    async def _run_inline(
        self,
        workflow_id: str,
        tenant_id: str,
        trigger_data: dict[str, Any],
        repository=None,
    ) -> dict:
        """Execute the workflow synchronously in-process."""
        started = time.monotonic()
        execution = await self._engine.run_workflow(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            trigger_data=trigger_data,
        )
        duration_ms = int((time.monotonic() - started) * 1000)

        # Persist if we have a repository
        seal_count = 0
        if repository is not None:
            try:
                await repository.save_execution(execution)
                if execution.chain_id:
                    seals = await repository.get_chain_seals(execution.chain_id)
                    seal_count = len(seals)
            except Exception as exc:
                logger.warning("Failed to save inline execution: %s", exc)

        return {
            "execution_id": execution.id,
            "status": execution.status.value if hasattr(execution.status, "value") else str(execution.status),
            "seal_count": seal_count,
            "duration_ms": duration_ms,
            "mode": "inline",
        }
