"""Standalone entry point for the NEXUS CronScheduler.

Runs as a singleton process (enforced by Redis distributed lock).
Start with:  python -m nexus.triggers.cron_runner

The lock prevents duplicate cron fires when the container restarts or when
a second instance is accidentally started.  The healthcheck key
``nexus:scheduler:heartbeat`` lets Docker/Kubernetes verify liveness.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Redis lock constants ────────────────────────────────────────────────────

LOCK_KEY        = "nexus:scheduler:lock"
HEARTBEAT_KEY   = "nexus:scheduler:heartbeat"
LOCK_TTL_S      = 30   # lock expires after 30 s if the holder crashes
RENEW_INTERVAL  = 10   # renew lock + write heartbeat every 10 s
RETRY_INTERVAL  = 5    # retry lock acquisition every 5 s when another instance holds it

# Lua CAS renew: only renew if we still own the lock
_LUA_RENEW = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('EXPIRE', KEYS[1], ARGV[2])
else
    return 0
end
"""

# Lua CAS release: only delete if we still own the lock
_LUA_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""


class _CronRepository:
    """Session-per-call proxy — safe for a long-running scheduler process.

    The Repository class takes a single AsyncSession; keeping one session open
    for the lifetime of the process risks stale connections and transaction
    timeouts.  This proxy opens a fresh session for every DB call instead.

    Only the methods called by CronScheduler and TriggerManager during cron
    operation are forwarded here.
    """

    def __init__(self, session_factory) -> None:
        self._sf = session_factory

    async def list_triggers(
        self,
        tenant_id: Optional[str],
        workflow_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        from nexus.db.repository import Repository
        async with self._sf() as session:
            return await Repository(session).list_triggers(
                tenant_id, workflow_id=workflow_id, enabled=enabled
            )

    async def get_trigger(self, tenant_id: str, trigger_id: str):
        from nexus.db.repository import Repository
        async with self._sf() as session:
            return await Repository(session).get_trigger(tenant_id, trigger_id)

    async def save_trigger(self, trigger: Any):
        from nexus.db.repository import Repository
        async with self._sf() as session:
            return await Repository(session).save_trigger(trigger)

    async def update_trigger(self, trigger: Any):
        from nexus.db.repository import Repository
        async with self._sf() as session:
            return await Repository(session).update_trigger(trigger)

    async def delete_trigger(self, tenant_id: str, trigger_id: str):
        from nexus.db.repository import Repository
        async with self._sf() as session:
            return await Repository(session).delete_trigger(tenant_id, trigger_id)


async def run() -> None:
    """Main coroutine — initialise dependencies and run the scheduler loop."""
    import redis.asyncio as aioredis
    import arq

    from nexus.config import config
    from nexus.db.database import async_session
    from nexus.triggers.event_bus import EventBus
    from nexus.workflows.manager import WorkflowManager
    from nexus.workers.dispatcher import WorkflowDispatcher
    from nexus.triggers.manager import TriggerManager
    from nexus.triggers.cron import CronScheduler

    instance_id = str(uuid.uuid4())
    stop_event  = asyncio.Event()

    # ── Signal handling (asyncio-safe via loop.add_signal_handler) ──────────

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(signal.SIGINT,  stop_event.set)

    # ── Dependency chain ────────────────────────────────────────────────────

    # _CronRepository opens a fresh session per call — safe for long-running processes.
    repository = _CronRepository(async_session)

    event_bus        = EventBus()
    workflow_manager = WorkflowManager(repository=repository, config=config)

    # ARQ pool for the task queue (separate Redis DB from the lock/heartbeat)
    arq_pool = await arq.create_pool(
        arq.connections.RedisSettings.from_dsn(config.task_queue_url)
    )

    # engine=None is safe: cron-fired workflows always have _source="cron",
    # which WorkflowDispatcher routes to ARQ (never inline).
    dispatcher = WorkflowDispatcher(
        engine=None,
        redis_pool=arq_pool,
        config=config,
    )

    trigger_manager = TriggerManager(
        engine=None,
        workflow_manager=workflow_manager,
        repository=repository,
        event_bus=event_bus,
        config=config,
        dispatcher=dispatcher,
    )

    tick_seconds = config.cron_check_interval
    scheduler    = CronScheduler(trigger_manager, config, tick_seconds=tick_seconds)
    trigger_manager.set_cron_scheduler(scheduler)

    # ── Redis connection for lock + heartbeat ───────────────────────────────

    redis_client = await aioredis.from_url(
        config.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )

    # Pre-compile Lua scripts
    script_renew   = redis_client.register_script(_LUA_RENEW)
    script_release = redis_client.register_script(_LUA_RELEASE)

    # ── Lock helpers ────────────────────────────────────────────────────────

    async def acquire_lock() -> bool:
        # SET NX EX is atomic — no Lua needed for acquire
        result = await redis_client.set(LOCK_KEY, instance_id, nx=True, ex=LOCK_TTL_S)
        return result is not None  # True = "OK", None = already set

    async def renew_lock() -> bool:
        result = await script_renew(keys=[LOCK_KEY], args=[instance_id, LOCK_TTL_S])
        return bool(result)

    async def release_lock() -> None:
        await script_release(keys=[LOCK_KEY], args=[instance_id])
        logger.info("Scheduler lock released (instance=%s)", instance_id)

    # Wait until we own the lock
    while not stop_event.is_set():
        if await acquire_lock():
            logger.info("Scheduler lock acquired (instance=%s)", instance_id)
            break
        holder = await redis_client.get(LOCK_KEY)
        logger.info("Lock held by %s — retrying in %ds", holder, RETRY_INTERVAL)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=float(RETRY_INTERVAL))
        except asyncio.TimeoutError:
            pass

    if stop_event.is_set():
        logger.info("Stop requested before lock acquired — exiting cleanly")
        await redis_client.aclose()
        await arq_pool.aclose()
        return

    # ── Heartbeat + renew loop (runs concurrently with scheduler) ───────────

    async def _heartbeat_loop() -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=float(RENEW_INTERVAL))
            except asyncio.TimeoutError:
                pass
            if stop_event.is_set():
                break
            still_owner = await renew_lock()
            if not still_owner:
                logger.error(
                    "Scheduler lock lost (instance=%s) — another process took over. "
                    "Stopping this instance to prevent duplicate cron fires.",
                    instance_id,
                )
                stop_event.set()
                break
            await redis_client.set(HEARTBEAT_KEY, instance_id, ex=LOCK_TTL_S * 2)
            logger.debug("Scheduler heartbeat written (instance=%s)", instance_id)

    # ── Run ─────────────────────────────────────────────────────────────────

    await scheduler.start()
    # Write initial heartbeat immediately so Docker healthcheck passes fast
    await redis_client.set(HEARTBEAT_KEY, instance_id, ex=LOCK_TTL_S * 2)

    heartbeat_task = asyncio.create_task(_heartbeat_loop(), name="nexus-scheduler-heartbeat")

    logger.info(
        "CronScheduler running (instance=%s, tick=%ds)",
        instance_id,
        tick_seconds,
    )

    # Block until a signal fires stop_event
    await stop_event.wait()

    # ── Graceful shutdown ───────────────────────────────────────────────────

    logger.info("Shutting down CronScheduler…")
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

    await scheduler.stop()
    await release_lock()
    await redis_client.aclose()
    await arq_pool.aclose()
    logger.info("CronScheduler shut down cleanly")


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("NEXUS_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()
