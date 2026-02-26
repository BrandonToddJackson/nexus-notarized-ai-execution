"""CronScheduler — asyncio loop that fires cron triggers on schedule."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from nexus.exceptions import TriggerError
from nexus.types import TriggerConfig, TriggerType

logger = logging.getLogger(__name__)

_DEFAULT_TICK_SECONDS = 60


class CronScheduler:
    """Polls registered cron triggers every *tick_seconds* and fires overdue ones.

    The scheduler does NOT rely on wall-clock time for correctness — it computes
    the next scheduled run using croniter and compares against *now* on each tick.
    """

    def __init__(self, trigger_manager, config, tick_seconds: int = _DEFAULT_TICK_SECONDS) -> None:
        self._trigger_manager = trigger_manager
        self._config          = config
        self._tick_seconds    = tick_seconds

        # Maps trigger_id → {"trigger": TriggerConfig, "next_run": datetime}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._task: asyncio.Task | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Load persisted cron triggers and start the background loop."""
        # Load all enabled cron triggers across all tenants
        triggers = await self._trigger_manager._repository.list_triggers(
            tenant_id=None, enabled=True
        )
        for t in triggers:
            if t.trigger_type == TriggerType.CRON:
                await self.register(t)

        self._task = asyncio.create_task(self._loop(), name="nexus-cron-scheduler")
        logger.info("CronScheduler started with %d jobs", len(self._jobs))

    async def stop(self) -> None:
        """Cancel the background loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("CronScheduler stopped")

    # ── Job management ───────────────────────────────────────────────────────

    async def register(self, trigger: TriggerConfig) -> None:
        """Add (or replace) a cron job for *trigger*."""
        next_run = self._compute_next_run(trigger)
        self._jobs[trigger.id] = {"trigger": trigger, "next_run": next_run}
        logger.debug("CronScheduler registered trigger=%s next_run=%s", trigger.id, next_run)

    def unregister(self, trigger_id: str) -> None:
        """Remove the cron job for *trigger_id*.  Silently ignores missing IDs."""
        self._jobs.pop(trigger_id, None)

    # ── Core tick ────────────────────────────────────────────────────────────

    async def check_and_fire(self) -> None:
        """Check all registered jobs and fire any that are overdue (single tick)."""
        now = datetime.now(timezone.utc)
        for trigger_id, job in list(self._jobs.items()):
            trigger  = job["trigger"]
            next_run = job["next_run"]

            if now < next_run:
                continue

            # Advance next_run BEFORE firing so a slow/failing run doesn't
            # cause the job to fire again on the next tick.
            new_next = self._compute_next_run(trigger, after=now)
            self._jobs[trigger_id]["next_run"] = new_next

            try:
                await self._trigger_manager.fire(trigger, {
                    "scheduled_at":    next_run.isoformat(),
                    "_source":         "cron",
                    "cron_expression": trigger.config.get("expression", ""),
                })
            except TriggerError as exc:
                logger.warning("Cron trigger '%s' fire blocked: %s", trigger_id, exc)
            except Exception:
                logger.exception("Cron trigger '%s' fire raised", trigger_id)

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while True:
            interval = self._tick_seconds if self._tick_seconds != _DEFAULT_TICK_SECONDS \
                else self._config.cron_check_interval
            await asyncio.sleep(interval)
            try:
                await self.check_and_fire()
            except Exception:
                logger.exception("CronScheduler tick raised unexpectedly")

    def _compute_next_run(
        self,
        trigger: TriggerConfig,
        after: datetime | None = None,
    ) -> datetime:
        from croniter import croniter
        expression = trigger.config.get("expression", "* * * * *")
        base = after or datetime.now(timezone.utc)
        # croniter returns tz-naive — force UTC
        cron = croniter(expression, base)
        naive = cron.get_next(datetime)
        return naive.replace(tzinfo=timezone.utc)
