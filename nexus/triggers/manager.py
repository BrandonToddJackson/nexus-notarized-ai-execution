"""TriggerManager — central registry for all trigger types.

Responsibilities:
- Create / delete / enable / disable triggers
- Route incoming events to the right trigger handler
- Fire workflows via NexusEngine
- Coordinate with CronScheduler for cron triggers
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from nexus.exceptions import TriggerError, WorkflowNotFound
from nexus.types import TriggerConfig, TriggerType, WorkflowExecution

logger = logging.getLogger(__name__)


class TriggerManager:
    """Manages the full lifecycle of workflow triggers."""

    def __init__(
        self,
        engine,          # NexusEngine — no circular import: plain type hint
        workflow_manager,
        repository,
        event_bus,
        config,
    ) -> None:
        self._engine          = engine
        self._workflow_manager = workflow_manager
        self._repository      = repository
        self._event_bus       = event_bus
        self._config          = config
        self._cron_scheduler  = None  # injected after construction

        # Maps trigger.id → subscriber callable (for EVENT / WORKFLOW_COMPLETE triggers)
        self._event_handlers: dict[str, Callable] = {}

    # ── Cron scheduler injection (avoids circular constructor args) ──────────

    def set_cron_scheduler(self, scheduler) -> None:
        self._cron_scheduler = scheduler

    # ── Create ───────────────────────────────────────────────────────────────

    async def create_trigger(
        self,
        tenant_id: str,
        workflow_id: str,
        trigger_type: TriggerType,
        config: dict[str, Any] | None = None,
    ) -> TriggerConfig:
        """Validate, persist, and activate a new trigger."""
        config = config or {}

        # Verify the workflow exists (raises WorkflowNotFound if not)
        await self._workflow_manager.get(workflow_id, tenant_id)

        # Type-specific validation
        webhook_path: Optional[str] = None
        if trigger_type == TriggerType.WEBHOOK:
            webhook_path = self._generate_webhook_path(tenant_id)
        elif trigger_type == TriggerType.CRON:
            self._validate_cron_config(config)
        elif trigger_type == TriggerType.EVENT:
            if not config.get("event_name"):
                raise TriggerError("EVENT trigger requires config.event_name")
        elif trigger_type == TriggerType.WORKFLOW_COMPLETE:
            if not config.get("source_workflow_id"):
                raise TriggerError("WORKFLOW_COMPLETE trigger requires config.source_workflow_id")

        trigger = TriggerConfig(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            trigger_type=trigger_type,
            config=config,
            webhook_path=webhook_path,
            enabled=True,
        )
        trigger = await self._repository.save_trigger(trigger)

        # Register live handlers for event-driven trigger types
        if trigger_type == TriggerType.EVENT and trigger.enabled:
            await self._register_event_trigger(trigger)
        elif trigger_type == TriggerType.WORKFLOW_COMPLETE and trigger.enabled:
            await self._register_workflow_complete_trigger(trigger)
        elif trigger_type == TriggerType.CRON and self._cron_scheduler is not None:
            await self._cron_scheduler.register(trigger)

        return trigger

    # ── Enable / Disable ─────────────────────────────────────────────────────

    async def enable(self, tenant_id: str, trigger_id: str) -> TriggerConfig:
        trigger = await self._repository.get_trigger(tenant_id, trigger_id)
        if trigger is None:
            raise TriggerError(f"Trigger '{trigger_id}' not found")
        trigger = trigger.model_copy(update={"enabled": True})
        trigger = await self._repository.update_trigger(trigger)

        # Re-attach live handlers
        if trigger.trigger_type == TriggerType.EVENT:
            if trigger_id not in self._event_handlers:
                await self._register_event_trigger(trigger)
        elif trigger.trigger_type == TriggerType.WORKFLOW_COMPLETE:
            if trigger_id not in self._event_handlers:
                await self._register_workflow_complete_trigger(trigger)
        elif trigger.trigger_type == TriggerType.CRON and self._cron_scheduler is not None:
            await self._cron_scheduler.register(trigger)

        return trigger

    async def disable(self, tenant_id: str, trigger_id: str) -> TriggerConfig:
        trigger = await self._repository.get_trigger(tenant_id, trigger_id)
        if trigger is None:
            raise TriggerError(f"Trigger '{trigger_id}' not found")
        trigger = trigger.model_copy(update={"enabled": False})
        trigger = await self._repository.update_trigger(trigger)

        # Detach live handlers
        self._detach_handler(trigger)
        if trigger.trigger_type == TriggerType.CRON and self._cron_scheduler is not None:
            self._cron_scheduler.unregister(trigger_id)

        return trigger

    # ── Delete ───────────────────────────────────────────────────────────────

    async def delete(self, tenant_id: str, trigger_id: str) -> bool:
        trigger = await self._repository.get_trigger(tenant_id, trigger_id)
        if trigger is None:
            raise TriggerError(f"Trigger '{trigger_id}' not found")

        self._detach_handler(trigger)
        if trigger.trigger_type == TriggerType.CRON and self._cron_scheduler is not None:
            self._cron_scheduler.unregister(trigger_id)

        deleted = await self._repository.delete_trigger(tenant_id, trigger_id)
        if not deleted:
            raise TriggerError(f"Failed to delete trigger '{trigger_id}'")
        return True

    # ── List ─────────────────────────────────────────────────────────────────

    async def list(
        self,
        tenant_id: str,
        workflow_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> list[TriggerConfig]:
        return await self._repository.list_triggers(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            enabled=enabled,
        )

    # ── Fire ─────────────────────────────────────────────────────────────────

    async def fire(
        self,
        trigger: TriggerConfig,
        trigger_data: dict[str, Any] | None = None,
    ) -> WorkflowExecution:
        """Execute the workflow associated with *trigger*.

        Injects trigger metadata into *trigger_data* and updates
        ``last_triggered_at`` regardless of execution outcome.
        """
        if not trigger.enabled:
            raise TriggerError(f"Trigger '{trigger.id}' is disabled")

        now = datetime.now(timezone.utc)

        payload: dict[str, Any] = dict(trigger_data or {})
        payload["_trigger_id"]   = trigger.id
        payload["_trigger_type"] = trigger.trigger_type.value
        payload["_fired_at"]     = now.isoformat()

        # Always update last_triggered_at, even if the engine raises
        updated = trigger.model_copy(update={"last_triggered_at": now})
        await self._repository.update_trigger(updated)

        execution = await self._engine.run_workflow(
            workflow_id=trigger.workflow_id,
            tenant_id=trigger.tenant_id,
            trigger_data=payload,
        )
        return execution

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _generate_webhook_path(self, tenant_id: str) -> str:
        import uuid
        return f"/webhooks/{tenant_id[:8]}/{uuid.uuid4().hex[:12]}"

    def _validate_cron_config(self, config: dict[str, Any]) -> None:
        expression = config.get("expression")
        if not expression:
            raise TriggerError("CRON trigger requires config.expression")
        try:
            from croniter import croniter
            if not croniter.is_valid(expression):
                raise TriggerError(f"Invalid cron expression: {expression!r}")
        except ImportError as exc:
            raise TriggerError("croniter is required for CRON triggers") from exc

    async def _register_event_trigger(self, trigger: TriggerConfig) -> None:
        event_name = trigger.config.get("event_name", "")

        async def _handler(data: Any) -> None:
            if not trigger.enabled:
                return
            current = await self._repository.get_trigger(trigger.tenant_id, trigger.id)
            if current is None or not current.enabled:
                return
            try:
                await self.fire(current, {"event": event_name, "event_data": data})
            except Exception:
                logger.exception("EVENT trigger '%s' fire failed", trigger.id)

        self._event_handlers[trigger.id] = _handler
        self._event_bus.subscribe(event_name, _handler)

    async def _register_workflow_complete_trigger(self, trigger: TriggerConfig) -> None:
        from nexus.triggers.event_bus import EVENT_WORKFLOW_COMPLETED
        source_id = trigger.config.get("source_workflow_id", "")

        async def _handler(data: Any) -> None:
            if data is None or data.get("workflow_id") != source_id:
                return
            current = await self._repository.get_trigger(trigger.tenant_id, trigger.id)
            if current is None or not current.enabled:
                return
            try:
                await self.fire(current, {
                    "source_workflow_id":  source_id,
                    "source_execution_id": data.get("execution_id"),
                    "source_result":       data.get("result"),
                })
            except Exception:
                logger.exception("WORKFLOW_COMPLETE trigger '%s' fire failed", trigger.id)

        self._event_handlers[trigger.id] = _handler
        self._event_bus.subscribe(EVENT_WORKFLOW_COMPLETED, _handler)

    def _detach_handler(self, trigger: TriggerConfig) -> None:
        """Unsubscribe any event handler stored for *trigger.id*."""
        handler = self._event_handlers.pop(trigger.id, None)
        if handler is None:
            return

        if trigger.trigger_type == TriggerType.EVENT:
            event_name = trigger.config.get("event_name", "")
            self._event_bus.unsubscribe(event_name, handler)
        elif trigger.trigger_type == TriggerType.WORKFLOW_COMPLETE:
            from nexus.triggers.event_bus import EVENT_WORKFLOW_COMPLETED
            self._event_bus.unsubscribe(EVENT_WORKFLOW_COMPLETED, handler)
