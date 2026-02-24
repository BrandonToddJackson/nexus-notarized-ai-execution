"""Phase 30 — Trigger system tests: EventBus, TriggerManager, WebhookHandler, CronScheduler."""

import pytest
from freezegun import freeze_time
from unittest.mock import AsyncMock, MagicMock

from nexus.triggers.event_bus import EventBus
from nexus.triggers.cron import CronScheduler
from nexus.triggers.manager import TriggerManager
from nexus.triggers.webhook import WebhookHandler
from nexus.exceptions import TriggerError
from nexus.types import TriggerType


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_mock_repo():
    repo = MagicMock()
    repo.save_trigger = AsyncMock(side_effect=lambda t: t)
    repo.list_triggers = AsyncMock(return_value=[])
    repo.update_trigger = AsyncMock(side_effect=lambda t: t)
    repo.delete_trigger = AsyncMock(return_value=True)
    repo.get_trigger = AsyncMock(return_value=None)
    repo.get_trigger_by_webhook_path = AsyncMock(return_value=None)
    return repo


def _make_trigger_manager(repo=None, event_bus=None):
    engine = MagicMock()
    engine.run_workflow = AsyncMock()
    workflow_manager = MagicMock()
    workflow_manager.get = AsyncMock(return_value=MagicMock())
    repo = repo or _make_mock_repo()
    bus = event_bus or EventBus()
    config = MagicMock()
    return TriggerManager(engine, workflow_manager, repo, bus, config), repo, bus


# ── EventBus ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_event_bus_subscribe_emit():
    """Subscribe a callback, emit event, callback is called with data."""
    bus = EventBus()
    received = []

    async def handler(data):
        received.append(data)

    bus.subscribe("test.event", handler)
    await bus.emit("test.event", {"key": "value"})

    assert len(received) == 1
    assert received[0] == {"key": "value"}


@pytest.mark.asyncio
async def test_event_bus_unsubscribe():
    """After unsubscribe, callback is NOT called on emit."""
    bus = EventBus()
    received = []

    async def handler(data):
        received.append(data)

    bus.subscribe("test.event", handler)
    bus.unsubscribe("test.event", handler)
    await bus.emit("test.event", {"key": "value"})

    assert len(received) == 0


# ── TriggerManager ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_webhook_trigger():
    """Creating a WEBHOOK trigger generates a webhook_path."""
    manager, repo, _ = _make_trigger_manager()

    trigger = await manager.create_trigger(
        tenant_id="t-001",
        workflow_id="wf-001",
        trigger_type=TriggerType.WEBHOOK,
    )

    assert trigger.webhook_path is not None
    assert trigger.webhook_path.startswith("/webhooks/")
    assert trigger.enabled is True
    repo.save_trigger.assert_awaited_once()


@pytest.mark.asyncio
async def test_trigger_enable_disable():
    """Disable sets enabled=False, enable sets enabled=True."""
    manager, repo, _ = _make_trigger_manager()

    trigger = await manager.create_trigger(
        tenant_id="t-001",
        workflow_id="wf-001",
        trigger_type=TriggerType.WEBHOOK,
    )

    # get_trigger must return the trigger for enable/disable to work
    repo.get_trigger = AsyncMock(return_value=trigger)

    disabled = await manager.disable("t-001", trigger.id)
    assert disabled.enabled is False

    repo.get_trigger = AsyncMock(return_value=disabled)
    enabled = await manager.enable("t-001", trigger.id)
    assert enabled.enabled is True


# ── WebhookHandler ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_webhook_handler_fires():
    """WebhookHandler.handle() fires the trigger via TriggerManager."""
    manager, repo, _ = _make_trigger_manager()

    trigger = await manager.create_trigger(
        tenant_id="t-001",
        workflow_id="wf-001",
        trigger_type=TriggerType.WEBHOOK,
    )

    # Wire up repository to return the trigger by webhook path
    repo.get_trigger_by_webhook_path = AsyncMock(return_value=trigger)
    # Wire up get_trigger for the fire() path (update_trigger needs it)
    repo.get_trigger = AsyncMock(return_value=trigger)

    handler = WebhookHandler(manager, repo)
    # Mock engine.run_workflow to return a mock execution
    manager._engine.run_workflow = AsyncMock(return_value=MagicMock())

    await handler.handle(
        webhook_path=trigger.webhook_path,
        method="POST",
        headers={"content-type": "application/json"},
        query_params={},
        body={"payload": "test"},
    )

    manager._engine.run_workflow.assert_awaited_once()


@pytest.mark.asyncio
async def test_webhook_not_found_raises():
    """WebhookHandler.handle() raises TriggerError for unknown path."""
    manager, repo, _ = _make_trigger_manager()

    # repo.get_trigger_by_webhook_path returns None by default
    handler = WebhookHandler(manager, repo)

    with pytest.raises(TriggerError, match="Unknown webhook path"):
        await handler.handle(
            webhook_path="/bad/path",
            method="POST",
        )


# ── CronScheduler ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cron_fires_overdue_job():
    """CronScheduler.check_and_fire() fires a trigger whose next_run has passed."""
    manager, repo, _ = _make_trigger_manager()

    # Create a cron trigger with "* * * * *" (every minute)
    trigger = await manager.create_trigger(
        tenant_id="t-001",
        workflow_id="wf-001",
        trigger_type=TriggerType.CRON,
        config={"expression": "* * * * *"},
    )

    cfg = MagicMock()
    cfg.cron_check_interval = 60
    scheduler = CronScheduler(manager, cfg)

    # Register the trigger at T=10:00:00 — next_run will be 10:01:00
    with freeze_time("2026-02-24 10:00:00+00:00"):
        await scheduler.register(trigger)

    # Patch manager.fire so we can observe it without real execution
    manager.fire = AsyncMock(return_value=None)

    # Advance to T=10:01:30 — past the scheduled run at 10:01:00
    with freeze_time("2026-02-24 10:01:30+00:00"):
        await scheduler.check_and_fire()

    manager.fire.assert_awaited_once()
    call_args = manager.fire.call_args
    fired_trigger = call_args[0][0]
    assert fired_trigger.id == trigger.id
