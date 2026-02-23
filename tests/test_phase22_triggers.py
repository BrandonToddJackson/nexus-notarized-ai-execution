"""Phase 22 — Trigger System tests.

No mocks.  Uses real in-memory stubs and asserts on actual returned state.
Pattern matches Phase 18 (CredentialVault in-memory dict) and
Phase 19 (real component wiring).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import pytest

from nexus.config import NexusConfig
from nexus.exceptions import TriggerError, WorkflowNotFound
from nexus.triggers import (
    EventBus,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_FAILED,
    EVENT_SEAL_BLOCKED,
    TriggerManager,
    WebhookHandler,
    CronScheduler,
)
from nexus.types import (
    TriggerConfig,
    TriggerType,
    WorkflowDefinition,
    WorkflowExecution,
    ChainStatus,
    WorkflowStatus,
)


# ── Fake infrastructure ───────────────────────────────────────────────────────

class FakeRepository:
    """In-memory trigger store — no DB, no SQLAlchemy session."""

    def __init__(self) -> None:
        self._triggers: dict[str, TriggerConfig] = {}

    async def save_trigger(self, trigger: TriggerConfig) -> TriggerConfig:
        self._triggers[trigger.id] = trigger
        return trigger

    async def get_trigger(self, tenant_id: str, trigger_id: str) -> Optional[TriggerConfig]:
        t = self._triggers.get(trigger_id)
        return t if (t is not None and t.tenant_id == tenant_id) else None

    async def get_trigger_by_webhook_path(self, path: str) -> Optional[TriggerConfig]:
        return next((t for t in self._triggers.values() if t.webhook_path == path), None)

    async def list_triggers(
        self,
        tenant_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> list[TriggerConfig]:
        out = list(self._triggers.values())
        if tenant_id is not None:
            out = [t for t in out if t.tenant_id == tenant_id]
        if workflow_id is not None:
            out = [t for t in out if t.workflow_id == workflow_id]
        if enabled is not None:
            out = [t for t in out if t.enabled == enabled]
        return out

    async def update_trigger(self, trigger: TriggerConfig) -> TriggerConfig:
        self._triggers[trigger.id] = trigger
        return trigger

    async def delete_trigger(self, tenant_id: str, trigger_id: str) -> bool:
        self._triggers.pop(trigger_id, None)
        return True


class FakeWorkflowManager:
    """Returns a real WorkflowDefinition or raises WorkflowNotFound."""

    def __init__(self, workflows: dict | None = None) -> None:
        self._workflows: dict[tuple, WorkflowDefinition] = workflows or {}

    async def get(self, workflow_id: str, tenant_id: str) -> WorkflowDefinition:
        wf = self._workflows.get((workflow_id, tenant_id))
        if wf is None:
            raise WorkflowNotFound(f"Workflow '{workflow_id}' not found")
        return wf


class FakeEngine:
    """Records run_workflow() calls; returns a real WorkflowExecution each time."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._raise: Exception | None = None

    def set_raise(self, exc: Exception) -> None:
        self._raise = exc

    async def run_workflow(
        self,
        workflow_id: str,
        tenant_id: str,
        trigger_data: dict | None = None,
    ) -> WorkflowExecution:
        self.calls.append({
            "workflow_id":   workflow_id,
            "tenant_id":     tenant_id,
            "trigger_data":  dict(trigger_data or {}),
        })
        if self._raise is not None:
            raise self._raise
        return WorkflowExecution(
            workflow_id=workflow_id,
            workflow_version=1,
            tenant_id=tenant_id,
            trigger_type=TriggerType.MANUAL,
            status=ChainStatus.COMPLETED,
        )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_repo() -> FakeRepository:
    return FakeRepository()


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def fake_wf() -> WorkflowDefinition:
    return WorkflowDefinition(
        id="wf-1",
        tenant_id="t1",
        name="Test WF",
        steps=[],
        edges=[],
        version=1,
        status=WorkflowStatus.ACTIVE,
    )


@pytest.fixture
def wf_manager(fake_wf: WorkflowDefinition) -> FakeWorkflowManager:
    return FakeWorkflowManager({("wf-1", "t1"): fake_wf})


@pytest.fixture
def trigger_manager(
    fake_engine: FakeEngine,
    wf_manager: FakeWorkflowManager,
    fake_repo: FakeRepository,
    event_bus: EventBus,
) -> TriggerManager:
    return TriggerManager(fake_engine, wf_manager, fake_repo, event_bus, NexusConfig())


# ── TestEventBus (12 tests) ───────────────────────────────────────────────────

class TestEventBus:
    def test_constants_defined(self) -> None:
        assert EVENT_WORKFLOW_COMPLETED == "workflow.completed"
        assert EVENT_WORKFLOW_FAILED    == "workflow.failed"
        assert EVENT_SEAL_BLOCKED       == "seal.blocked"

    async def test_subscribe_and_emit_sync(self) -> None:
        bus = EventBus()
        received: list = []
        bus.subscribe("test", lambda d: received.append(d))
        data = {"key": "value"}
        await bus.emit("test", data)
        assert received[0] == data

    async def test_subscribe_and_emit_async(self) -> None:
        bus = EventBus()
        received: list = []

        async def handler(d: Any) -> None:
            received.append(d)

        bus.subscribe("test", handler)
        await bus.emit("test", 42)
        assert received[0] == 42

    async def test_emit_to_empty_event_noop(self) -> None:
        bus = EventBus()
        # Must not raise
        await bus.emit("no_subscribers", {})

    async def test_duplicate_subscribe_called_twice(self) -> None:
        bus = EventBus()
        received: list = []
        cb = lambda d: received.append(d)  # noqa: E731
        bus.subscribe("e", cb)
        bus.subscribe("e", cb)
        await bus.emit("e", 1)
        assert len(received) == 2

    async def test_unsubscribe_removes_callback(self) -> None:
        bus = EventBus()
        received: list = []
        cb = lambda d: received.append(d)  # noqa: E731
        bus.subscribe("e", cb)
        bus.unsubscribe("e", cb)
        await bus.emit("e", 1)
        assert len(received) == 0

    async def test_unsubscribe_missing_silent(self) -> None:
        bus = EventBus()
        bus.unsubscribe("nonexistent", lambda d: None)  # must not raise

    async def test_error_in_subscriber_continues_to_next(self) -> None:
        bus = EventBus()
        received: list = []

        def bad_cb(d: Any) -> None:
            raise RuntimeError("boom")

        bus.subscribe("e", bad_cb)
        bus.subscribe("e", lambda d: received.append(d))
        await bus.emit("e", "data")
        assert len(received) == 1

    async def test_emit_snapshot_prevents_mutation_bug(self) -> None:
        """A callback that adds a new subscriber must not affect the current emit."""
        bus = EventBus()
        received: list = []

        def adding_cb(d: Any) -> None:
            bus.subscribe("e", lambda x: received.append(("extra", x)))

        bus.subscribe("e", adding_cb)
        await bus.emit("e", 1)
        # The newly added subscriber should NOT have been called this round
        assert received == []

    async def test_multiple_events_independent(self) -> None:
        bus = EventBus()
        a_received: list = []
        b_received: list = []
        bus.subscribe("a", lambda d: a_received.append(d))
        bus.subscribe("b", lambda d: b_received.append(d))
        await bus.emit("a", "hello")
        assert a_received == ["hello"]
        assert b_received == []

    async def test_multiple_subscribers_all_called(self) -> None:
        bus = EventBus()
        received: list = []
        for _ in range(3):
            bus.subscribe("e", lambda d: received.append(d))
        await bus.emit("e", "x")
        assert len(received) == 3

    async def test_emit_returns_none(self) -> None:
        bus = EventBus()
        result = await bus.emit("e", {})
        assert result is None


# ── TestTriggerManagerCreate (10 tests) ───────────────────────────────────────

class TestTriggerManagerCreate:
    async def test_webhook_trigger_generates_path(
        self, trigger_manager: TriggerManager
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        assert t.webhook_path is not None
        parts = t.webhook_path.split("/")  # ['', 'webhooks', 'tenant-prefix', 'token']
        assert len(parts) == 4
        assert parts[1] == "webhooks"
        assert parts[2].startswith("t1")
        assert len(parts[3]) > 0

    async def test_webhook_path_in_repo(
        self, trigger_manager: TriggerManager, fake_repo: FakeRepository
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        stored = fake_repo._triggers[t.id]
        assert stored.webhook_path == t.webhook_path

    async def test_cron_trigger_persisted(
        self, trigger_manager: TriggerManager, fake_repo: FakeRepository
    ) -> None:
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.CRON, config={"expression": "* * * * *"}
        )
        assert fake_repo._triggers[t.id].trigger_type == TriggerType.CRON

    async def test_invalid_cron_expression_raises(
        self, trigger_manager: TriggerManager
    ) -> None:
        with pytest.raises(TriggerError):
            await trigger_manager.create_trigger(
                "t1", "wf-1", TriggerType.CRON, config={"expression": "not-a-cron"}
            )

    async def test_missing_cron_expression_raises(
        self, trigger_manager: TriggerManager
    ) -> None:
        with pytest.raises(TriggerError):
            await trigger_manager.create_trigger("t1", "wf-1", TriggerType.CRON, config={})

    async def test_event_trigger_requires_event_name(
        self, trigger_manager: TriggerManager
    ) -> None:
        with pytest.raises(TriggerError):
            await trigger_manager.create_trigger("t1", "wf-1", TriggerType.EVENT, config={})

        # With event_name → should succeed
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.EVENT, config={"event_name": "my.event"}
        )
        assert t.trigger_type == TriggerType.EVENT

    async def test_workflow_complete_requires_source_id(
        self, trigger_manager: TriggerManager
    ) -> None:
        with pytest.raises(TriggerError):
            await trigger_manager.create_trigger(
                "t1", "wf-1", TriggerType.WORKFLOW_COMPLETE, config={}
            )

    async def test_manual_trigger_no_required_fields(
        self, trigger_manager: TriggerManager
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        assert t.trigger_type == TriggerType.MANUAL

    async def test_unknown_workflow_raises(
        self, trigger_manager: TriggerManager
    ) -> None:
        with pytest.raises(WorkflowNotFound):
            await trigger_manager.create_trigger("t1", "wf-unknown", TriggerType.MANUAL)

    async def test_event_trigger_subscribed_to_bus(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
        event_bus: EventBus,
    ) -> None:
        await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.EVENT, config={"event_name": "custom.event"}
        )
        await event_bus.emit("custom.event", {"some": "data"})
        await asyncio.sleep(0)
        assert len(fake_engine.calls) == 1

    async def test_event_trigger_data_has_event_key(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
        event_bus: EventBus,
    ) -> None:
        await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.EVENT, config={"event_name": "my.event"}
        )
        await event_bus.emit("my.event", {"payload": 42})
        await asyncio.sleep(0)
        td = fake_engine.calls[0]["trigger_data"]
        assert td["event"] == "my.event"
        assert td["event_data"] == {"payload": 42}


# ── TestTriggerManagerLifecycle (10 tests) ────────────────────────────────────

class TestTriggerManagerLifecycle:
    async def test_enable_sets_enabled_true(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        await trigger_manager.disable("t1", t.id)
        await trigger_manager.enable("t1", t.id)
        assert fake_repo._triggers[t.id].enabled is True

    async def test_disable_sets_enabled_false(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        await trigger_manager.disable("t1", t.id)
        assert fake_repo._triggers[t.id].enabled is False

    async def test_enable_unknown_trigger_raises(
        self, trigger_manager: TriggerManager
    ) -> None:
        with pytest.raises(TriggerError):
            await trigger_manager.enable("t1", "nonexistent-id")

    async def test_disable_unknown_trigger_raises(
        self, trigger_manager: TriggerManager
    ) -> None:
        with pytest.raises(TriggerError):
            await trigger_manager.disable("t1", "nonexistent-id")

    async def test_delete_removes_from_repo(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        await trigger_manager.delete("t1", t.id)
        assert fake_repo._triggers == {}

    async def test_delete_unknown_raises(
        self, trigger_manager: TriggerManager
    ) -> None:
        with pytest.raises(TriggerError):
            await trigger_manager.delete("t1", "no-such-id")

    async def test_list_all_for_tenant(
        self, trigger_manager: TriggerManager
    ) -> None:
        await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        # Add a second workflow for t2 (need to add it to wf_manager)
        trigger_manager._workflow_manager._workflows[("wf-1", "t2")] = WorkflowDefinition(
            id="wf-1", tenant_id="t2", name="WF T2", steps=[], edges=[], version=1,
            status=WorkflowStatus.ACTIVE,
        )
        await trigger_manager.create_trigger("t2", "wf-1", TriggerType.MANUAL)
        result = await trigger_manager.list("t1")
        assert len(result) == 2

    async def test_list_filtered_by_workflow(
        self, trigger_manager: TriggerManager, fake_repo: FakeRepository
    ) -> None:
        await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        # Add a second workflow
        trigger_manager._workflow_manager._workflows[("wf-2", "t1")] = WorkflowDefinition(
            id="wf-2", tenant_id="t1", name="WF2", steps=[], edges=[], version=1,
            status=WorkflowStatus.ACTIVE,
        )
        await trigger_manager.create_trigger("t1", "wf-2", TriggerType.MANUAL)
        result = await trigger_manager.list("t1", workflow_id="wf-1")
        assert len(result) == 1
        assert result[0].workflow_id == "wf-1"

    async def test_cron_disable_unregisters_scheduler(
        self, trigger_manager: TriggerManager, fake_repo: FakeRepository
    ) -> None:
        scheduler = CronScheduler(trigger_manager, NexusConfig(), tick_seconds=9999)
        trigger_manager.set_cron_scheduler(scheduler)
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.CRON, config={"expression": "* * * * *"}
        )
        assert t.id in scheduler._jobs
        await trigger_manager.disable("t1", t.id)
        assert t.id not in scheduler._jobs

    async def test_delete_event_trigger_unsubscribes_bus(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
        event_bus: EventBus,
    ) -> None:
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.EVENT, config={"event_name": "delete.test"}
        )
        await trigger_manager.delete("t1", t.id)
        await event_bus.emit("delete.test", {})
        await asyncio.sleep(0)
        assert fake_engine.calls == []


# ── TestTriggerManagerFire (8 tests) ──────────────────────────────────────────

class TestTriggerManagerFire:
    async def test_fire_injects_trigger_id(
        self, trigger_manager: TriggerManager, fake_engine: FakeEngine
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        await trigger_manager.fire(t)
        assert fake_engine.calls[0]["trigger_data"]["_trigger_id"] == t.id

    async def test_fire_injects_trigger_type(
        self, trigger_manager: TriggerManager, fake_engine: FakeEngine
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        await trigger_manager.fire(t)
        assert fake_engine.calls[0]["trigger_data"]["_trigger_type"] == "webhook"

    async def test_fire_injects_fired_at(
        self, trigger_manager: TriggerManager, fake_engine: FakeEngine
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        await trigger_manager.fire(t)
        fired_at_str = fake_engine.calls[0]["trigger_data"]["_fired_at"]
        # Must parse as ISO datetime
        parsed = datetime.fromisoformat(fired_at_str)
        assert isinstance(parsed, datetime)

    async def test_fire_updates_last_triggered_at(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        assert t.last_triggered_at is None
        await trigger_manager.fire(t)
        updated = fake_repo._triggers[t.id]
        assert updated.last_triggered_at is not None

    async def test_fire_persists_last_triggered_at(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        await trigger_manager.fire(t)
        assert fake_repo._triggers[t.id].last_triggered_at is not None

    async def test_fire_returns_workflow_execution(
        self, trigger_manager: TriggerManager
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        result = await trigger_manager.fire(t)
        assert isinstance(result, WorkflowExecution)
        assert result.workflow_id == "wf-1"

    async def test_fire_on_disabled_raises(
        self, trigger_manager: TriggerManager
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        disabled = t.model_copy(update={"enabled": False})
        with pytest.raises(TriggerError):
            await trigger_manager.fire(disabled)

    async def test_fire_propagates_engine_exception(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
        fake_repo: FakeRepository,
    ) -> None:
        fake_engine.set_raise(RuntimeError("engine boom"))
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.MANUAL)
        with pytest.raises(RuntimeError, match="engine boom"):
            await trigger_manager.fire(t)
        # last_triggered_at should still be set
        assert fake_repo._triggers[t.id].last_triggered_at is not None


# ── TestWebhookHandler (8 tests) ──────────────────────────────────────────────

class TestWebhookHandler:
    def _make_handler(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
    ) -> WebhookHandler:
        return WebhookHandler(trigger_manager, fake_repo)

    async def test_unknown_path_raises_trigger_error(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
    ) -> None:
        handler = self._make_handler(trigger_manager, fake_repo)
        with pytest.raises(TriggerError):
            await handler.handle("/webhooks/unknown/path")

    async def test_disabled_trigger_raises_trigger_error(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        # Manually disable in repo so webhook_path is still present
        fake_repo._triggers[t.id] = t.model_copy(update={"enabled": False})
        handler = self._make_handler(trigger_manager, fake_repo)
        with pytest.raises(TriggerError):
            await handler.handle(t.webhook_path)

    async def test_method_uppercased(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
        fake_engine: FakeEngine,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        handler = self._make_handler(trigger_manager, fake_repo)
        await handler.handle(t.webhook_path, method="get")
        assert fake_engine.calls[0]["trigger_data"]["method"] == "GET"

    async def test_headers_copied(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
        fake_engine: FakeEngine,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        handler = self._make_handler(trigger_manager, fake_repo)
        headers = {"X-Custom": "header-value"}
        await handler.handle(t.webhook_path, headers=headers)
        assert fake_engine.calls[0]["trigger_data"]["headers"] == headers

    async def test_body_passed_through(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
        fake_engine: FakeEngine,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        handler = self._make_handler(trigger_manager, fake_repo)
        body = {"payload": "data"}
        await handler.handle(t.webhook_path, body=body)
        assert fake_engine.calls[0]["trigger_data"]["body"] == body

    async def test_query_params_copied(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
        fake_engine: FakeEngine,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        handler = self._make_handler(trigger_manager, fake_repo)
        qp = {"ref": "main"}
        await handler.handle(t.webhook_path, query_params=qp)
        assert fake_engine.calls[0]["trigger_data"]["query_params"] == qp

    async def test_source_is_webhook(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
        fake_engine: FakeEngine,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        handler = self._make_handler(trigger_manager, fake_repo)
        await handler.handle(t.webhook_path)
        assert fake_engine.calls[0]["trigger_data"]["_source"] == "webhook"

    async def test_returns_workflow_execution(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        handler = self._make_handler(trigger_manager, fake_repo)
        result = await handler.handle(t.webhook_path)
        assert isinstance(result, WorkflowExecution)

    async def test_received_at_in_trigger_data(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
        fake_engine: FakeEngine,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        handler = self._make_handler(trigger_manager, fake_repo)
        await handler.handle(t.webhook_path)
        td = fake_engine.calls[0]["trigger_data"]
        assert "received_at" in td
        parsed = datetime.fromisoformat(td["received_at"])
        assert isinstance(parsed, datetime)


# ── TestCronScheduler (7 tests) ───────────────────────────────────────────────

class TestCronScheduler:
    def _make_scheduler(self, trigger_manager: TriggerManager) -> CronScheduler:
        return CronScheduler(trigger_manager, NexusConfig(), tick_seconds=9999)

    async def test_register_stores_job(
        self, trigger_manager: TriggerManager
    ) -> None:
        scheduler = self._make_scheduler(trigger_manager)
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.CRON, config={"expression": "* * * * *"}
        )
        await scheduler.register(t)
        assert t.id in scheduler._jobs

    async def test_unregister_removes_job(
        self, trigger_manager: TriggerManager
    ) -> None:
        scheduler = self._make_scheduler(trigger_manager)
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.CRON, config={"expression": "* * * * *"}
        )
        await scheduler.register(t)
        scheduler.unregister(t.id)
        assert t.id not in scheduler._jobs

    def test_unregister_missing_silent(self, trigger_manager: TriggerManager) -> None:
        scheduler = self._make_scheduler(trigger_manager)
        scheduler.unregister("no-such-id")  # must not raise

    async def test_check_and_fire_fires_due_trigger(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
    ) -> None:
        scheduler = self._make_scheduler(trigger_manager)
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.CRON, config={"expression": "* * * * *"}
        )
        await scheduler.register(t)
        # Force next_run into the past
        scheduler._jobs[t.id]["next_run"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        await scheduler.check_and_fire()
        assert len(fake_engine.calls) == 1

    async def test_check_and_fire_skips_future_trigger(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
    ) -> None:
        scheduler = self._make_scheduler(trigger_manager)
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.CRON, config={"expression": "* * * * *"}
        )
        await scheduler.register(t)
        # next_run is in the future (set by register)
        await scheduler.check_and_fire()
        assert fake_engine.calls == []

    async def test_next_run_advances_on_success(
        self,
        trigger_manager: TriggerManager,
    ) -> None:
        scheduler = self._make_scheduler(trigger_manager)
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.CRON, config={"expression": "* * * * *"}
        )
        await scheduler.register(t)
        # Set next_run to past and record that value as the baseline
        scheduler._jobs[t.id]["next_run"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        old_next = scheduler._jobs[t.id]["next_run"]
        await scheduler.check_and_fire()
        assert scheduler._jobs[t.id]["next_run"] > old_next

    async def test_next_run_advances_even_on_failure(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
    ) -> None:
        scheduler = self._make_scheduler(trigger_manager)
        fake_engine.set_raise(RuntimeError("engine down"))
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.CRON, config={"expression": "* * * * *"}
        )
        await scheduler.register(t)
        # Set next_run to past and record that value as the baseline
        scheduler._jobs[t.id]["next_run"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        old_next = scheduler._jobs[t.id]["next_run"]
        await scheduler.check_and_fire()  # engine raises but scheduler continues
        assert scheduler._jobs[t.id]["next_run"] > old_next

    async def test_cron_source_in_trigger_data(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
    ) -> None:
        scheduler = self._make_scheduler(trigger_manager)
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.CRON, config={"expression": "* * * * *"}
        )
        await scheduler.register(t)
        scheduler._jobs[t.id]["next_run"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        await scheduler.check_and_fire()
        td = fake_engine.calls[0]["trigger_data"]
        assert td["_source"] == "cron"
        assert "scheduled_at" in td


# ── TestPhase22Smoketest (5 tests) ────────────────────────────────────────────

class TestPhase22Smoketest:
    """Full component wiring, no shortcuts."""

    async def test_webhook_trigger_fires_workflow(
        self,
        trigger_manager: TriggerManager,
        fake_repo: FakeRepository,
        fake_engine: FakeEngine,
    ) -> None:
        t = await trigger_manager.create_trigger("t1", "wf-1", TriggerType.WEBHOOK)
        handler = WebhookHandler(trigger_manager, fake_repo)
        execution = await handler.handle(t.webhook_path)
        assert execution.status == ChainStatus.COMPLETED
        assert fake_engine.calls[0]["trigger_data"]["_source"] == "webhook"

    async def test_event_trigger_fires_on_bus_emit(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
        event_bus: EventBus,
    ) -> None:
        await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.EVENT, config={"event_name": "app.deploy"}
        )
        await event_bus.emit("app.deploy", {"key": "value"})
        await asyncio.sleep(0)
        td = fake_engine.calls[0]["trigger_data"]
        assert td["event"] == "app.deploy"
        assert td["event_data"]["key"] == "value"

    async def test_workflow_complete_trigger_chains_workflows(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
        event_bus: EventBus,
    ) -> None:
        # Add target workflow for chaining (wf-2)
        trigger_manager._workflow_manager._workflows[("wf-2", "t1")] = WorkflowDefinition(
            id="wf-2", tenant_id="t1", name="Chain WF", steps=[], edges=[], version=1,
            status=WorkflowStatus.ACTIVE,
        )
        await trigger_manager.create_trigger(
            "t1", "wf-2",
            TriggerType.WORKFLOW_COMPLETE,
            config={"source_workflow_id": "wf-1"},
        )
        await event_bus.emit(EVENT_WORKFLOW_COMPLETED, {
            "workflow_id": "wf-1",
            "execution_id": "exec-abc",
            "tenant_id": "t1",
        })
        await asyncio.sleep(0)
        assert len(fake_engine.calls) == 1
        assert fake_engine.calls[0]["workflow_id"] == "wf-2"
        td = fake_engine.calls[0]["trigger_data"]
        assert td["source_workflow_id"] == "wf-1"
        assert td["source_execution_id"] == "exec-abc"
        assert td["source_result"] is None  # not provided in emit payload

    async def test_cron_trigger_check_and_fire(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
        fake_repo: FakeRepository,
    ) -> None:
        scheduler = CronScheduler(trigger_manager, NexusConfig(), tick_seconds=9999)
        trigger_manager.set_cron_scheduler(scheduler)
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.CRON, config={"expression": "* * * * *"}
        )
        scheduler._jobs[t.id]["next_run"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        await scheduler.check_and_fire()
        assert len(fake_engine.calls) == 1
        assert fake_repo._triggers[t.id].last_triggered_at is not None

    async def test_disabled_trigger_not_fired_by_event(
        self,
        trigger_manager: TriggerManager,
        fake_engine: FakeEngine,
        event_bus: EventBus,
    ) -> None:
        t = await trigger_manager.create_trigger(
            "t1", "wf-1", TriggerType.EVENT, config={"event_name": "disabled.event"}
        )
        await trigger_manager.disable("t1", t.id)
        await event_bus.emit("disabled.event", {"some": "data"})
        await asyncio.sleep(0)
        assert fake_engine.calls == []
