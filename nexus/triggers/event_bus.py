"""In-process pub/sub event bus for NEXUS triggers.

Subscribers are plain callables (sync or async).  The bus snapshots the
subscriber list before iterating so that callbacks added during emit don't
cause mutation issues.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Well-known event names ─────────────────────────────────────────────────
EVENT_WORKFLOW_COMPLETED = "workflow.completed"
EVENT_WORKFLOW_FAILED    = "workflow.failed"
EVENT_SEAL_BLOCKED       = "seal.blocked"


class EventBus:
    """Lightweight, in-process pub/sub bus.

    Usage::

        bus = EventBus()
        bus.subscribe("workflow.completed", my_handler)
        await bus.emit("workflow.completed", {"workflow_id": "..."})
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = {}

    def subscribe(self, event: str, callback: Callable) -> None:
        """Register *callback* for *event*.  Same callback may be registered multiple times."""
        self._subscribers.setdefault(event, []).append(callback)

    def unsubscribe(self, event: str, callback: Callable) -> None:
        """Remove the first occurrence of *callback* from *event*.  Silently ignores missing."""
        callbacks = self._subscribers.get(event, [])
        try:
            callbacks.remove(callback)
        except ValueError:
            pass

    async def emit(self, event: str, data: Any = None) -> None:
        """Emit *event* to all subscribers, passing *data* as the sole argument.

        Exceptions raised by individual subscribers are logged and swallowed so
        that one failing handler cannot block the rest.
        """
        # Snapshot prevents mutation bugs if a callback subscribes/unsubscribes
        callbacks = list(self._subscribers.get(event, []))
        for cb in callbacks:
            try:
                result = cb(data)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("EventBus subscriber raised for event=%r", event)
