"""WebhookHandler — HTTP adapter that maps incoming requests to triggers.

This is a pure adapter: no business logic.  Routing and validation are
delegated to TriggerManager and the Repository.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Union

from nexus.exceptions import TriggerError
from nexus.types import WorkflowExecution


class WebhookHandler:
    """Resolves an inbound HTTP request to the matching trigger and fires it."""

    def __init__(self, trigger_manager, repository) -> None:
        self._trigger_manager = trigger_manager
        self._repository      = repository

    async def handle(
        self,
        webhook_path: str,
        method: str = "POST",
        headers: Optional[dict[str, str]] = None,
        query_params: Optional[dict[str, str]] = None,
        body: Optional[Union[dict[str, Any], str]] = None,
    ) -> WorkflowExecution:
        """Find the trigger for *webhook_path* and fire it.

        Args:
            webhook_path: Webhook path, e.g. ``/webhooks/abc12345/xyz098``.
            method:       HTTP method string (normalised to upper-case).
            headers:      Request headers dict.
            query_params: URL query parameters.
            body:         Parsed request body (dict, raw string, or None — passed through as-is).

        Raises:
            TriggerError: if no trigger matches *webhook_path* or the trigger is disabled.
        """
        trigger = await self._repository.get_trigger_by_webhook_path(webhook_path)
        if trigger is None:
            raise TriggerError("Unknown webhook path")
        if not trigger.enabled:
            raise TriggerError("Trigger is disabled")

        trigger_data: dict[str, Any] = {
            "_source":      "webhook",
            "method":       method.upper(),
            "headers":      dict(headers or {}),
            "query_params": dict(query_params or {}),
            "body":         body,
            "received_at":  datetime.now(timezone.utc).isoformat(),
        }

        return await self._trigger_manager.fire(trigger, trigger_data)
