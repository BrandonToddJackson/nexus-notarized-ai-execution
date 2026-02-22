"""Structured JSON logging callback for NEXUS lifecycle events."""

import json
import logging
from datetime import datetime
from typing import Any

from nexus.callbacks.base import BaseCallback
from nexus.types import ChainPlan, GateResult, Seal

logger = logging.getLogger("nexus.audit")


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


class LoggingCallback(BaseCallback):
    """Emits structured JSON log lines for every lifecycle event.

    Each log line is a self-contained JSON object with:
      - event: event type name
      - ts: ISO-8601 UTC timestamp
      - relevant fields depending on event

    Log level: INFO for normal events, ERROR for errors.
    Logger name: nexus.audit (configure in your logging setup)

    **Integration note:** NexusEngine accepts callbacks as plain async callables
    ``async def cb(event: str, data: dict)``. This class implements both the
    named-method Protocol (for IDE completion / type checking) AND ``__call__``
    so an instance can be passed directly to NexusEngine:

        engine = NexusEngine(..., callbacks=[LoggingCallback()])
    """

    async def __call__(self, event: str, data: dict) -> None:
        """Dispatch engine lifecycle events to the appropriate named method."""
        if event == "chain_created":
            # Build a minimal ChainPlan-like object from the callback data
            from nexus.types import ChainPlan, ChainStatus
            chain = ChainPlan(
                id=data.get("chain_id", ""),
                tenant_id=data.get("tenant_id", ""),
                task=data.get("task", ""),
                steps=[{}] * data.get("steps", 0),
                status=ChainStatus.EXECUTING,
            )
            await self.on_chain_start(chain)
        elif event == "chain_completed":
            logger.info(json.dumps({
                "event": "chain_complete",
                "ts": _now(),
                "chain_id": data.get("chain_id", ""),
                "status": data.get("status", ""),
                "seal_count": data.get("seal_count", 0),
            }))
        elif event == "anomaly_checked":
            for gate in data.get("gates", []):
                logger.info(json.dumps({
                    "event": "gate_check",
                    "ts": _now(),
                    "seal_id": "",
                    "gate": gate.get("gate_name", ""),
                    "verdict": gate.get("verdict", ""),
                    "score": round(gate.get("score", 0.0), 4),
                    "threshold": gate.get("threshold", 0.0),
                    "details": gate.get("details", ""),
                }))
        elif event == "step_completed":
            logger.info(json.dumps({
                "event": "step_completed",
                "ts": _now(),
                "chain_id": data.get("chain_id", ""),
                "step_index": data.get("step_index", 0),
                "seal_id": data.get("seal_id", ""),
                "status": data.get("status", ""),
                "tool": data.get("tool", ""),
            }))
        elif event == "step_blocked":
            logger.info(json.dumps({
                "event": "step_blocked",
                "ts": _now(),
                "chain_id": data.get("chain_id", ""),
                "step_index": data.get("step_index", 0),
                "seal_id": data.get("seal_id", ""),
                "reason": data.get("reason", ""),
            }))
        elif event in ("chain_escalated", "task_started", "step_started"):
            logger.info(json.dumps({"event": event, "ts": _now(), **{
                k: (str(v)[:200] if not isinstance(v, (int, float, bool)) else v)
                for k, v in data.items()
            }}))

    async def on_chain_start(self, chain: ChainPlan, **kwargs: Any) -> None:
        logger.info(json.dumps({
            "event": "chain_start",
            "ts": _now(),
            "chain_id": chain.id,
            "tenant_id": chain.tenant_id,
            "task": chain.task[:200],
            "step_count": len(chain.steps),
        }))

    async def on_chain_complete(
        self, chain: ChainPlan, seals: list[Seal], **kwargs: Any
    ) -> None:
        logger.info(json.dumps({
            "event": "chain_complete",
            "ts": _now(),
            "chain_id": chain.id,
            "tenant_id": chain.tenant_id,
            "status": chain.status,
            "seal_count": len(seals),
        }))

    async def on_gate_check(
        self, gate_result: GateResult, seal_id: str, **kwargs: Any
    ) -> None:
        logger.info(json.dumps({
            "event": "gate_check",
            "ts": _now(),
            "seal_id": seal_id,
            "gate": gate_result.gate_name,
            "verdict": gate_result.verdict,
            "score": round(gate_result.score, 4),
            "threshold": gate_result.threshold,
            "details": gate_result.details,
        }))

    async def on_seal_create(self, seal: Seal, **kwargs: Any) -> None:
        logger.info(json.dumps({
            "event": "seal_create",
            "ts": _now(),
            "seal_id": seal.id,
            "chain_id": seal.chain_id,
            "tenant_id": seal.tenant_id,
            "persona_id": seal.persona_id,
            "tool": seal.tool_name,
            "status": seal.status,
            "fingerprint": seal.fingerprint[:16] + "..." if seal.fingerprint else "",
        }))

    async def on_tool_execute(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        result: Any,
        seal_id: str,
        **kwargs: Any,
    ) -> None:
        logger.info(json.dumps({
            "event": "tool_execute",
            "ts": _now(),
            "seal_id": seal_id,
            "tool": tool_name,
            "param_keys": list(tool_params.keys()),
            "result_type": type(result).__name__,
        }))

    async def on_error(
        self, error: Exception, context: dict[str, Any], **kwargs: Any
    ) -> None:
        logger.error(json.dumps({
            "event": "error",
            "ts": _now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "context": {k: str(v)[:200] for k, v in context.items()},
        }))


class WebhookCallback(BaseCallback):
    """[Beta] POSTs JSON payloads to a URL on configurable events.

    Deferred to post-v1: requires httpx dependency and retry logic.
    Use LoggingCallback for v1 audit trails.
    """

    def __init__(self, url: str, events: list[str] = None, secret: str = None):
        self.url = url
        self.events = set(events or ["chain_complete", "on_error"])
        self.secret = secret
