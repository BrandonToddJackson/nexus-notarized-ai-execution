"""Base callback protocol for NEXUS lifecycle hooks.

Callbacks are called at key points in the agent execution lifecycle.
Implement this protocol to observe or instrument NEXUS without modifying core logic.

Usage:
    class MyCallback(NexusCallback):
        async def on_gate_check(self, gate_name, verdict, details, **kw):
            print(f"Gate {gate_name}: {verdict}")

    engine = NexusEngine(..., callbacks=[MyCallback()])
"""

from typing import Any, Protocol, runtime_checkable

from nexus.types import ChainPlan, GateResult, Seal


@runtime_checkable
class NexusCallback(Protocol):
    """Protocol defining hooks for NEXUS lifecycle events.

    All methods are optional — implement only the hooks you need.
    Unimplemented methods should return None (the default).

    All methods are async; the engine awaits each registered callback in order.
    """

    async def on_chain_start(
        self,
        chain: ChainPlan,
        **kwargs: Any,
    ) -> None:
        """Called when a new chain begins planning."""
        ...

    async def on_chain_complete(
        self,
        chain: ChainPlan,
        seals: list[Seal],
        **kwargs: Any,
    ) -> None:
        """Called when a chain finishes (completed, failed, or escalated)."""
        ...

    async def on_gate_check(
        self,
        gate_result: GateResult,
        seal_id: str,
        **kwargs: Any,
    ) -> None:
        """Called after each anomaly gate evaluates.

        Fires 4 times per seal: scope, intent, ttl, drift.
        """
        ...

    async def on_seal_create(
        self,
        seal: Seal,
        **kwargs: Any,
    ) -> None:
        """Called when a seal is finalized and appended to the ledger."""
        ...

    async def on_tool_execute(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        result: Any,
        seal_id: str,
        **kwargs: Any,
    ) -> None:
        """Called after a tool executes (whether successful or failed)."""
        ...

    async def on_error(
        self,
        error: Exception,
        context: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when an unhandled error occurs during chain execution."""
        ...


class BaseCallback:
    """Concrete base with no-op implementations of all hooks.

    Subclass this instead of implementing the Protocol directly
    to avoid implementing every method.
    """

    async def on_chain_start(self, chain: ChainPlan, **kwargs: Any) -> None:
        pass

    async def on_chain_complete(
        self, chain: ChainPlan, seals: list[Seal], **kwargs: Any
    ) -> None:
        pass

    async def on_gate_check(
        self, gate_result: GateResult, seal_id: str, **kwargs: Any
    ) -> None:
        pass

    async def on_seal_create(self, seal: Seal, **kwargs: Any) -> None:
        pass

    async def on_tool_execute(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        result: Any,
        seal_id: str,
        **kwargs: Any,
    ) -> None:
        pass

    async def on_error(
        self, error: Exception, context: dict[str, Any], **kwargs: Any
    ) -> None:
        pass
