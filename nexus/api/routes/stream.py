"""POST /v1/execute/stream — SSE real-time gate progression + token streaming.

SSE Event Format:
    event: chain_started       data: {"chain_id": "...", "steps": 3}
    event: step_started        data: {"step": 0, "persona": "...", "tool": "..."}
    event: gate_result         data: {"step": 0, "gate": "scope", "verdict": "pass", "score": 1.0}
    event: seal_created        data: {"step": 0, "seal_id": "...", "status": "executed"}
    event: step_completed      data: {"step": 0}
    event: chain_completed     data: {"chain_id": "...", "status": "completed", "cost": {...}}
"""

import asyncio
import json
import logging
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from nexus.api.schemas import ExecuteRequest
from nexus.core.engine import NexusEngine
from nexus.exceptions import AnomalyDetected, ChainAborted, EscalationRequired

logger = logging.getLogger(__name__)
router = APIRouter(tags=["stream"])


def _sse(event_type: str, data: dict) -> str:
    """Format a server-sent event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post("/execute/stream")
async def stream_execute(request: Request, body: ExecuteRequest):
    """Execute a task with SSE streaming of gate results."""
    engine: NexusEngine = request.app.state.engine
    tenant_id = getattr(request.state, "tenant_id", "demo")

    # Queue for callback → generator communication
    queue: asyncio.Queue = asyncio.Queue()

    async def engine_callback(event: str, data: dict) -> None:
        """Bridge engine lifecycle events to SSE queue."""
        if event == "chain_created":
            await queue.put(_sse("chain_started", {
                "chain_id": data["chain_id"],
                "steps": data["steps"],
            }))

        elif event == "step_started":
            step = data.get("step", {})
            await queue.put(_sse("step_started", {
                "step": data["step_index"],
                "persona": data.get("persona", ""),
                "tool": step.get("tool", ""),
            }))

        elif event == "anomaly_checked":
            for gate in data.get("gates", []):
                await queue.put(_sse("gate_result", {
                    "step": data["step_index"],
                    "gate": gate.get("gate_name", ""),
                    "verdict": gate.get("verdict", ""),
                    "score": gate.get("score", 0.0),
                    "threshold": gate.get("threshold", 0.0),
                }))

        elif event == "step_completed":
            # Emit seal_created before step_completed
            await queue.put(_sse("seal_created", {
                "step": data["step_index"],
                "seal_id": data.get("seal_id", ""),
                "status": data.get("status", ""),
                "result_preview": str(data.get("result", ""))[:200] if data.get("result") else "",
            }))
            await queue.put(_sse("step_completed", {"step": data["step_index"]}))

        elif event == "step_blocked":
            await queue.put(_sse("seal_created", {
                "step": data["step_index"],
                "seal_id": data.get("seal_id", ""),
                "status": "blocked",
                "result_preview": data.get("reason", ""),
            }))
            await queue.put(_sse("step_completed", {"step": data["step_index"]}))

        elif event == "chain_completed":
            await queue.put(_sse("chain_completed", {
                "chain_id": data["chain_id"],
                "status": data["status"],
                "cost": {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0},
            }))
            await queue.put(None)  # sentinel: done

        elif event in ("chain_escalated",):
            await queue.put(_sse("chain_completed", {
                "chain_id": data["chain_id"],
                "status": "escalated",
                "cost": {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0},
            }))
            await queue.put(None)

    async def event_generator():
        # Run engine in a background task so we can yield from the queue concurrently
        async def run_engine():
            try:
                # Create a fresh engine instance with our callback attached
                # (engine.callbacks is a list — we don't mutate the shared engine)
                from nexus.core.engine import NexusEngine
                stream_engine = NexusEngine(
                    persona_manager=engine.persona_manager,
                    anomaly_engine=engine.anomaly_engine,
                    notary=engine.notary,
                    ledger=engine.ledger,
                    chain_manager=engine.chain_manager,
                    context_builder=engine.context_builder,
                    tool_registry=engine.tool_registry,
                    tool_selector=engine.tool_selector,
                    tool_executor=engine.tool_executor,
                    output_validator=engine.output_validator,
                    cot_logger=engine.cot_logger,
                    think_act_gate=engine.think_act_gate,
                    continue_complete_gate=engine.continue_complete_gate,
                    escalate_gate=engine.escalate_gate,
                    llm_client=engine.llm_client,
                    cost_tracker=engine.cost_tracker,
                    config=engine.config,
                    callbacks=[engine_callback],
                )
                await stream_engine.run(
                    task=body.task,
                    tenant_id=tenant_id,
                    persona_name=body.persona,
                )
            except (AnomalyDetected, ChainAborted, EscalationRequired) as exc:
                await queue.put(_sse("error", {"message": str(exc)}))
                await queue.put(None)
            except Exception as exc:
                logger.error(f"[stream] Engine error: {exc}", exc_info=True)
                await queue.put(_sse("error", {"message": str(exc)}))
                await queue.put(None)

        task = asyncio.create_task(run_engine())

        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=120.0)
                if item is None:
                    break
                yield item
        except asyncio.TimeoutError:
            yield _sse("error", {"message": "Stream timed out"})
        finally:
            task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
