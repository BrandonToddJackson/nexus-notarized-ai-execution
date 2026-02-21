"""POST /v1/execute/stream — SSE real-time gate progression + token streaming.

SSE Event Format:
    event: chain_started       data: {"chain_id": "...", "steps": 3}
    event: step_started        data: {"step": 0, "persona": "...", "tool": "..."}
    event: gate_result         data: {"step": 0, "gate": "scope", "verdict": "pass", "score": 1.0}
    event: seal_created        data: {"step": 0, "seal_id": "...", "status": "executed"}
    event: step_completed      data: {"step": 0}
    event: chain_completed     data: {"chain_id": "...", "status": "completed", "cost": {...}}
"""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from nexus.api.schemas import ExecuteRequest

router = APIRouter(tags=["stream"])


@router.post("/execute/stream")
async def stream_execute(request: Request, body: ExecuteRequest):
    """Execute a task with SSE streaming of gate results.

    Sends real-time events as each gate checks, each seal is created,
    and each step completes.
    """
    # TODO: Implement SSE streaming
    # Use async generator that yields SSE-formatted events
    # Format: f"event: {event_type}\ndata: {json_data}\n\n"

    async def event_generator():
        yield "event: error\ndata: {\"message\": \"Streaming not yet implemented\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
