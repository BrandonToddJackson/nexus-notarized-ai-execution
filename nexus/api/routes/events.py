"""SSE event stream API route."""

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["events"])


@router.get("/events/stream")
async def event_stream(
    request: Request,
    token: str = "",
):
    """SSE stream for real-time execution updates.

    Auth via ?token= query param (JWT).
    Subscribes to app.state.event_bus if available.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Missing token query parameter")

    # Verify JWT token
    from nexus.auth.jwt import JWTManager
    jwt_manager = JWTManager()
    try:
        payload = await jwt_manager.verify_token(token)
        tenant_id = payload.get("tenant_id")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    event_bus = getattr(request.app.state, "event_bus", None)

    async def generate():
        # Send initial connection event
        data = json.dumps({"type": "connected", "tenant_id": tenant_id})
        yield f"data: {data}\n\n"

        if event_bus and hasattr(event_bus, "subscribe"):
            queue = asyncio.Queue()
            sub_id = None
            try:
                sub_id = await event_bus.subscribe(queue, tenant_id=tenant_id)
            except Exception:
                pass

            if sub_id:
                try:
                    while True:
                        if await request.is_disconnected():
                            break
                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=30.0)
                            data = json.dumps(event) if isinstance(event, dict) else str(event)
                            yield f"data: {data}\n\n"
                        except asyncio.TimeoutError:
                            yield ": keepalive\n\n"
                finally:
                    if hasattr(event_bus, "unsubscribe"):
                        await event_bus.unsubscribe(sub_id)
                return

        # Fallback: keepalive loop if no event bus
        while True:
            if await request.is_disconnected():
                break
            yield ": keepalive\n\n"
            await asyncio.sleep(30)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
