"""Execution history, retry, and pin API routes."""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["executions"])

# In-memory pin storage: execution_id → {step_id: output_data}
_pins: dict[str, dict[str, dict]] = {}


# ── Request schemas ───────────────────────────────────────────────────────────

class PinRequest(BaseModel):
    step_id: str
    output_data: dict


# ── Dependencies ──────────────────────────────────────────────────────────────

def _get_tenant(request: Request) -> str:
    tenant = getattr(request.state, "tenant_id", None)
    if not tenant:
        raise HTTPException(status_code=401, detail="Tenant not authenticated.")
    return tenant


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/executions")
async def list_executions(
    request: Request,
    workflow_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    cursor: Optional[str] = None,
    tenant_id: str = Depends(_get_tenant),
):
    """List executions with optional filters."""
    ledger = getattr(request.app.state, "ledger", None)
    if ledger is None:
        return {"executions": [], "total": 0}

    # Use chain list as execution proxy
    chains = ledger.list_chains(tenant_id) if hasattr(ledger, "list_chains") else []
    results = []
    for chain in chains:
        entry = {
            "id": chain.id if hasattr(chain, "id") else str(chain),
            "task": getattr(chain, "task", ""),
            "status": getattr(chain, "status", "unknown"),
            "created_at": str(getattr(chain, "created_at", "")),
        }
        if workflow_id and getattr(chain, "workflow_id", None) != workflow_id:
            continue
        if status and str(getattr(chain, "status", "")) != status:
            continue
        results.append(entry)

    return {"executions": results[:limit], "total": len(results)}


@router.get("/executions/{execution_id}")
async def get_execution(
    execution_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
):
    """Get execution detail with gate data from seals."""
    ledger = getattr(request.app.state, "ledger", None)
    if ledger is None:
        raise HTTPException(status_code=404, detail="Execution not found")

    chain = None
    if hasattr(ledger, "get_chain"):
        chain = await ledger.get_chain(execution_id)
    if chain is None:
        raise HTTPException(status_code=404, detail="Execution not found")

    seals = []
    if hasattr(ledger, "get_seals_for_chain"):
        seals = ledger.get_seals_for_chain(execution_id)

    seal_data = []
    for seal in seals:
        entry = seal.model_dump(mode="json") if hasattr(seal, "model_dump") else dict(seal)
        seal_data.append(entry)

    return {
        "id": chain.id,
        "task": getattr(chain, "task", ""),
        "status": str(getattr(chain, "status", "")),
        "seals": seal_data,
        "created_at": str(getattr(chain, "created_at", "")),
    }


@router.post("/executions/{execution_id}/retry", status_code=201)
async def retry_execution(
    execution_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
):
    """Create a new execution from the same input."""
    engine = getattr(request.app.state, "engine", None)
    ledger = getattr(request.app.state, "ledger", None)
    if engine is None or ledger is None:
        raise HTTPException(status_code=503, detail="Engine not available")

    chain = None
    if hasattr(ledger, "get_chain"):
        chain = await ledger.get_chain(execution_id)
    if chain is None:
        raise HTTPException(status_code=404, detail="Execution not found")

    task = getattr(chain, "task", "")
    if not task:
        raise HTTPException(status_code=422, detail="No task found to retry")

    result = await engine.execute(task=task, tenant_id=tenant_id)
    return result.model_dump(mode="json") if hasattr(result, "model_dump") else result


@router.get("/executions/{execution_id}/pins")
async def get_pins(
    execution_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
):
    pins = _pins.get(execution_id, {})
    return {"execution_id": execution_id, "pins": pins}


@router.post("/executions/{execution_id}/pins")
async def add_pin(
    execution_id: str,
    body: PinRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
):
    if execution_id not in _pins:
        _pins[execution_id] = {}
    _pins[execution_id][body.step_id] = body.output_data
    return {"pinned": True, "step_id": body.step_id}


@router.delete("/executions/{execution_id}/pins/{step_id}")
async def remove_pin(
    execution_id: str,
    step_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
):
    if execution_id in _pins:
        _pins[execution_id].pop(step_id, None)
    return {"unpinned": True, "step_id": step_id}


@router.delete("/executions/{execution_id}", status_code=204)
async def delete_execution(
    execution_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
):
    """Delete an execution record (only if not currently running)."""
    ledger = getattr(request.app.state, "ledger", None)
    if ledger is None:
        raise HTTPException(status_code=404, detail="Execution not found")

    chain = None
    if hasattr(ledger, "get_chain"):
        chain = await ledger.get_chain(execution_id)
    if chain is None:
        raise HTTPException(status_code=404, detail="Execution not found")

    status = str(getattr(chain, "status", ""))
    if status == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running execution")

    if hasattr(ledger, "delete_chain"):
        ledger.delete_chain(execution_id)
    # Remove any associated pins
    _pins.pop(execution_id, None)


@router.get("/executions/{execution_id}/stream")
async def stream_execution(
    execution_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
):
    """SSE stream for a specific execution. Replays seals then streams live events."""
    ledger = getattr(request.app.state, "ledger", None)
    event_bus = getattr(request.app.state, "event_bus", None)

    async def generate():
        # Replay existing seals first
        if ledger is not None and hasattr(ledger, "get_seals_for_chain"):
            seals = ledger.get_seals_for_chain(execution_id)
            for seal in seals:
                data = seal.model_dump(mode="json") if hasattr(seal, "model_dump") else dict(seal)
                yield f"data: {json.dumps({'type': 'seal', 'seal': data})}\n\n"

        # Check if execution is already complete
        if ledger is not None and hasattr(ledger, "get_chain"):
            chain = await ledger.get_chain(execution_id)
            if chain is not None:
                chain_status = str(getattr(chain, "status", ""))
                if chain_status not in ("running", "pending"):
                    yield f"data: {json.dumps({'type': 'complete', 'status': chain_status})}\n\n"
                    return

        # Stream live events from event_bus
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
                            if isinstance(event, dict) and event.get("execution_id") == execution_id:
                                yield f"data: {json.dumps(event)}\n\n"
                                if event.get("type") == "complete":
                                    break
                            elif isinstance(event, dict) and event.get("type") == "complete":
                                yield f"data: {json.dumps(event)}\n\n"
                                break
                        except asyncio.TimeoutError:
                            yield ": keepalive\n\n"
                finally:
                    if hasattr(event_bus, "unsubscribe"):
                        await event_bus.unsubscribe(sub_id)
                return

        # Fallback keepalive
        for _ in range(3):
            if await request.is_disconnected():
                break
            yield ": keepalive\n\n"
            await asyncio.sleep(10)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
