"""Execution history, retry, and pin API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
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
        chain = ledger.get_chain(execution_id)
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
        chain = ledger.get_chain(execution_id)
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
