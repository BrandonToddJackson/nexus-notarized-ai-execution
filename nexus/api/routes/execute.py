"""POST /v1/execute — Primary NEXUS endpoint."""

import time
from fastapi import APIRouter, Request, HTTPException

from nexus.api.schemas import ExecuteRequest, ExecuteResponse

router = APIRouter(tags=["execute"])


@router.post("/execute", response_model=ExecuteResponse)
async def execute_task(request: Request, body: ExecuteRequest):
    """Execute a task through the NEXUS pipeline.

    Creates a chain, runs through anomaly gates, executes tools,
    and returns the full audit trail.

    Args:
        request: FastAPI request (has state.tenant_id from auth middleware)
        body: ExecuteRequest with task and optional persona

    Returns:
        ExecuteResponse with chain, seals, cost, and duration
    """
    start = time.time()

    # TODO: Get engine from app.state
    # TODO: Call engine.run(body.task, tenant_id, body.persona)
    # TODO: Convert chain + seals to ExecuteResponse

    raise HTTPException(status_code=501, detail="Execute endpoint not yet implemented")
