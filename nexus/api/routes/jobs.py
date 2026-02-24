"""Job status endpoint for background workflow executions."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["jobs"])


@router.get("/{job_id}")
async def get_job_status(job_id: str, request: Request) -> dict:
    """Get the status and result of a background workflow job."""
    dispatcher = request.app.state.dispatcher
    return await dispatcher.get_job_status(job_id)
