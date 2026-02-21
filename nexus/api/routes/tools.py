"""CRUD /v1/tools — Tool registry management."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["tools"])


@router.get("/tools")
async def list_tools(request: Request):
    """List all registered tools."""
    # TODO: Get tools from registry
    return {"tools": []}
