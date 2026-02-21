"""GET /v1/tools — Tool registry listing."""

from fastapi import APIRouter, Request
from nexus.api.schemas import ToolResponse

router = APIRouter(tags=["tools"])


@router.get("/tools")
async def list_tools(request: Request):
    """List all registered tools."""
    tool_registry = request.app.state.tool_registry
    tools = tool_registry.list_tools()
    return {
        "tools": [
            ToolResponse(
                name=t.name,
                description=t.description,
                risk_level=t.risk_level.value if hasattr(t.risk_level, "value") else t.risk_level,
                resource_pattern=t.resource_pattern,
                requires_approval=t.requires_approval,
            )
            for t in tools
        ]
    }
