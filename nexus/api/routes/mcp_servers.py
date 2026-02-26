"""MCP server management API routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from nexus.exceptions import MCPConnectionError
from nexus.types import MCPServerConfig

logger = logging.getLogger(__name__)
router = APIRouter(tags=["mcp"])


# ── Request schemas ───────────────────────────────────────────────────────────

class MCPServerCreateRequest(BaseModel):
    name: str
    url: str = ""
    transport: str = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    credential_id: str = ""


class MCPServerTestRequest(BaseModel):
    name: str
    url: str = ""
    transport: str = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)


# ── Dependencies ──────────────────────────────────────────────────────────────

def _get_tenant(request: Request) -> str:
    tenant = getattr(request.state, "tenant_id", None)
    if not tenant:
        raise HTTPException(status_code=401, detail="Tenant not authenticated.")
    return tenant


def _get_adapter(request: Request):
    adapter = getattr(request.app.state, "mcp_adapter", None)
    if adapter is None:
        raise HTTPException(status_code=503, detail="MCPToolAdapter not initialised.")
    return adapter


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/mcp/servers")
async def list_servers(
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    adapter=Depends(_get_adapter),
):
    servers = adapter.list_servers(tenant_id)
    return {"servers": [s.model_dump(mode="json") for s in servers]}


@router.post("/mcp/servers", status_code=201)
async def add_server(
    body: MCPServerCreateRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    adapter=Depends(_get_adapter),
):
    config = MCPServerConfig(
        tenant_id=tenant_id,
        name=body.name,
        url=body.url,
        transport=body.transport,
        command=body.command or None,
        args=body.args,
        env=body.env,
        credential_id=body.credential_id or None,
    )
    try:
        tools = await adapter.register_server(tenant_id, config)
    except MCPConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {
        "server": config.model_dump(mode="json"),
        "tools_registered": len(tools),
    }


@router.delete("/mcp/servers/{server_id}")
async def remove_server(
    server_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    adapter=Depends(_get_adapter),
):
    server = adapter.get_server(server_id)
    if server is None or server.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="MCP server not found")
    await adapter.unregister_server(server_id)
    return {"deleted": True}


@router.get("/mcp/servers/{server_id}/tools")
async def list_server_tools(
    server_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    adapter=Depends(_get_adapter),
):
    server = adapter.get_server(server_id)
    if server is None or server.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="MCP server not found")
    tool_names = adapter._server_tools.get(server_id, [])
    return {"server_id": server_id, "tools": tool_names}


@router.post("/mcp/servers/test")
async def test_server(
    body: MCPServerTestRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
):
    from nexus.mcp.client import MCPClient
    client = MCPClient()
    config = MCPServerConfig(
        tenant_id=tenant_id,
        name=body.name,
        url=body.url,
        transport=body.transport,
        command=body.command or None,
        args=body.args,
    )
    try:
        tools = await client.connect(config)
        await client.disconnect(config.id)
        return {"success": True, "tools_discovered": len(tools)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.post("/mcp/servers/{server_id}/reconnect")
async def reconnect_server(
    server_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    adapter=Depends(_get_adapter),
):
    server = adapter.get_server(server_id)
    if server is None or server.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="MCP server not found")
    # Unregister and re-register
    await adapter.unregister_server(server_id)
    try:
        tools = await adapter.register_server(tenant_id, server)
    except MCPConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"reconnected": True, "tools_registered": len(tools)}


@router.post("/mcp/servers/{server_id}/refresh")
async def refresh_server(
    server_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    adapter=Depends(_get_adapter),
):
    """Refresh tool discovery for an MCP server without full reconnect."""
    server = adapter.get_server(server_id)
    if server is None or server.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="MCP server not found")
    # Re-discover tools by reconnecting
    await adapter.unregister_server(server_id)
    try:
        tools = await adapter.register_server(tenant_id, server)
    except MCPConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"refreshed": True, "server_id": server_id, "tools_registered": len(tools)}
