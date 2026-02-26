"""Phase 30 — MCP client and adapter tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from nexus.mcp.client import MCPClient, make_tool_name
from nexus.mcp.adapter import MCPToolAdapter
from nexus.exceptions import ToolError
from nexus.tools.registry import ToolRegistry
from nexus.types import MCPServerConfig, ToolDefinition, RiskLevel

TENANT_A = "tenant-alpha-001"


def _make_server_config(name: str = "test-server") -> MCPServerConfig:
    return MCPServerConfig(
        tenant_id=TENANT_A,
        name=name,
        url="",
        transport="stdio",
        command="echo",
    )


def _make_tool_def(server_name: str, tool_name: str) -> ToolDefinition:
    return ToolDefinition(
        name=make_tool_name(server_name, tool_name),
        description=f"MCP tool '{tool_name}'",
        parameters={"type": "object", "properties": {}},
        risk_level=RiskLevel.MEDIUM,
        resource_pattern=f"mcp:{server_name}:*",
        timeout_seconds=30,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_returns_tool_definitions():
    """Mocked MCPClient.connect returns ToolDefinitions via adapter.register_server."""
    registry = ToolRegistry()
    client = MCPClient()
    adapter = MCPToolAdapter(registry=registry, client=client)

    expected_tools = [_make_tool_def("test-server", "search")]
    client.connect = AsyncMock(return_value=expected_tools)

    cfg = _make_server_config()
    result = await adapter.register_server(TENANT_A, cfg)

    assert len(result) == 1
    assert result[0].name == "mcp_test_server_search"


@pytest.mark.asyncio
async def test_tool_name_normalization():
    """Tool names with spaces/special chars are normalized to underscores."""
    assert make_tool_name("my server", "read file!") == "mcp_my_server_read_file_"
    assert make_tool_name("srv-1", "do.thing") == "mcp_srv_1_do_thing"


@pytest.mark.asyncio
async def test_adapter_register_server():
    """register_server adds tools to the registry; impl closure routes to MCPClient.call_tool."""
    registry = ToolRegistry()
    client = MCPClient()
    adapter = MCPToolAdapter(registry=registry, client=client)

    tools = [_make_tool_def("test-server", "lookup")]
    client.connect = AsyncMock(return_value=tools)
    client.call_tool = AsyncMock(return_value={"result": "found"})

    cfg = _make_server_config()
    await adapter.register_server(TENANT_A, cfg)

    defn, impl = registry.get("mcp_test_server_lookup")
    assert defn.name == "mcp_test_server_lookup"

    # Call the impl — it must route through MCPClient.call_tool
    await impl(query="test")
    client.call_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_adapter_unregister_server():
    """unregister_server removes tools from the registry."""
    registry = ToolRegistry()
    client = MCPClient()
    client.disconnect = AsyncMock()
    adapter = MCPToolAdapter(registry=registry, client=client)

    tools = [_make_tool_def("test-server", "lookup")]
    client.connect = AsyncMock(return_value=tools)

    cfg = _make_server_config()
    await adapter.register_server(TENANT_A, cfg)

    await adapter.unregister_server(cfg.id)

    with pytest.raises(ToolError):
        registry.get("mcp_test_server_lookup")


@pytest.mark.asyncio
async def test_list_servers_returns_registered():
    """list_servers returns servers registered for the tenant."""
    registry = ToolRegistry()
    client = MCPClient()
    adapter = MCPToolAdapter(registry=registry, client=client)

    tools = [_make_tool_def("test-server", "ping")]
    client.connect = AsyncMock(return_value=tools)

    cfg = _make_server_config()
    await adapter.register_server(TENANT_A, cfg)

    servers = adapter.list_servers(TENANT_A)
    assert len(servers) == 1
    assert servers[0].name == "test-server"
