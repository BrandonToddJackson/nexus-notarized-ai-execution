"""
Phase 19 — MCP Integration tests.

Coverage:
  client.py     — make_tool_name, _normalize, _convert_tools, connect (all 3
                  transports), call_tool (success/error/structuredContent),
                  refresh_tools, disconnect, disconnect_all, introspection helpers
  adapter.py    — register_server, unregister_server, list_servers, get_server,
                  reconnect_all (with and without repo), _make_tool_impl closure
  registry.py   — register with source, unregister, get_by_source

  INTEGRATION (real subprocess, no mocks):
  TestMCPClientIntegration    — echo server (FastMCP, mcp SDK): connect/call/disconnect
  TestMCPFetchServerIntegration — mcp-server-fetch (modelcontextprotocol/servers):
                                   connect, discover fetch tool, call fetch against
                                   a local HTTP server — CONCRETE FAILURE if broken
  TestMCPToolAdapterIntegration — adapter + registry end-to-end with real subprocess

  CREDENTIAL INJECTION (vault → MCP server env vars):
  TestVaultGetEnvVars           — unit: get_env_vars() for all credential types
  TestMCPAdapterCredentialInject — unit: adapter injects vault creds before connect
  TestMCPAdapterCredentialIntegration — integration: real subprocess receives injected env
"""

from __future__ import annotations

import sys
import threading
import uuid
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Paths to the fixture server scripts
_FIXTURE_DIR = Path(__file__).parent / "fixtures"
ECHO_SERVER_SCRIPT = str(_FIXTURE_DIR / "mcp_echo_server.py")
FETCH_SERVER_SCRIPT = str(_FIXTURE_DIR / "mcp_fetch_server.py")

from nexus.exceptions import MCPConnectionError, MCPToolError  # noqa: E402
from nexus.mcp.client import MCPClient, make_tool_name, _normalize  # noqa: E402
from nexus.mcp.adapter import MCPToolAdapter  # noqa: E402
from nexus.tools.registry import ToolRegistry  # noqa: E402
from nexus.types import MCPServerConfig, RiskLevel, ToolDefinition  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_server_config(
    name: str = "test-server",
    transport: str = "stdio",
    url: str = "http://localhost:9000",
    command: str = "python",
    args: list[str] | None = None,
    enabled: bool = True,
) -> MCPServerConfig:
    return MCPServerConfig(
        id=str(uuid.uuid4()),
        tenant_id="tenant-1",
        name=name,
        url=url,
        transport=transport,
        command=command,
        args=args or [],
        env={},
        enabled=enabled,
    )


def _make_mcp_tool(name: str, description: str = "A tool", schema: dict | None = None):
    """Create a minimal mock MCP Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema or {"type": "object", "properties": {}}
    return tool


def _make_list_tools_result(tools: list[Any]):
    result = MagicMock()
    result.tools = tools
    return result


def _make_call_tool_result(
    text: str = "ok",
    is_error: bool = False,
    structured: dict | None = None,
) -> MagicMock:
    result = MagicMock()
    result.isError = is_error
    result.structuredContent = structured
    text_content = MagicMock()
    text_content.text = text
    result.content = [text_content]
    return result


@asynccontextmanager
async def _fake_transport(read, write):
    yield read, write


@asynccontextmanager
async def _fake_transport3(read, write):
    yield read, write, lambda: "session-id"


def _mock_session(list_tools_result=None, call_tool_result=None):
    """Return a mock ClientSession that behaves as an async context manager."""
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=list_tools_result or _make_list_tools_result([]))
    session.call_tool = AsyncMock(return_value=call_tool_result or _make_call_tool_result())
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


# ---------------------------------------------------------------------------
# make_tool_name / _normalize
# ---------------------------------------------------------------------------

class TestNamespacing:
    def test_normalize_replaces_hyphens(self):
        assert _normalize("my-server") == "my_server"

    def test_normalize_replaces_spaces(self):
        assert _normalize("my server") == "my_server"

    def test_normalize_replaces_dots(self):
        assert _normalize("a.b.c") == "a_b_c"

    def test_normalize_keeps_alphanumeric(self):
        assert _normalize("abc123") == "abc123"

    def test_make_tool_name_basic(self):
        assert make_tool_name("my-server", "list_files") == "mcp_my_server_list_files"

    def test_make_tool_name_complex(self):
        name = make_tool_name("Acme Corp", "fetch-data.v2")
        assert name == "mcp_Acme_Corp_fetch_data_v2"

    def test_make_tool_name_prefix(self):
        assert make_tool_name("s", "t").startswith("mcp_")


# ---------------------------------------------------------------------------
# ToolRegistry — source / unregister / get_by_source
# ---------------------------------------------------------------------------

class TestRegistrySource:
    def _make_defn(self, name: str) -> ToolDefinition:
        return ToolDefinition(
            name=name,
            description="desc",
            parameters={"type": "object", "properties": {}},
        )

    def test_register_default_source_is_local(self):
        reg = ToolRegistry()
        defn = self._make_defn("my_tool")
        reg.register(defn, AsyncMock())
        assert reg._sources["my_tool"] == "local"

    def test_register_custom_source(self):
        reg = ToolRegistry()
        defn = self._make_defn("mcp_s_t")
        reg.register(defn, AsyncMock(), source="mcp")
        assert reg._sources["mcp_s_t"] == "mcp"

    def test_get_by_source_filters(self):
        reg = ToolRegistry()
        reg.register(self._make_defn("local_tool"), AsyncMock(), source="local")
        reg.register(self._make_defn("mcp_s_t1"), AsyncMock(), source="mcp")
        reg.register(self._make_defn("mcp_s_t2"), AsyncMock(), source="mcp")

        mcp_tools = reg.get_by_source("mcp")
        assert len(mcp_tools) == 2
        assert all(t.name.startswith("mcp_") for t in mcp_tools)

    def test_get_by_source_empty_for_unknown(self):
        reg = ToolRegistry()
        reg.register(self._make_defn("x"), AsyncMock())
        assert reg.get_by_source("nonexistent") == []

    def test_unregister_removes_tool(self):
        reg = ToolRegistry()
        defn = self._make_defn("bye")
        reg.register(defn, AsyncMock())
        reg.unregister("bye")
        assert "bye" not in reg._tools
        assert "bye" not in reg._implementations
        assert "bye" not in reg._sources

    def test_unregister_noop_for_missing(self):
        reg = ToolRegistry()
        reg.unregister("ghost")  # should not raise

    def test_existing_register_unchanged(self):
        """Existing register() call still works (backward compat)."""
        reg = ToolRegistry()
        defn = self._make_defn("compat")
        impl = AsyncMock()
        reg.register(defn, impl)
        d, i = reg.get("compat")
        assert d.name == "compat"
        assert i is impl


# ---------------------------------------------------------------------------
# MCPClient — unit tests with mocked MCP SDK
# ---------------------------------------------------------------------------

class TestMCPClientConvertTools:
    """Test _convert_tools without network."""

    def test_converts_basic_tool(self):
        client = MCPClient()
        mcp_tool = _make_mcp_tool("list_files", "Lists files")
        defs, name_map = client._convert_tools("my-server", [mcp_tool])
        assert len(defs) == 1
        defn = defs[0]
        assert defn.name == "mcp_my_server_list_files"
        assert "Lists files" in defn.description
        assert defn.resource_pattern == "mcp:my_server:*"
        assert name_map["mcp_my_server_list_files"] == "list_files"

    def test_converts_multiple_tools(self):
        client = MCPClient()
        tools = [_make_mcp_tool(f"tool_{i}") for i in range(3)]
        defs, name_map = client._convert_tools("srv", tools)
        assert len(defs) == 3
        assert len(name_map) == 3

    def test_missing_description_uses_fallback(self):
        client = MCPClient()
        tool = _make_mcp_tool("x")
        tool.description = None
        defs, _ = client._convert_tools("s", [tool])
        assert "MCP tool" in defs[0].description

    def test_missing_schema_uses_empty_object(self):
        client = MCPClient()
        tool = _make_mcp_tool("x")
        tool.inputSchema = None
        defs, _ = client._convert_tools("s", [tool])
        assert defs[0].parameters == {"type": "object", "properties": {}}

    def test_empty_tools_list(self):
        client = MCPClient()
        defs, name_map = client._convert_tools("s", [])
        assert defs == []
        assert name_map == {}


class TestMCPClientExtractText:
    def test_single_text_content(self):
        item = MagicMock()
        item.text = "hello"
        assert MCPClient._extract_text([item]) == "hello"

    def test_multiple_items_joined(self):
        items = [MagicMock() for _ in range(3)]
        for i, it in enumerate(items):
            it.text = str(i)
        assert MCPClient._extract_text(items) == "0\n1\n2"

    def test_no_text_attribute(self):
        item = MagicMock(spec=[])  # no attributes
        assert MCPClient._extract_text([item]) == ""

    def test_empty_list(self):
        assert MCPClient._extract_text([]) == ""


class TestMCPClientConnect:
    """Tests for connect() using mocked transports."""

    def _patch_stdio(self, session):
        @asynccontextmanager
        async def _cm(params):
            yield MagicMock(), MagicMock()

        return patch("nexus.mcp.client.stdio_client", _cm), \
               patch("nexus.mcp.client.ClientSession", return_value=session)

    @pytest.mark.asyncio
    async def test_connect_stdio_registers_tools(self):
        mcp_tools = [_make_mcp_tool("search"), _make_mcp_tool("write")]
        list_result = _make_list_tools_result(mcp_tools)
        session = _mock_session(list_tools_result=list_result)

        cfg = _make_server_config(transport="stdio", command="python")

        with patch("nexus.mcp.client.stdio_client") as mock_stdio, \
             patch("nexus.mcp.client.ClientSession", return_value=session):

            mock_stdio.return_value.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock())
            )
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

            client = MCPClient()
            defs = await client.connect(cfg)

        assert len(defs) == 2
        assert client.is_connected(cfg.id)
        assert all(d.name.startswith("mcp_") for d in defs)

    @pytest.mark.asyncio
    async def test_connect_sse_transport(self):
        list_result = _make_list_tools_result([_make_mcp_tool("sse_tool")])
        session = _mock_session(list_tools_result=list_result)
        cfg = _make_server_config(transport="sse", url="http://localhost:8080/sse")

        with patch("nexus.mcp.client.sse_client") as mock_sse, \
             patch("nexus.mcp.client.ClientSession", return_value=session):
            mock_sse.return_value.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock())
            )
            mock_sse.return_value.__aexit__ = AsyncMock(return_value=False)

            client = MCPClient()
            defs = await client.connect(cfg)

        assert len(defs) == 1
        assert defs[0].name == "mcp_test_server_sse_tool"

    @pytest.mark.asyncio
    async def test_connect_streamable_http_transport(self):
        list_result = _make_list_tools_result([_make_mcp_tool("http_tool")])
        session = _mock_session(list_tools_result=list_result)
        cfg = _make_server_config(transport="streamable_http", url="http://localhost:9090/mcp")

        with patch("nexus.mcp.client.streamable_http_client") as mock_http, \
             patch("nexus.mcp.client.ClientSession", return_value=session):
            mock_http.return_value.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock(), lambda: "sid")
            )
            mock_http.return_value.__aexit__ = AsyncMock(return_value=False)

            client = MCPClient()
            defs = await client.connect(cfg)

        assert len(defs) == 1

    @pytest.mark.asyncio
    async def test_connect_already_connected_calls_refresh(self):
        cfg = _make_server_config(transport="stdio")
        client = MCPClient()
        # Manually inject a fake session
        fake_session = AsyncMock()
        fake_session.list_tools = AsyncMock(
            return_value=_make_list_tools_result([_make_mcp_tool("t")])
        )
        client._sessions[cfg.id] = fake_session
        client._configs[cfg.id] = cfg
        client._tool_name_map[cfg.id] = {}

        defs = await client.connect(cfg)
        assert len(defs) == 1

    @pytest.mark.asyncio
    async def test_connect_unknown_transport_raises(self):
        cfg = _make_server_config(transport="websocket")
        client = MCPClient()
        with pytest.raises(MCPConnectionError, match="Unknown transport"):
            await client.connect(cfg)

    @pytest.mark.asyncio
    async def test_connect_stdio_missing_command_raises(self):
        cfg = _make_server_config(transport="stdio", command=None)
        cfg.command = None
        client = MCPClient()
        with pytest.raises(MCPConnectionError, match="command"):
            await client.connect(cfg)

    @pytest.mark.asyncio
    async def test_connect_failure_raises_connection_error(self):
        cfg = _make_server_config(transport="sse")
        with patch("nexus.mcp.client.sse_client") as mock_sse:
            mock_sse.return_value.__aenter__ = AsyncMock(
                side_effect=RuntimeError("Connection refused")
            )
            mock_sse.return_value.__aexit__ = AsyncMock(return_value=False)
            client = MCPClient()
            with pytest.raises(MCPConnectionError, match="Connection refused"):
                await client.connect(cfg)


class TestMCPClientCallTool:

    def _connected_client(self, cfg: MCPServerConfig, call_tool_result=None):
        """Return a client with a fake session already connected."""
        client = MCPClient()
        session = _mock_session(call_tool_result=call_tool_result)
        client._sessions[cfg.id] = session
        client._configs[cfg.id] = cfg
        namespaced = make_tool_name(cfg.name, "search")
        client._tool_name_map[cfg.id] = {namespaced: "search"}
        return client, session, namespaced

    @pytest.mark.asyncio
    async def test_call_tool_returns_text(self):
        cfg = _make_server_config()
        result = _make_call_tool_result(text="found it")
        client, session, namespaced = self._connected_client(cfg, result)

        out = await client.call_tool(cfg.id, namespaced, {"query": "hello"})
        assert out == "found it"
        session.call_tool.assert_called_once_with("search", {"query": "hello"})

    @pytest.mark.asyncio
    async def test_call_tool_returns_structured_content(self):
        cfg = _make_server_config()
        result = _make_call_tool_result(structured={"rows": [1, 2, 3]})
        client, _, namespaced = self._connected_client(cfg, result)

        out = await client.call_tool(cfg.id, namespaced, {})
        assert out == {"rows": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_call_tool_is_error_raises(self):
        cfg = _make_server_config()
        result = _make_call_tool_result(text="permission denied", is_error=True)
        client, _, namespaced = self._connected_client(cfg, result)

        with pytest.raises(MCPToolError, match="permission denied"):
            await client.call_tool(cfg.id, namespaced, {})

    @pytest.mark.asyncio
    async def test_call_tool_session_raises_wraps_error(self):
        cfg = _make_server_config()
        client, session, namespaced = self._connected_client(cfg)
        session.call_tool.side_effect = RuntimeError("network timeout")

        with pytest.raises(MCPToolError, match="network timeout"):
            await client.call_tool(cfg.id, namespaced, {})

    @pytest.mark.asyncio
    async def test_call_tool_not_connected_raises(self):
        client = MCPClient()
        with pytest.raises(MCPConnectionError, match="not connected"):
            await client.call_tool("nonexistent", "mcp_x_y", {})

    @pytest.mark.asyncio
    async def test_call_tool_uses_original_name(self):
        """Namespaced name is translated back to the original MCP tool name."""
        cfg = _make_server_config(name="my-server")
        client = MCPClient()
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=_make_call_tool_result())
        client._sessions[cfg.id] = session
        client._configs[cfg.id] = cfg
        client._tool_name_map[cfg.id] = {"mcp_my_server_list": "list-files"}

        await client.call_tool(cfg.id, "mcp_my_server_list", {})
        session.call_tool.assert_called_once_with("list-files", {})


class TestMCPClientRefreshTools:

    @pytest.mark.asyncio
    async def test_refresh_tools_not_connected_raises(self):
        client = MCPClient()
        with pytest.raises(MCPConnectionError):
            await client.refresh_tools("missing-id")

    @pytest.mark.asyncio
    async def test_refresh_tools_updates_map(self):
        cfg = _make_server_config()
        client = MCPClient()
        new_tools = [_make_mcp_tool("new_tool")]
        session = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=_make_list_tools_result(new_tools)
        )
        client._sessions[cfg.id] = session
        client._configs[cfg.id] = cfg
        client._tool_name_map[cfg.id] = {}

        defs = await client.refresh_tools(cfg.id)
        assert len(defs) == 1
        assert "mcp_test_server_new_tool" in client._tool_name_map[cfg.id]


class TestMCPClientDisconnect:

    @pytest.mark.asyncio
    async def test_disconnect_removes_session(self):
        cfg = _make_server_config()
        client = MCPClient()
        session = AsyncMock()
        stack = AsyncMock()
        stack.aclose = AsyncMock()
        client._sessions[cfg.id] = session
        client._exit_stacks[cfg.id] = stack
        client._configs[cfg.id] = cfg
        client._tool_name_map[cfg.id] = {}

        await client.disconnect(cfg.id)
        assert not client.is_connected(cfg.id)
        stack.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_noop_for_unknown(self):
        client = MCPClient()
        await client.disconnect("ghost")  # should not raise

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        client = MCPClient()
        ids = [str(uuid.uuid4()) for _ in range(3)]
        for sid in ids:
            session = AsyncMock()
            stack = AsyncMock()
            stack.aclose = AsyncMock()
            client._sessions[sid] = session
            client._exit_stacks[sid] = stack

        await client.disconnect_all()
        assert client.list_connected() == []


class TestMCPClientIntrospection:

    def test_is_connected_false_initially(self):
        client = MCPClient()
        assert not client.is_connected("x")

    def test_list_connected_empty_initially(self):
        client = MCPClient()
        assert client.list_connected() == []

    def test_get_tool_name_map_empty_for_unknown(self):
        client = MCPClient()
        assert client.get_tool_name_map("x") == {}

    def test_get_tool_name_map_returns_copy(self):
        client = MCPClient()
        client._tool_name_map["srv"] = {"a": "b"}
        m = client.get_tool_name_map("srv")
        m["c"] = "d"
        assert "c" not in client._tool_name_map["srv"]


# ---------------------------------------------------------------------------
# MCPToolAdapter
# ---------------------------------------------------------------------------

class TestMCPToolAdapter:

    def _make_adapter(self, list_tools_result=None):
        """Return (adapter, mock_client, registry)."""
        registry = ToolRegistry()
        mock_client = AsyncMock(spec=MCPClient)
        mock_client.is_connected = MagicMock(return_value=False)
        mock_client.disconnect = AsyncMock()
        tools = list_tools_result or [_make_mcp_tool("search")]
        mock_client.connect = AsyncMock(return_value=[
            ToolDefinition(
                name=make_tool_name("test-server", t.name),
                description=t.description,
                parameters=t.inputSchema,
            )
            for t in tools
        ])
        adapter = MCPToolAdapter(registry, client=mock_client)
        return adapter, mock_client, registry

    @pytest.mark.asyncio
    async def test_register_server_adds_tools_to_registry(self):
        adapter, _, registry = self._make_adapter()
        cfg = _make_server_config()
        defs = await adapter.register_server("t1", cfg)

        assert len(defs) == 1
        assert "mcp_test_server_search" in registry._tools

    @pytest.mark.asyncio
    async def test_register_server_source_is_mcp(self):
        adapter, _, registry = self._make_adapter()
        cfg = _make_server_config()
        await adapter.register_server("t1", cfg)
        assert registry._sources.get("mcp_test_server_search") == "mcp"

    @pytest.mark.asyncio
    async def test_register_server_stores_config(self):
        adapter, _, _ = self._make_adapter()
        cfg = _make_server_config()
        await adapter.register_server("t1", cfg)
        assert adapter.get_server(cfg.id) == cfg

    @pytest.mark.asyncio
    async def test_unregister_server_removes_tools(self):
        adapter, mock_client, registry = self._make_adapter()
        cfg = _make_server_config()
        await adapter.register_server("t1", cfg)
        assert "mcp_test_server_search" in registry._tools

        await adapter.unregister_server(cfg.id)
        assert "mcp_test_server_search" not in registry._tools
        mock_client.disconnect.assert_called_once_with(cfg.id)

    @pytest.mark.asyncio
    async def test_unregister_server_removes_from_servers(self):
        adapter, _, _ = self._make_adapter()
        cfg = _make_server_config()
        await adapter.register_server("t1", cfg)
        await adapter.unregister_server(cfg.id)
        assert adapter.get_server(cfg.id) is None

    @pytest.mark.asyncio
    async def test_list_servers_filters_by_tenant(self):
        adapter, _, _ = self._make_adapter()
        cfg1 = _make_server_config(name="s1")
        cfg2 = _make_server_config(name="s2")
        await adapter.register_server("t1", cfg1)
        await adapter.register_server("t2", cfg2)

        t1_servers = adapter.list_servers("t1")
        assert len(t1_servers) == 1
        assert t1_servers[0].name == "s1"

    @pytest.mark.asyncio
    async def test_list_servers_empty_for_unknown_tenant(self):
        adapter, _, _ = self._make_adapter()
        cfg = _make_server_config()
        await adapter.register_server("t1", cfg)
        assert adapter.list_servers("nobody") == []

    @pytest.mark.asyncio
    async def test_reconnect_all_skips_already_connected(self):
        adapter, mock_client, _ = self._make_adapter()
        cfg = _make_server_config()
        adapter._servers[cfg.id] = cfg
        mock_client.is_connected.return_value = True  # already connected

        await adapter.reconnect_all("tenant-1")
        mock_client.connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconnect_all_skips_disabled(self):
        adapter, mock_client, _ = self._make_adapter()
        cfg = _make_server_config(enabled=False)
        cfg.tenant_id = "tenant-1"
        adapter._servers[cfg.id] = cfg

        await adapter.reconnect_all("tenant-1")
        mock_client.connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconnect_all_connects_enabled(self):
        adapter, mock_client, _ = self._make_adapter()
        cfg = _make_server_config(enabled=True)
        cfg.tenant_id = "tenant-1"
        adapter._servers[cfg.id] = cfg

        await adapter.reconnect_all("tenant-1")
        mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_all_logs_connection_failure(self, caplog):
        """A server that fails to reconnect should not block others."""
        adapter, mock_client, _ = self._make_adapter()
        mock_client.connect.side_effect = MCPConnectionError("timeout", server_name="s")
        cfg = _make_server_config(enabled=True)
        cfg.tenant_id = "tenant-1"
        adapter._servers[cfg.id] = cfg

        import logging
        with caplog.at_level(logging.WARNING):
            await adapter.reconnect_all("tenant-1")
        assert any("timeout" in r.message or "Failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_reconnect_all_with_repo(self):
        """When a repo is provided, servers are loaded from it."""
        repo = AsyncMock()
        cfg = _make_server_config(enabled=True)
        cfg.tenant_id = "tenant-1"
        repo.list_mcp_servers = AsyncMock(return_value=[cfg])

        registry = ToolRegistry()
        mock_client = AsyncMock(spec=MCPClient)
        mock_client.is_connected = MagicMock(return_value=False)
        mock_client.connect = AsyncMock(return_value=[
            ToolDefinition(
                name=make_tool_name(cfg.name, "search"),
                description="desc",
                parameters={},
            )
        ])
        mock_client.disconnect = AsyncMock()

        adapter = MCPToolAdapter(registry, client=mock_client, repository=repo)
        await adapter.reconnect_all("tenant-1")

        repo.list_mcp_servers.assert_called_once_with("tenant-1")
        mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_server_persists_to_repo(self):
        repo = AsyncMock()
        repo.save_mcp_server = AsyncMock()

        registry = ToolRegistry()
        mock_client = AsyncMock(spec=MCPClient)
        mock_client.connect = AsyncMock(return_value=[
            ToolDefinition(name="mcp_s_t", description="d", parameters={})
        ])
        mock_client.disconnect = AsyncMock()

        adapter = MCPToolAdapter(registry, client=mock_client, repository=repo)
        cfg = _make_server_config()
        await adapter.register_server("t1", cfg)

        repo.save_mcp_server.assert_called_once_with(cfg)

    @pytest.mark.asyncio
    async def test_tool_impl_routes_to_client(self):
        """The impl closure registered in the registry calls client.call_tool."""
        adapter, mock_client, registry = self._make_adapter()
        mock_client.call_tool = AsyncMock(return_value="tool output")
        cfg = _make_server_config()
        await adapter.register_server("t1", cfg)

        _, impl = registry.get("mcp_test_server_search")
        result = await impl(query="hello")
        mock_client.call_tool.assert_called_once()
        assert result == "tool output"

# ===========================================================================
# INTEGRATION TESTS — real subprocesses, no mocks
# ===========================================================================

def _echo_server_cfg(server_id: str = "echo-integration") -> "MCPServerConfig":
    """MCPServerConfig for the FastMCP echo server subprocess."""
    from nexus.types import MCPServerConfig
    return MCPServerConfig(
        id=server_id,
        tenant_id="integration-tenant",
        name="nexus-test-echo-server",
        url="",
        transport="stdio",
        command=sys.executable,
        args=[ECHO_SERVER_SCRIPT],
    )


def _fetch_server_cfg(server_id: str = "fetch-integration") -> "MCPServerConfig":
    """MCPServerConfig for the mcp-server-fetch subprocess."""
    from nexus.types import MCPServerConfig
    return MCPServerConfig(
        id=server_id,
        tenant_id="integration-tenant",
        name="mcp-server-fetch",
        url="",
        transport="stdio",
        command=sys.executable,
        args=[FETCH_SERVER_SCRIPT, "--ignore-robots-txt"],
    )


class _LocalHTTPServer:
    """Tiny HTTP server that serves static HTML content on a free port."""

    BODY = b"<html><body><h1>NEXUS MCP Integration Test</h1></body></html>"

    def __init__(self):
        body = self.BODY

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # silence access logs

        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/"
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()


# ---------------------------------------------------------------------------
# TestMCPClientIntegration — FastMCP echo server (real stdio transport)
# ---------------------------------------------------------------------------

class TestMCPClientIntegration:
    """End-to-end tests using a REAL FastMCP subprocess (mcp SDK).

    The echo server exposes three deterministic tools (echo, add, greet).
    All tests use real stdio transport — no mocks anywhere.
    """

    @pytest.mark.asyncio
    async def test_connect_discovers_three_tools(self):
        """MCPClient.connect() lists the 3 tools registered in the echo server."""
        from nexus.mcp.client import MCPClient
        client = MCPClient()
        cfg = _echo_server_cfg()
        try:
            defs = await client.connect(cfg)
            tool_names = {d.name for d in defs}
            assert len(defs) == 3
            assert "mcp_nexus_test_echo_server_echo" in tool_names
            assert "mcp_nexus_test_echo_server_add" in tool_names
            assert "mcp_nexus_test_echo_server_greet" in tool_names
        finally:
            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_tool_names_are_namespaced_correctly(self):
        """All tool names carry the mcp_{server_name}_{tool} prefix."""
        from nexus.mcp.client import MCPClient
        client = MCPClient()
        cfg = _echo_server_cfg()
        try:
            defs = await client.connect(cfg)
            assert all(d.name.startswith("mcp_nexus_test_echo_server_") for d in defs)
        finally:
            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_echo_tool_returns_input_unchanged(self):
        """call_tool('echo', {'text': 'hello'}) → {'result': 'hello'}."""
        from nexus.mcp.client import MCPClient
        client = MCPClient()
        cfg = _echo_server_cfg()
        try:
            await client.connect(cfg)
            result = await client.call_tool(
                cfg.id, "mcp_nexus_test_echo_server_echo", {"text": "hello-integration"}
            )
            assert result == {"result": "hello-integration"}
        finally:
            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_add_tool_returns_correct_sum(self):
        """call_tool('add', {'a': 7, 'b': 13}) → {'result': 20}."""
        from nexus.mcp.client import MCPClient
        client = MCPClient()
        cfg = _echo_server_cfg()
        try:
            await client.connect(cfg)
            result = await client.call_tool(
                cfg.id, "mcp_nexus_test_echo_server_add", {"a": 7, "b": 13}
            )
            assert result == {"result": 20}
        finally:
            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_greet_tool_returns_greeting(self):
        """call_tool('greet', {'name': 'World'}) → {'result': 'Hello, World!'}."""
        from nexus.mcp.client import MCPClient
        client = MCPClient()
        cfg = _echo_server_cfg()
        try:
            await client.connect(cfg)
            result = await client.call_tool(
                cfg.id, "mcp_nexus_test_echo_server_greet", {"name": "World"}
            )
            assert result == {"result": "Hello, World!"}
        finally:
            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_disconnect_ends_session(self):
        """After disconnect(), is_connected() returns False."""
        from nexus.mcp.client import MCPClient
        client = MCPClient()
        cfg = _echo_server_cfg()
        await client.connect(cfg)
        assert client.is_connected(cfg.id)
        await client.disconnect(cfg.id)
        assert not client.is_connected(cfg.id)

    @pytest.mark.asyncio
    async def test_reconnect_rediscovers_tools(self):
        """Disconnect then reconnect to the same server returns the same tools."""
        from nexus.mcp.client import MCPClient
        client = MCPClient()
        cfg = _echo_server_cfg()
        try:
            defs1 = await client.connect(cfg)
            await client.disconnect(cfg.id)
            defs2 = await client.connect(cfg)
            assert {d.name for d in defs1} == {d.name for d in defs2}
        finally:
            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_tool_definition_has_resource_pattern(self):
        """Every discovered tool has the server-scoped resource_pattern."""
        from nexus.mcp.client import MCPClient
        client = MCPClient()
        cfg = _echo_server_cfg()
        try:
            defs = await client.connect(cfg)
            for d in defs:
                assert d.resource_pattern == "mcp:nexus_test_echo_server:*"
        finally:
            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_refresh_tools_returns_same_tools(self):
        """refresh_tools() on an already-connected server returns same tool set."""
        from nexus.mcp.client import MCPClient
        client = MCPClient()
        cfg = _echo_server_cfg()
        try:
            defs1 = await client.connect(cfg)
            defs2 = await client.refresh_tools(cfg.id)
            assert {d.name for d in defs1} == {d.name for d in defs2}
        finally:
            await client.disconnect_all()


# ---------------------------------------------------------------------------
# TestMCPFetchServerIntegration — mcp-server-fetch (modelcontextprotocol)
# ---------------------------------------------------------------------------

class TestMCPFetchServerIntegration:
    """Integration tests against the REAL mcp-server-fetch from modelcontextprotocol/servers.

    Uses a local HTTP server so tests are network-independent and deterministic.

    All tests are CONCRETE FAILURES — no skips, no mocks.  If mcp-server-fetch
    cannot connect, start, or call the fetch tool, the test fails explicitly.
    """

    @pytest.mark.asyncio
    async def test_fetch_server_connects_and_discovers_fetch_tool(self):
        """mcp-server-fetch starts and exposes exactly one 'fetch' tool."""
        from nexus.mcp.client import MCPClient
        client = MCPClient()
        cfg = _fetch_server_cfg()
        try:
            defs = await client.connect(cfg)
            tool_names = {d.name for d in defs}
            # mcp-server-fetch registers a single 'fetch' tool
            assert "mcp_mcp_server_fetch_fetch" in tool_names, (
                f"Expected 'mcp_mcp_server_fetch_fetch' in discovered tools: {tool_names}"
            )
        finally:
            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_fetch_server_fetch_tool_has_url_parameter(self):
        """The fetch tool's JSON Schema declares 'url' as a required string parameter."""
        from nexus.mcp.client import MCPClient
        client = MCPClient()
        cfg = _fetch_server_cfg()
        try:
            defs = await client.connect(cfg)
            fetch_def = next(d for d in defs if d.name == "mcp_mcp_server_fetch_fetch")
            assert "url" in fetch_def.parameters.get("properties", {})
            assert "url" in fetch_def.parameters.get("required", [])
        finally:
            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_fetch_tool_retrieves_content_from_local_server(self):
        """Calling fetch with a local HTTP URL returns the page content.

        A local HTTP server is used so the test is network-independent.
        CONCRETE FAILURE: if mcp-server-fetch cannot call the tool, the test fails.
        """
        local_srv = _LocalHTTPServer()
        local_srv.start()
        try:
            from nexus.mcp.client import MCPClient
            client = MCPClient()
            cfg = _fetch_server_cfg()
            try:
                await client.connect(cfg)
                result = await client.call_tool(
                    cfg.id,
                    "mcp_mcp_server_fetch_fetch",
                    {"url": local_srv.url, "max_length": 500},
                )
                content = str(result)
                assert "NEXUS MCP Integration Test" in content, (
                    f"Expected test content in fetch result, got: {content[:300]}"
                )
            finally:
                await client.disconnect_all()
        finally:
            local_srv.stop()

    @pytest.mark.asyncio
    async def test_fetch_tool_content_is_non_empty(self):
        """fetch returns a non-empty string when the URL is reachable."""
        local_srv = _LocalHTTPServer()
        local_srv.start()
        try:
            from nexus.mcp.client import MCPClient
            client = MCPClient()
            cfg = _fetch_server_cfg()
            try:
                await client.connect(cfg)
                result = await client.call_tool(
                    cfg.id,
                    "mcp_mcp_server_fetch_fetch",
                    {"url": local_srv.url},
                )
                assert result, "fetch returned empty result — concrete failure"
                assert len(str(result)) > 0
            finally:
                await client.disconnect_all()
        finally:
            local_srv.stop()


# ---------------------------------------------------------------------------
# TestMCPToolAdapterIntegration — real subprocess + ToolRegistry
# ---------------------------------------------------------------------------

class TestMCPToolAdapterIntegration:
    """MCPToolAdapter wired to a REAL echo server subprocess.

    Verifies that tools flow from the subprocess through the adapter
    into the ToolRegistry and can be called via the registry's normal
    get/execute pathway.  No mocks.
    """

    @pytest.mark.asyncio
    async def test_adapter_registers_echo_server_tools_with_mcp_source(self):
        """register_server() puts all tools in registry with source='mcp'."""
        from nexus.mcp.adapter import MCPToolAdapter
        from nexus.tools.registry import ToolRegistry
        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry)
        cfg = _echo_server_cfg("adapter-echo-1")
        try:
            await adapter.register_server("integration-tenant", cfg)
            mcp_tools = registry.get_by_source("mcp")
            names = {t.name for t in mcp_tools}
            assert "mcp_nexus_test_echo_server_echo" in names
            assert "mcp_nexus_test_echo_server_add" in names
            assert "mcp_nexus_test_echo_server_greet" in names
        finally:
            await adapter._client.disconnect_all()

    @pytest.mark.asyncio
    async def test_adapter_tool_callable_via_registry_returns_real_result(self):
        """A tool retrieved from the registry and called returns a real result."""
        from nexus.mcp.adapter import MCPToolAdapter
        from nexus.tools.registry import ToolRegistry
        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry)
        cfg = _echo_server_cfg("adapter-echo-2")
        try:
            await adapter.register_server("integration-tenant", cfg)
            _, impl = registry.get("mcp_nexus_test_echo_server_echo")
            result = await impl(text="registry-callable")
            assert result == {"result": "registry-callable"}, (
                f"Expected {{'result': 'registry-callable'}}, got: {result}"
            )
        finally:
            await adapter._client.disconnect_all()

    @pytest.mark.asyncio
    async def test_adapter_add_tool_callable_via_registry(self):
        """add tool called via registry performs real integer addition."""
        from nexus.mcp.adapter import MCPToolAdapter
        from nexus.tools.registry import ToolRegistry
        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry)
        cfg = _echo_server_cfg("adapter-echo-3")
        try:
            await adapter.register_server("integration-tenant", cfg)
            _, impl = registry.get("mcp_nexus_test_echo_server_add")
            result = await impl(a=100, b=42)
            assert result == {"result": 142}
        finally:
            await adapter._client.disconnect_all()

    @pytest.mark.asyncio
    async def test_unregister_server_removes_tools_from_registry(self):
        """unregister_server() removes all MCP tools from the registry."""
        from nexus.mcp.adapter import MCPToolAdapter
        from nexus.tools.registry import ToolRegistry
        from nexus.exceptions import ToolError
        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry)
        cfg = _echo_server_cfg("adapter-echo-4")
        await adapter.register_server("integration-tenant", cfg)
        assert len(registry.get_by_source("mcp")) == 3

        await adapter.unregister_server(cfg.id)
        assert len(registry.get_by_source("mcp")) == 0
        with pytest.raises(ToolError):
            registry.get("mcp_nexus_test_echo_server_echo")

    @pytest.mark.asyncio
    async def test_mcp_tools_do_not_shadow_local_tools(self):
        """MCP and local tools coexist independently in the same registry."""
        from nexus.mcp.adapter import MCPToolAdapter
        from nexus.tools.registry import ToolRegistry
        from nexus.types import ToolDefinition
        registry = ToolRegistry()

        # Register a local tool first
        local_def = ToolDefinition(name="local_search", description="local", parameters={})
        registry.register(local_def, AsyncMock(), source="local")

        adapter = MCPToolAdapter(registry)
        cfg = _echo_server_cfg("adapter-echo-5")
        try:
            await adapter.register_server("integration-tenant", cfg)
            assert len(registry.get_by_source("local")) == 1
            assert len(registry.get_by_source("mcp")) == 3
            # Both are accessible
            registry.get("local_search")
            registry.get("mcp_nexus_test_echo_server_echo")
        finally:
            await adapter._client.disconnect_all()


# ===========================================================================
# ENGINE INTEGRATION — MCP tool flows through the full NEXUS pipeline
# ===========================================================================

class TestMCPNexusEngineIntegration:
    """Proves MCP tools are first-class NEXUS integrations.

    An MCP server subprocess is connected, its tools registered, and
    NexusEngine.run() executes one through all 4 anomaly gates → notary →
    ledger.  The sealed audit record proves end-to-end integration.

    No mocks — real subprocess, real engine, real ledger.
    """

    def _build_engine(self, registry):
        from nexus.config import NexusConfig
        from nexus.core.anomaly import AnomalyEngine
        from nexus.core.chain import ChainManager
        from nexus.core.cot_logger import CoTLogger
        from nexus.core.engine import NexusEngine
        from nexus.core.ledger import Ledger
        from nexus.core.notary import Notary
        from nexus.core.output_validator import OutputValidator
        from nexus.core.personas import PersonaManager
        from nexus.core.verifier import IntentVerifier
        from nexus.knowledge.context import ContextBuilder
        from nexus.knowledge.store import KnowledgeStore
        from nexus.reasoning.continue_complete import ContinueCompleteGate
        from nexus.reasoning.escalate import EscalateGate
        from nexus.reasoning.think_act import ThinkActGate
        from nexus.tools.executor import ToolExecutor
        from nexus.tools.sandbox import Sandbox
        from nexus.tools.selector import ToolSelector
        from nexus.types import PersonaContract

        MCP_TOOL = "mcp_nexus_test_echo_server_echo"

        persona = PersonaContract(
            name="mcp-engine-test",
            description="Agent that uses MCP tools",
            allowed_tools=[MCP_TOOL],
            resource_scopes=["mcp:nexus_test_echo_server:*"],
            intent_patterns=["echo", "repeat", "send message"],
            max_ttl_seconds=120,
            risk_tolerance=RiskLevel.MEDIUM,
        )

        cfg = NexusConfig(
            database_url="sqlite+aiosqlite:///test.db",
            redis_url="redis://localhost:6379/15",
        )

        store = KnowledgeStore(persist_dir="/tmp/nexus_mcp_engine_test_chroma")
        return NexusEngine(
            persona_manager        = PersonaManager([persona]),
            anomaly_engine         = AnomalyEngine(config=cfg),
            notary                 = Notary(),
            ledger                 = Ledger(),
            chain_manager          = ChainManager(),
            context_builder        = ContextBuilder(knowledge_store=store),
            tool_registry          = registry,
            tool_selector          = ToolSelector(registry=registry),
            tool_executor          = ToolExecutor(
                registry=registry,
                sandbox=Sandbox(),
                verifier=IntentVerifier(),
            ),
            output_validator       = OutputValidator(),
            cot_logger             = CoTLogger(),
            think_act_gate         = ThinkActGate(),
            continue_complete_gate = ContinueCompleteGate(),
            escalate_gate          = EscalateGate(),
            config                 = cfg,
        )

    @pytest.mark.asyncio
    async def test_mcp_tool_selected_and_sealed_by_engine(self):
        """engine.run() selects the MCP tool, passes gates, seals result in ledger."""
        from nexus.mcp.adapter import MCPToolAdapter
        from nexus.tools.registry import ToolRegistry
        from nexus.types import ActionStatus

        registry = ToolRegistry()
        adapter  = MCPToolAdapter(registry)
        cfg = _echo_server_cfg("engine-echo")
        try:
            await adapter.register_server("test-tenant", cfg)
            engine = self._build_engine(registry)

            chain = await engine.run("echo: NEXUS MCP integration confirmed", "test-tenant")
            seals = await engine.ledger.get_chain(chain.id)

            assert seals, "No seals produced — engine did not execute"
            seal = seals[0]

            # The MCP tool was selected and is recorded in the seal
            assert seal.tool_name == "mcp_nexus_test_echo_server_echo", (
                f"Expected MCP tool in seal, got: {seal.tool_name}"
            )
            # The seal was executed (not blocked)
            assert seal.status == ActionStatus.EXECUTED, (
                f"Expected EXECUTED, got: {seal.status}"
            )
            # The MCP subprocess returned a real result
            assert seal.tool_result is not None, "MCP tool returned no result"
            assert seal.tool_result == {"result": "echo: NEXUS MCP integration confirmed"}, (
                f"Unexpected result: {seal.tool_result}"
            )
            # 4 anomaly gate results are present
            assert len(seal.anomaly_result.gates) == 4
            # Scope gate must pass (tool is in persona's allowed_tools)
            scope_gate = next(g for g in seal.anomaly_result.gates if g.gate_name == "scope")
            assert scope_gate.verdict.value == "pass", "Scope gate should PASS for allowed MCP tool"
            # Merkle fingerprint is set
            assert seal.fingerprint, "Seal has no fingerprint"
            # Chain completed
            from nexus.types import ChainStatus
            assert chain.status == ChainStatus.COMPLETED
        finally:
            await adapter._client.disconnect_all()

    @pytest.mark.asyncio
    async def test_mcp_tool_sealed_in_immutable_ledger(self):
        """Ledger records the MCP tool execution; Notary verifies chain integrity."""
        from nexus.core.notary import Notary
        from nexus.mcp.adapter import MCPToolAdapter
        from nexus.tools.registry import ToolRegistry

        registry = ToolRegistry()
        adapter  = MCPToolAdapter(registry)
        cfg = _echo_server_cfg("engine-echo-2")
        try:
            await adapter.register_server("test-tenant", cfg)
            engine = self._build_engine(registry)

            chain = await engine.run("echo: ledger integrity test", "test-tenant")
            seals = await engine.ledger.get_chain(chain.id)

            # Merkle chain is cryptographically intact
            assert Notary().verify_chain(seals) is True, (
                "Merkle chain verification failed for MCP-tool seal"
            )
            # Audit trail contains the MCP tool name
            assert any(s.tool_name == "mcp_nexus_test_echo_server_echo" for s in seals)
        finally:
            await adapter._client.disconnect_all()


# ===========================================================================
# Credential injection tests
# ===========================================================================

_ENV_SERVER_SCRIPT = str(_FIXTURE_DIR / "mcp_env_server.py")


def _make_vault():
    """Return a CredentialVault with an ephemeral encryption key."""
    from nexus.credentials.encryption import CredentialEncryption
    from nexus.credentials.vault import CredentialVault
    return CredentialVault(CredentialEncryption())


def _env_server_cfg(server_id: str = "env-srv") -> MCPServerConfig:
    return MCPServerConfig(
        id=server_id,
        tenant_id="t-cred",
        name="nexus-env-server",
        url="",
        transport="stdio",
        command=sys.executable,
        args=[_ENV_SERVER_SCRIPT],
    )


class TestVaultGetEnvVars:
    """Unit tests for CredentialVault.get_env_vars()."""

    def test_custom_type_returns_all_keys_as_env_vars(self):
        from nexus.types import CredentialType
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="ctx7",
            credential_type=CredentialType.CUSTOM,
            service_name="context7",
            data={
                "UPSTASH_REDIS_REST_URL": "https://example.upstash.io",
                "UPSTASH_REDIS_REST_TOKEN": "tok-abc123",
            },
        )
        env = vault.get_env_vars(rec.id, "t1")
        assert env == {
            "UPSTASH_REDIS_REST_URL": "https://example.upstash.io",
            "UPSTASH_REDIS_REST_TOKEN": "tok-abc123",
        }

    def test_api_key_type_returns_api_key_env_var(self):
        from nexus.types import CredentialType
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="gh-key",
            credential_type=CredentialType.API_KEY,
            service_name="github",
            data={"api_key": "ghp_secret123"},
        )
        env = vault.get_env_vars(rec.id, "t1")
        assert env == {"API_KEY": "ghp_secret123"}

    def test_oauth2_type_returns_access_token_env_var(self):
        from nexus.types import CredentialType
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="oauth-cred",
            credential_type=CredentialType.OAUTH2,
            service_name="google",
            data={"access_token": "ya29.token", "refresh_token": "1//refresh"},
        )
        env = vault.get_env_vars(rec.id, "t1")
        assert env == {"ACCESS_TOKEN": "ya29.token"}

    def test_bearer_token_type_returns_bearer_token_env_var(self):
        from nexus.types import CredentialType
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="bearer-cred",
            credential_type=CredentialType.BEARER_TOKEN,
            service_name="openai",
            data={"token": "sk-proj-abc"},
        )
        env = vault.get_env_vars(rec.id, "t1")
        assert env == {"BEARER_TOKEN": "sk-proj-abc"}

    def test_basic_auth_type_returns_username_password_env_vars(self):
        from nexus.types import CredentialType
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="basic-cred",
            credential_type=CredentialType.BASIC_AUTH,
            service_name="postgres",
            data={"username": "admin", "password": "s3cret"},
        )
        env = vault.get_env_vars(rec.id, "t1")
        assert env == {"USERNAME": "admin", "PASSWORD": "s3cret"}

    def test_tenant_mismatch_raises(self):
        from nexus.types import CredentialType
        from nexus.exceptions import CredentialNotFound
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="cred",
            credential_type=CredentialType.CUSTOM,
            service_name="svc",
            data={"KEY": "val"},
        )
        with pytest.raises(CredentialNotFound):
            vault.get_env_vars(rec.id, "wrong-tenant")

    def test_unknown_credential_id_raises(self):
        from nexus.exceptions import CredentialNotFound
        vault = _make_vault()
        with pytest.raises(CredentialNotFound):
            vault.get_env_vars("00000000-0000-0000-0000-000000000000", "t1")

    def test_scoped_persona_allowed(self):
        from nexus.types import CredentialType
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="scoped",
            credential_type=CredentialType.CUSTOM,
            service_name="svc",
            data={"MY_KEY": "my-val"},
            scoped_personas=["researcher"],
        )
        env = vault.get_env_vars(rec.id, "t1", persona_name="researcher")
        assert env["MY_KEY"] == "my-val"

    def test_scoped_persona_denied_raises(self):
        from nexus.types import CredentialType
        from nexus.exceptions import CredentialError
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="scoped",
            credential_type=CredentialType.CUSTOM,
            service_name="svc",
            data={"MY_KEY": "my-val"},
            scoped_personas=["researcher"],
        )
        with pytest.raises(CredentialError):
            vault.get_env_vars(rec.id, "t1", persona_name="other-persona")

    def test_all_values_cast_to_strings(self):
        from nexus.types import CredentialType
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="mixed",
            credential_type=CredentialType.CUSTOM,
            service_name="svc",
            data={"PORT": 6379, "ENABLED": True},
        )
        env = vault.get_env_vars(rec.id, "t1")
        assert env["PORT"] == "6379"
        assert env["ENABLED"] == "True"


class TestMCPAdapterCredentialInject:
    """Unit tests for MCPToolAdapter._inject_credential() and vault wiring."""

    def test_inject_credential_merges_env_vars(self):
        from nexus.types import CredentialType
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="ctx7",
            credential_type=CredentialType.CUSTOM,
            service_name="context7",
            data={
                "UPSTASH_REDIS_REST_URL": "https://example.upstash.io",
                "UPSTASH_REDIS_REST_TOKEN": "tok-abc",
            },
        )
        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry, vault=vault)

        cfg = MCPServerConfig(
            id="s1", tenant_id="t1", name="ctx7-server", url="",
            transport="stdio", credential_id=rec.id,
        )
        patched = adapter._inject_credential(cfg, "t1")

        assert patched.env["UPSTASH_REDIS_REST_URL"] == "https://example.upstash.io"
        assert patched.env["UPSTASH_REDIS_REST_TOKEN"] == "tok-abc"
        assert cfg.env == {}  # original not mutated

    def test_explicit_env_overrides_credential_env(self):
        """Server-level env takes priority over credential env (intentional override)."""
        from nexus.types import CredentialType
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="cred",
            credential_type=CredentialType.CUSTOM,
            service_name="svc",
            data={"KEY": "from-vault"},
        )
        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry, vault=vault)

        cfg = MCPServerConfig(
            id="s2", tenant_id="t1", name="svc", url="",
            transport="stdio", credential_id=rec.id,
            env={"KEY": "explicit-override"},
        )
        patched = adapter._inject_credential(cfg, "t1")
        assert patched.env["KEY"] == "explicit-override"

    def test_no_vault_skips_injection(self):
        """If adapter has no vault, credential_id is silently ignored."""
        registry = ToolRegistry()
        no_vault_adapter = MCPToolAdapter(registry, vault=None)
        MCPServerConfig(
            id="s3", tenant_id="t1", name="svc", url="",
            transport="stdio", credential_id="some-id",
        )
        # _inject_credential is only called when vault is set; adapter.register_server
        # guards on: if server_config.credential_id and self._vault is not None
        assert no_vault_adapter._vault is None

    def test_no_credential_id_skips_injection(self):
        """If server_config has no credential_id, env is not modified."""
        vault = _make_vault()
        registry = ToolRegistry()
        MCPToolAdapter(registry, vault=vault)

        no_cred_cfg = MCPServerConfig(
            id="s4", tenant_id="t1", name="svc", url="",
            transport="stdio",  # credential_id=None (default)
        )
        assert no_cred_cfg.credential_id is None

    def test_inject_credential_raises_on_tenant_mismatch(self):
        from nexus.types import CredentialType
        from nexus.exceptions import CredentialNotFound
        vault = _make_vault()
        rec = vault.store(
            tenant_id="t1",
            name="cred",
            credential_type=CredentialType.CUSTOM,
            service_name="svc",
            data={"KEY": "val"},
        )
        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry, vault=vault)

        cfg = MCPServerConfig(
            id="s5", tenant_id="wrong-tenant", name="svc", url="",
            transport="stdio", credential_id=rec.id,
        )
        with pytest.raises(CredentialNotFound):
            adapter._inject_credential(cfg, "wrong-tenant")


@pytest.mark.asyncio
class TestMCPAdapterCredentialIntegration:
    """Integration: real subprocess receives vault credentials via env var injection."""

    async def test_injected_env_var_visible_to_mcp_server(self):
        """Store a CUSTOM credential with NEXUS_TEST_SECRET, connect to env server,
        call get_secret() and verify the server read it from its subprocess env."""
        from nexus.types import CredentialType

        vault = _make_vault()
        rec = vault.store(
            tenant_id="t-cred",
            name="test-secret",
            credential_type=CredentialType.CUSTOM,
            service_name="nexus-env-server",
            data={"NEXUS_TEST_SECRET": "vault-injected-value"},
        )

        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry, vault=vault)

        cfg = _env_server_cfg("env-srv-1")
        cfg = cfg.model_copy(update={"credential_id": rec.id})

        try:
            await adapter.register_server("t-cred", cfg)

            tool_fn = registry._implementations.get("mcp_nexus_env_server_get_secret")
            assert tool_fn is not None, "Tool was not registered in registry"

            result = await tool_fn()
            # FastMCP wraps scalar returns in {"result": value}
            actual = result.get("result") if isinstance(result, dict) else result
            assert actual == "vault-injected-value", (
                f"Expected 'vault-injected-value', got {result!r}. "
                "Credential injection did not reach the subprocess env."
            )
        finally:
            await adapter._client.disconnect_all()

    async def test_multiple_env_vars_injected(self):
        """Multiple env vars from a CUSTOM credential all reach the subprocess."""
        from nexus.types import CredentialType

        vault = _make_vault()
        rec = vault.store(
            tenant_id="t-cred",
            name="multi-secret",
            credential_type=CredentialType.CUSTOM,
            service_name="nexus-env-server",
            data={
                "NEXUS_TEST_SECRET": "primary-secret",
                "NEXUS_TEST_SECONDARY": "secondary-value",
            },
        )

        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry, vault=vault)

        cfg = _env_server_cfg("env-srv-2")
        cfg = cfg.model_copy(update={"credential_id": rec.id})

        try:
            await adapter.register_server("t-cred", cfg)

            tool_fn = registry._implementations.get("mcp_nexus_env_server_get_all_test_vars")
            assert tool_fn is not None

            result = await tool_fn()
            # FastMCP may return dict as structuredContent directly or wrap in {"result": ...}
            import json as _json
            if isinstance(result, str):
                result = _json.loads(result)
            elif isinstance(result, dict) and set(result.keys()) == {"result"}:
                result = result["result"]
                if isinstance(result, str):
                    result = _json.loads(result)
            assert result["NEXUS_TEST_SECRET"] == "primary-secret"
            assert result["NEXUS_TEST_SECONDARY"] == "secondary-value"
        finally:
            await adapter._client.disconnect_all()

    async def test_server_without_credential_id_starts_normally(self):
        """Servers with no credential_id still connect and function correctly."""
        registry = ToolRegistry()
        adapter = MCPToolAdapter(registry, vault=_make_vault())

        cfg = _env_server_cfg("env-srv-3")
        assert cfg.credential_id is None

        try:
            defs = await adapter.register_server("t-cred", cfg)
            assert len(defs) >= 1

            tool_fn = registry._implementations.get("mcp_nexus_env_server_get_secret")
            assert tool_fn is not None
            result = await tool_fn()
            # FastMCP wraps scalar returns in {"result": value}
            actual = result.get("result") if isinstance(result, dict) else result
            assert actual == "__NOT_SET__"  # env var was not injected
        finally:
            await adapter._client.disconnect_all()
