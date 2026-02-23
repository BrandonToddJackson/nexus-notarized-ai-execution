"""MCPClient — manages connections to MCP tool servers.

Supports three transports:
  - stdio:            subprocess via StdioServerParameters
  - sse:              HTTP+SSE via sse_client
  - streamable_http:  HTTP streaming via streamable_http_client

Tool names are namespaced as ``mcp_{server_name}_{tool_name}`` with
non-alphanumeric characters replaced by underscores.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import AsyncExitStack
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from nexus.exceptions import MCPConnectionError, MCPToolError
from nexus.types import MCPServerConfig, RiskLevel, ToolDefinition

logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    """Replace non-alphanumeric chars with underscores."""
    return re.sub(r"[^a-zA-Z0-9]", "_", s)


def make_tool_name(server_name: str, tool_name: str) -> str:
    """Build the NEXUS-namespaced tool name for an MCP tool."""
    return f"mcp_{_normalize(server_name)}_{_normalize(tool_name)}"


class MCPClient:
    """Manages live sessions to one or more MCP servers."""

    def __init__(self) -> None:
        # server_id → ClientSession
        self._sessions: dict[str, ClientSession] = {}
        # server_id → AsyncExitStack (keeps transport alive)
        self._exit_stacks: dict[str, AsyncExitStack] = {}
        # server_id → MCPServerConfig
        self._configs: dict[str, MCPServerConfig] = {}
        # server_id → {namespaced_name: original_mcp_name}
        self._tool_name_map: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self, server_config: MCPServerConfig) -> list[ToolDefinition]:
        """Connect to an MCP server and discover its tools.

        Args:
            server_config: Server configuration (transport, url/command, etc.)

        Returns:
            List of ToolDefinitions ready to register in the NEXUS registry.

        Raises:
            MCPConnectionError: If the connection or tool listing fails.
        """
        server_id = server_config.id
        if server_id in self._sessions:
            logger.info("MCP server %s already connected — refreshing", server_config.name)
            return await self.refresh_tools(server_id)

        stack = AsyncExitStack()
        try:
            session = await self._open_session(stack, server_config)
            await session.initialize()
            tools_result = await session.list_tools()
        except Exception as exc:
            await stack.aclose()
            raise MCPConnectionError(
                f"Failed to connect to MCP server '{server_config.name}': {exc}",
                server_name=server_config.name,
            ) from exc

        self._sessions[server_id] = session
        self._exit_stacks[server_id] = stack
        self._configs[server_id] = server_config

        definitions, name_map = self._convert_tools(server_config.name, tools_result.tools)
        self._tool_name_map[server_id] = name_map

        logger.info(
            "Connected to MCP server '%s' (%s); discovered %d tools",
            server_config.name,
            server_config.transport,
            len(definitions),
        )
        return definitions

    async def _open_session(
        self, stack: AsyncExitStack, cfg: MCPServerConfig
    ) -> ClientSession:
        """Enter the appropriate transport context and return a ClientSession."""
        transport = cfg.transport.lower()

        if transport == "stdio":
            if not cfg.command:
                raise MCPConnectionError(
                    f"stdio transport requires 'command' for server '{cfg.name}'",
                    server_name=cfg.name,
                )
            # Merge extra env vars on top of the inherited environment so
            # the subprocess keeps PATH, PYTHONPATH, etc. while accepting
            # overrides like SSL_CERT_FILE.
            merged_env = {**os.environ, **cfg.env} if cfg.env else None
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args,
                env=merged_env,
            )
            read, write = await stack.enter_async_context(stdio_client(params))

        elif transport == "sse":
            read, write = await stack.enter_async_context(sse_client(url=cfg.url))

        elif transport in ("streamable_http", "streamable-http", "http"):
            read, write, _ = await stack.enter_async_context(
                streamable_http_client(url=cfg.url)
            )

        else:
            raise MCPConnectionError(
                f"Unknown transport '{cfg.transport}' for server '{cfg.name}'",
                server_name=cfg.name,
            )

        session = await stack.enter_async_context(ClientSession(read, write))
        return session

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def _convert_tools(
        self, server_name: str, mcp_tools: list[Any]
    ) -> tuple[list[ToolDefinition], dict[str, str]]:
        """Convert MCP Tool objects to NEXUS ToolDefinitions.

        Returns:
            (definitions, name_map) where name_map is
            {namespaced_name: original_mcp_name}.
        """
        definitions: list[ToolDefinition] = []
        name_map: dict[str, str] = {}

        for tool in mcp_tools:
            namespaced = make_tool_name(server_name, tool.name)
            name_map[namespaced] = tool.name

            defn = ToolDefinition(
                name=namespaced,
                description=tool.description or f"MCP tool '{tool.name}' from '{server_name}'",
                parameters=tool.inputSchema or {"type": "object", "properties": {}},
                risk_level=RiskLevel.MEDIUM,
                resource_pattern=f"mcp:{_normalize(server_name)}:*",
                timeout_seconds=30,
                requires_approval=False,
            )
            definitions.append(defn)

        return definitions, name_map

    async def refresh_tools(self, server_id: str) -> list[ToolDefinition]:
        """Re-list tools for an already-connected server.

        Args:
            server_id: Server config ID.

        Returns:
            Updated list of ToolDefinitions.

        Raises:
            MCPConnectionError: If the server is not connected or listing fails.
        """
        if server_id not in self._sessions:
            raise MCPConnectionError(
                f"Server '{server_id}' is not connected",
                server_name=server_id,
            )
        cfg = self._configs[server_id]
        try:
            tools_result = await self._sessions[server_id].list_tools()
        except Exception as exc:
            raise MCPConnectionError(
                f"Failed to refresh tools for '{cfg.name}': {exc}",
                server_name=cfg.name,
            ) from exc

        definitions, name_map = self._convert_tools(cfg.name, tools_result.tools)
        self._tool_name_map[server_id] = name_map
        return definitions

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def call_tool(
        self, server_id: str, namespaced_tool_name: str, params: dict[str, Any]
    ) -> Any:
        """Execute an MCP tool.

        Args:
            server_id: Server config ID.
            namespaced_tool_name: NEXUS-namespaced name (``mcp_{server}_{tool}``).
            params: Tool parameters.

        Returns:
            Extracted result — structured content dict if available, else
            concatenated text from TextContent items.

        Raises:
            MCPToolError: If the tool execution fails or returns an error.
            MCPConnectionError: If the server is not connected.
        """
        if server_id not in self._sessions:
            raise MCPConnectionError(
                f"Server '{server_id}' is not connected",
                server_name=server_id,
            )

        name_map = self._tool_name_map.get(server_id, {})
        original_name = name_map.get(namespaced_tool_name, namespaced_tool_name)

        try:
            result = await self._sessions[server_id].call_tool(original_name, params)
        except Exception as exc:
            raise MCPToolError(
                f"MCP tool '{original_name}' raised: {exc}",
                tool_name=namespaced_tool_name,
            ) from exc

        if result.isError:
            error_text = self._extract_text(result.content)
            raise MCPToolError(
                f"MCP tool '{original_name}' returned error: {error_text}",
                tool_name=namespaced_tool_name,
            )

        if result.structuredContent:
            return result.structuredContent

        return self._extract_text(result.content)

    @staticmethod
    def _extract_text(content: list[Any]) -> str:
        """Join text from TextContent items."""
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
        return "\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def disconnect(self, server_id: str) -> None:
        """Close the connection to a single MCP server.

        Args:
            server_id: Server config ID.
        """
        stack = self._exit_stacks.pop(server_id, None)
        self._sessions.pop(server_id, None)
        self._configs.pop(server_id, None)
        self._tool_name_map.pop(server_id, None)
        if stack:
            await stack.aclose()
            logger.info("Disconnected MCP server '%s'", server_id)

    async def disconnect_all(self) -> None:
        """Close all active MCP server connections."""
        for server_id in list(self._sessions.keys()):
            await self.disconnect(server_id)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_connected(self, server_id: str) -> bool:
        """Return True if the server has an active session."""
        return server_id in self._sessions

    def list_connected(self) -> list[str]:
        """Return IDs of all currently connected servers."""
        return list(self._sessions.keys())

    def get_tool_name_map(self, server_id: str) -> dict[str, str]:
        """Return the {namespaced → original} tool name map for a server."""
        return dict(self._tool_name_map.get(server_id, {}))
