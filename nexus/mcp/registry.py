"""MCPServerRegistry — manages MCP server connections and auto-registers their tools.

Loaded at startup from config.mcp_servers list. Each server's tools are
registered into the shared ToolRegistry under the mcp__{server}__{tool}
namespace so they are first-class NEXUS tools.

Graceful degradation: if any server is unavailable at startup, a WARNING is
logged and startup continues. The engine never fails because of a missing
MCP server.
"""

import logging
from typing import Any

from nexus.mcp.client import MCPClient, MCPClientError
from nexus.mcp.adapter import MCPToolAdapter

logger = logging.getLogger(__name__)


class MCPServerRegistry:
    """Manages MCP server connections and registers their tools into ToolRegistry.

    Args:
        config: NexusConfig instance (reads config.mcp_servers)
        adapter: MCPToolAdapter instance for converting specs to NEXUS tools
    """

    def __init__(self, config: Any, adapter: MCPToolAdapter = None):
        self._config = config
        self._adapter = adapter or MCPToolAdapter()
        self._clients: dict[str, MCPClient] = {}  # server_name → MCPClient

    async def load_all(self, tool_registry: Any) -> None:
        """Connect to all configured MCP servers and register their tools.

        Called once during application startup (lifespan). Servers that are
        unavailable are skipped with a warning — they do not block startup.

        Args:
            tool_registry: ToolRegistry instance to register tools into
        """
        server_configs = getattr(self._config, "mcp_servers", []) or []

        if not server_configs:
            logger.debug("No MCP servers configured (NEXUS_MCP_SERVERS not set)")
            return

        total_tools = 0
        for server_cfg in server_configs:
            name = getattr(server_cfg, "name", "<unnamed>")
            try:
                client = MCPClient(server_cfg)
                await client.connect()
                tools = await client.list_tools()

                for spec in tools:
                    definition, fn = self._adapter.adapt(spec, client)
                    tool_registry.register(definition, fn)
                    total_tools += 1

                self._clients[name] = client
                logger.info(
                    "MCP server '%s': %d tool(s) registered (%s)",
                    name, len(tools),
                    ", ".join(f"mcp__{spec.server_name}__{spec.name}" for spec in tools[:5])
                    + ("…" if len(tools) > 5 else ""),
                )

            except (MCPClientError, Exception) as exc:
                logger.warning(
                    "MCP server '%s' unavailable — skipping: %s", name, exc
                )

        if total_tools:
            logger.info("MCP: %d total tool(s) registered from %d server(s)", total_tools, len(self._clients))

    async def shutdown(self) -> None:
        """Gracefully disconnect all MCP servers."""
        for name, client in list(self._clients.items()):
            try:
                await client.disconnect()
                logger.debug("MCP server '%s' disconnected", name)
            except Exception as exc:
                logger.warning("MCP server '%s' disconnect error: %s", name, exc)
        self._clients.clear()

    def get_client(self, server_name: str) -> MCPClient | None:
        """Return the active client for a server, or None if not connected."""
        return self._clients.get(server_name)

    def list_connected(self) -> list[str]:
        """Return names of currently connected MCP servers."""
        return list(self._clients.keys())
