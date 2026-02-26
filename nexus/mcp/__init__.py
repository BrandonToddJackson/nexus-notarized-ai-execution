"""MCP (Model Context Protocol) integration for NEXUS.

Enables connecting external MCP servers and exposing their tools
through the NEXUS tool registry, subject to the same anomaly gates
and persona contracts as built-in tools.
"""

from nexus.mcp.client import MCPClient
from nexus.mcp.adapter import MCPToolAdapter
from nexus.mcp.known_servers import (
    shadcn_server, nextjs_server, context7_server,
    sequential_thinking_server, KNOWN_SERVERS,
)

__all__ = [
    "MCPClient", "MCPToolAdapter",
    "shadcn_server", "nextjs_server", "context7_server",
    "sequential_thinking_server", "KNOWN_SERVERS",
]
