"""MCP (Model Context Protocol) integration for NEXUS.

Enables connecting external MCP servers and exposing their tools
through the NEXUS tool registry, subject to the same anomaly gates
and persona contracts as built-in tools.
"""

from nexus.mcp.client import MCPClient
from nexus.mcp.adapter import MCPToolAdapter

__all__ = ["MCPClient", "MCPToolAdapter"]
