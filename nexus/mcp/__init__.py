"""MCP (Model Context Protocol) adapter layer for NEXUS.

Any service that exposes an MCP server becomes instantly available as a
NEXUS tool — with zero changes to the engine, gates, notary, or ledger.

Architecture:
    External MCP Server → MCPClient → MCPToolAdapter → ToolRegistry → 4 Gates → Notary → Ledger
    (Slack, GitHub…)     (connect)    (wraps tools)    (register)     (full)    (seal)  (audit)

Usage (config/.env):
    NEXUS_MCP_SERVERS='[
      {"name": "slack", "transport": "stdio",
       "command": "npx @modelcontextprotocol/server-slack",
       "env": {"SLACK_TOKEN": "xoxb-..."}},
      {"name": "github", "transport": "sse", "url": "http://localhost:8080"}
    ]'

At startup, MCPServerRegistry.load_all(tool_registry) discovers and registers
all tools from each configured server. Tools appear as mcp__{server}__{tool}
in the registry and go through the full 4-gate accountability pipeline.
"""

from nexus.mcp.client import MCPClient, MCPToolSpec
from nexus.mcp.adapter import MCPToolAdapter
from nexus.mcp.registry import MCPServerRegistry

__all__ = ["MCPClient", "MCPToolSpec", "MCPToolAdapter", "MCPServerRegistry"]
