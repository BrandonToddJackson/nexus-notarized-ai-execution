"""Pre-configured MCPServerConfig factories for well-known MCP servers.

Usage:
    from nexus.mcp.known_servers import shadcn_server, nextjs_server, context7_server

    await adapter.register_server(tenant_id="...", server_config=shadcn_server("t1"))
    await adapter.register_server(tenant_id="...", server_config=nextjs_server("t1"))
"""

from nexus.types import MCPServerConfig


def shadcn_server(tenant_id: str, **kwargs) -> MCPServerConfig:
    """shadcn/ui component registry — list, search, view, install components."""
    return MCPServerConfig(
        id="shadcn", tenant_id=tenant_id, name="shadcn", url="",
        transport="stdio", command="npx", args=["shadcn@latest", "mcp"], **kwargs,
    )


def nextjs_server(tenant_id: str, **kwargs) -> MCPServerConfig:
    """Next.js devtools — errors, logs, routes, server actions from a running dev server."""
    return MCPServerConfig(
        id="next-devtools", tenant_id=tenant_id, name="next-devtools", url="",
        transport="stdio", command="npx", args=["-y", "next-devtools-mcp@latest"], **kwargs,
    )


def context7_server(tenant_id: str, api_key: str = "", **kwargs) -> MCPServerConfig:
    """Context7 — up-to-date library docs for any npm/PyPI package."""
    return MCPServerConfig(
        id="context7", tenant_id=tenant_id, name="context7", url="",
        transport="stdio", command="npx", args=["-y", "@upstash/context7-mcp"],
        env={"CONTEXT7_API_KEY": api_key} if api_key else {}, **kwargs,
    )


def sequential_thinking_server(tenant_id: str, **kwargs) -> MCPServerConfig:
    """Sequential thinking — step-by-step reasoning tool."""
    return MCPServerConfig(
        id="sequential-thinking", tenant_id=tenant_id, name="sequential-thinking", url="",
        transport="stdio", command="npx",
        args=["-y", "@modelcontextprotocol/server-sequential-thinking"], **kwargs,
    )


# ── Export all known servers ───────────────────────────────────────────────────

KNOWN_SERVERS = {
    "shadcn":              shadcn_server,
    "nextjs":              nextjs_server,
    "context7":            context7_server,
    "sequential-thinking": sequential_thinking_server,
}

__all__ = [
    "shadcn_server", "nextjs_server", "context7_server",
    "sequential_thinking_server", "KNOWN_SERVERS",
]
