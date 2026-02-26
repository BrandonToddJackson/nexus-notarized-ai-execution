#!/usr/bin/env python
"""MCP server that exposes an env var via a tool — used to prove credential injection.

The `get_secret` tool reads NEXUS_TEST_SECRET from the subprocess environment and
returns it.  When MCPToolAdapter correctly injects vault credentials as env vars
before spawning this server, the tool returns the injected secret value.

Run standalone:
    NEXUS_TEST_SECRET=hello python tests/fixtures/mcp_env_server.py
"""

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("nexus-env-server")


@mcp.tool()
def get_secret() -> str:
    """Return the value of NEXUS_TEST_SECRET from the subprocess environment."""
    return os.environ.get("NEXUS_TEST_SECRET", "__NOT_SET__")


@mcp.tool()
def get_all_test_vars() -> dict:
    """Return all env vars that start with NEXUS_TEST_."""
    return {k: v for k, v in os.environ.items() if k.startswith("NEXUS_TEST_")}


if __name__ == "__main__":
    mcp.run()
