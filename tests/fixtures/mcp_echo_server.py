#!/usr/bin/env python
"""Minimal MCP server for integration testing.

Built with FastMCP (part of the official modelcontextprotocol Python SDK).
Exposes three deterministic tools so integration tests can verify the full
stdio transport stack end-to-end without network I/O.

Run standalone:
    python tests/fixtures/mcp_echo_server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("nexus-test-echo-server")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the input text back unchanged."""
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


@mcp.tool()
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run()
