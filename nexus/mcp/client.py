"""MCPClient — connects to an MCP server and discovers/calls its tools.

Supports two transports:
- stdio: spawns a subprocess (e.g. npx @modelcontextprotocol/server-slack)
- sse / streamable_http: connects to an HTTP endpoint

The MCP protocol is JSON-RPC 2.0. This client implements:
  - initialize  (negotiation)
  - tools/list  (discovery)
  - tools/call  (execution)

Reconnect on failure: the client attempts reconnect up to max_retries times
before raising. This makes it safe to use in the lifespan even if an MCP
server is temporarily unavailable.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# MCP JSON-RPC protocol constants
_JSONRPC = "2.0"
_METHOD_INIT = "initialize"
_METHOD_LIST_TOOLS = "tools/list"
_METHOD_CALL_TOOL = "tools/call"


@dataclass
class MCPToolSpec:
    """Specification for a tool discovered from an MCP server."""
    name: str
    server_name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


class MCPClientError(Exception):
    """Raised when the MCP client cannot connect or a call fails."""


class MCPClient:
    """Connects to an MCP server and wraps its tools.

    Args:
        server_config: MCPServerConfig from NexusConfig.mcp_servers
        max_retries: Reconnection attempts before giving up (default 2)
    """

    def __init__(self, server_config: Any, max_retries: int = 2):
        self._config = server_config
        self._max_retries = max_retries
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._request_id = 0
        self._connected = False

    @property
    def server_name(self) -> str:
        return self._config.name

    async def connect(self) -> None:
        """Establish connection to the MCP server and run initialize handshake."""
        transport = getattr(self._config, "transport", "stdio")

        for attempt in range(1, self._max_retries + 2):
            try:
                if transport == "stdio":
                    await self._connect_stdio()
                else:
                    await self._connect_http()
                await self._initialize()
                self._connected = True
                logger.info("MCP server '%s' connected (transport=%s)", self.server_name, transport)
                return
            except Exception as exc:
                if attempt > self._max_retries:
                    raise MCPClientError(
                        f"MCP server '{self.server_name}' failed to connect after "
                        f"{self._max_retries + 1} attempt(s): {exc}"
                    ) from exc
                logger.warning(
                    "MCP server '%s' connect attempt %d failed: %s — retrying",
                    self.server_name, attempt, exc,
                )
                await asyncio.sleep(0.5 * attempt)

    async def list_tools(self) -> list[MCPToolSpec]:
        """Discover tools from the MCP server.

        Returns:
            List of MCPToolSpec for all tools the server exposes.
        """
        if not self._connected:
            raise MCPClientError(f"MCP server '{self.server_name}' not connected")

        response = await self._send_request(_METHOD_LIST_TOOLS, {})
        tools_data = response.get("result", {}).get("tools", [])

        specs = []
        for t in tools_data:
            specs.append(MCPToolSpec(
                name=t.get("name", ""),
                server_name=self.server_name,
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            ))
        return specs

    async def call_tool(self, name: str, params: dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Args:
            name: Tool name (as returned by list_tools)
            params: Tool parameters

        Returns:
            Tool result (may be str, dict, list, etc.)
        """
        if not self._connected:
            raise MCPClientError(f"MCP server '{self.server_name}' not connected")

        response = await self._send_request(
            _METHOD_CALL_TOOL,
            {"name": name, "arguments": params},
        )

        if "error" in response:
            err = response["error"]
            raise MCPClientError(
                f"MCP tool '{name}' error {err.get('code')}: {err.get('message')}"
            )

        result = response.get("result", {})
        # MCP returns content array; extract text for simple cases
        content = result.get("content", [])
        if content and isinstance(content, list):
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            if texts:
                return "\n".join(texts)
        return result

    async def disconnect(self) -> None:
        """Close connection and terminate subprocess if applicable."""
        self._connected = False
        transport = getattr(self._config, "transport", "stdio")
        if transport == "stdio" and self._proc is not None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except Exception:
                pass
            self._proc = None
        elif self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None

    # ── Private transport implementations ────────────────────────────────────

    async def _connect_stdio(self) -> None:
        """Spawn the MCP server as a subprocess using stdio transport."""
        command = getattr(self._config, "command", None)
        if not command:
            raise MCPClientError(
                f"MCP server '{self.server_name}': stdio transport requires 'command'"
            )

        env = dict(os.environ)
        env.update(getattr(self._config, "env", {}) or {})

        timeout = getattr(self._config, "timeout_seconds", 30)
        self._proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._reader = self._proc.stdout
        self._writer = self._proc.stdin
        # Brief wait for server to start
        await asyncio.sleep(0.2)

    async def _connect_http(self) -> None:
        """Connect to an SSE/HTTP MCP server. Placeholder for HTTP transport."""
        url = getattr(self._config, "url", None)
        if not url:
            raise MCPClientError(
                f"MCP server '{self.server_name}': sse/http transport requires 'url'"
            )
        # HTTP transport uses a simpler request/response model
        # Store the url for use in _send_request
        self._http_url = url

    async def _initialize(self) -> None:
        """Send MCP initialize request to negotiate capabilities."""
        response = await self._send_request(
            _METHOD_INIT,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "nexus", "version": "1.0"},
            },
        )
        if "error" in response:
            raise MCPClientError(
                f"MCP initialize failed for '{self.server_name}': {response['error']}"
            )

    async def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and receive the response."""
        self._request_id += 1
        req = {
            "jsonrpc": _JSONRPC,
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        transport = getattr(self._config, "transport", "stdio")
        timeout = getattr(self._config, "timeout_seconds", 30)

        if transport == "stdio":
            return await self._send_stdio(req, timeout)
        else:
            return await self._send_http(req, timeout)

    async def _send_stdio(self, req: dict, timeout: int) -> dict:
        """Send request over stdio pipe."""
        if self._writer is None or self._reader is None:
            raise MCPClientError(f"MCP '{self.server_name}': stdio pipe not open")

        line = json.dumps(req) + "\n"
        self._writer.write(line.encode())
        await self._writer.drain()

        try:
            raw = await asyncio.wait_for(self._reader.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            raise MCPClientError(
                f"MCP '{self.server_name}': timeout waiting for response to '{req['method']}'"
            )

        if not raw:
            raise MCPClientError(f"MCP '{self.server_name}': connection closed (empty response)")

        return json.loads(raw.decode().strip())

    async def _send_http(self, req: dict, timeout: int) -> dict:
        """Send request over HTTP (SSE/streamable_http transport)."""
        try:
            import httpx
        except ImportError:
            raise MCPClientError("httpx is required for HTTP MCP transport: pip install httpx")

        url = getattr(self, "_http_url", None) or getattr(self._config, "url", "")
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=req)
            resp.raise_for_status()
            return resp.json()
