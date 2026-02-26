"""MCPToolAdapter — converts MCP tool specs into NEXUS (ToolDefinition, callable) pairs.

Each MCP tool becomes a first-class NEXUS tool that passes through all 4
accountability gates, gets sealed by the notary, and is recorded in the ledger.

Naming convention:
    mcp__{server_name}__{tool_name}

This namespace prefix:
- Prevents collisions with built-in NEXUS tools
- Enables wildcard grants in personas.yaml (e.g. allowed_tools: ["mcp__slack__*"])
- Makes the origin of each tool immediately visible in seal records

Risk inference:
    - Tools with names containing write/send/post/create/delete/update → MEDIUM/HIGH
    - Everything else → LOW
    - Tools with names containing delete/remove → HIGH
"""

from typing import Any, Callable

from nexus.types import ToolDefinition, RiskLevel
from nexus.mcp.client import MCPToolSpec, MCPClient


# Keywords that indicate write/mutation operations
_WRITE_KEYWORDS = {"write", "send", "post", "create", "update", "put", "patch", "set", "push"}
_DELETE_KEYWORDS = {"delete", "remove", "destroy", "drop", "purge"}


class MCPToolAdapter:
    """Converts MCP tool specs into NEXUS (ToolDefinition, callable) pairs.

    Usage:
        adapter = MCPToolAdapter()
        definition, fn = adapter.adapt(spec, client)
        tool_registry.register(definition, fn)
    """

    def adapt(self, spec: MCPToolSpec, client: MCPClient) -> tuple[ToolDefinition, Callable]:
        """Convert an MCPToolSpec into a NEXUS ToolDefinition and async callable.

        The callable routes through the MCPClient so execution is audited.

        Args:
            spec: Tool specification from MCP server discovery
            client: Connected MCPClient instance

        Returns:
            Tuple of (ToolDefinition, async callable)
        """
        nexus_name = f"mcp__{spec.server_name}__{spec.name}"
        risk = self._infer_risk(spec.name)

        definition = ToolDefinition(
            name=nexus_name,
            description=f"[MCP:{spec.server_name}] {spec.description}",
            parameters=spec.input_schema or {},
            risk_level=risk,
            resource_pattern=f"mcp:{spec.server_name}:*",
        )

        # Capture spec/client in closure — each tool has its own bound call
        _tool_name = spec.name
        _client = client

        async def call(**kwargs: Any) -> Any:
            return await _client.call_tool(_tool_name, kwargs)

        return definition, call

    @staticmethod
    def _infer_risk(tool_name: str) -> RiskLevel:
        """Infer risk level from tool name keywords.

        HIGH  → delete/remove/destroy/drop/purge
        MEDIUM → write/send/post/create/update/put/patch/set/push
        LOW   → everything else (read-only assumed)
        """
        lower = tool_name.lower()
        if any(kw in lower for kw in _DELETE_KEYWORDS):
            return RiskLevel.HIGH
        if any(kw in lower for kw in _WRITE_KEYWORDS):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
