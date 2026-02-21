"""Central registry of all available tools."""

from typing import Any, Callable

from nexus.types import ToolDefinition, PersonaContract
from nexus.exceptions import ToolError


class ToolRegistry:
    """Central registry of all available tools."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._implementations: dict[str, Callable[..., Any]] = {}

    def register(self, definition: ToolDefinition, implementation: Callable[..., Any]) -> None:
        """Register a tool with its definition and implementation function.

        Args:
            definition: Tool metadata and schema
            implementation: Async callable that executes the tool
        """
        self._tools[definition.name] = definition
        self._implementations[definition.name] = implementation

    def get(self, name: str) -> tuple[ToolDefinition, Callable[..., Any]]:
        """Get tool definition and implementation.

        Args:
            name: Tool name

        Returns:
            Tuple of (ToolDefinition, implementation callable)

        Raises:
            ToolError: if tool not found
        """
        if name not in self._tools:
            raise ToolError(f"Tool \'{name}\' not found in registry", tool_name=name)
        return self._tools[name], self._implementations[name]

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_for_persona(self, persona: PersonaContract) -> list[ToolDefinition]:
        """List tools available to a specific persona.

        Args:
            persona: Persona contract with allowed_tools list

        Returns:
            Filtered list of tool definitions
        """
        return [t for t in self._tools.values() if t.name in persona.allowed_tools]

    def get_schema_for_llm(self, persona: PersonaContract) -> list[dict]:
        """Format tool definitions for LLM function calling.

        Only includes tools in persona.allowed_tools.
        Returns list of dicts matching OpenAI/Anthropic tool format.

        Args:
            persona: Persona contract to filter tools for

        Returns:
            List of tool schemas for LLM
        """
        schemas = []
        for tool in self._tools.values():
            if tool.name in persona.allowed_tools:
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                })
        return schemas
