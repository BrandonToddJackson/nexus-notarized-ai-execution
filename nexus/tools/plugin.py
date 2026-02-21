"""@tool decorator for registering functions as NEXUS tools.

Usage:
    @tool(name="web_search", description="Search the web", risk_level=RiskLevel.LOW)
    async def web_search(query: str) -> str:
        ...

Auto-generates ToolDefinition from function signature + type hints.
"""

import inspect
import functools
from typing import Any, Callable

from nexus.types import ToolDefinition, RiskLevel

# Global registry for decorated tools — collected at import time
_registered_tools: dict[str, tuple[ToolDefinition, Callable[..., Any]]] = {}


def tool(
    name: str = None,
    description: str = None,
    risk_level: RiskLevel = RiskLevel.LOW,
    resource_pattern: str = "*",
    timeout_seconds: int = 30,
    requires_approval: bool = False,
):
    """Decorator to register a function as a NEXUS tool.

    Auto-generates ToolDefinition from function signature + type hints.
    Extracts JSON Schema from type annotations for parameters.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        risk_level: Risk classification
        resource_pattern: What resources this tool accesses
        timeout_seconds: Execution timeout
        requires_approval: Whether human approval is needed
    """
    def decorator(func):
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip().split("\n")[0]

        # Extract parameters from function signature
        sig = inspect.signature(func)
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            param_type = "string"  # default
            if param.annotation != inspect.Parameter.empty:
                type_map = {str: "string", int: "integer", float: "number",
                            bool: "boolean", list: "array", dict: "object"}
                param_type = type_map.get(param.annotation, "string")
            params[param_name] = {
                "type": param_type,
                "description": f"Parameter: {param_name}",
            }

        definition = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters={"type": "object", "properties": params,
                         "required": [p for p in params if sig.parameters[p].default == inspect.Parameter.empty]},
            risk_level=risk_level,
            resource_pattern=resource_pattern,
            timeout_seconds=timeout_seconds,
            requires_approval=requires_approval,
        )

        _registered_tools[tool_name] = (definition, func)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper._nexus_tool = definition
        return wrapper

    return decorator


def get_registered_tools() -> dict[str, tuple[ToolDefinition, Callable[..., Any]]]:
    """Return all tools registered via @tool decorator."""
    return _registered_tools.copy()
