"""Tests for tool registry, sandbox, executor, plugin decorator.

Phase 3: Execution Layer — real assertions against live implementations.
"""

import asyncio
import pytest

from nexus.types import ToolDefinition, RiskLevel, PersonaContract
from nexus.exceptions import ToolError
from nexus.tools.registry import ToolRegistry
from nexus.tools.sandbox import Sandbox
from nexus.tools.plugin import tool, get_registered_tools


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tool_def(name: str = "test_search") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Search for things",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        risk_level=RiskLevel.LOW,
    )


async def _echo(query: str) -> str:
    return f"echo: {query}"


# ── TestToolRegistry ───────────────────────────────────────────────────────────

class TestToolRegistry:

    def test_register_and_retrieve(self):
        registry = ToolRegistry()
        defn = _make_tool_def("my_tool")
        registry.register(defn, _echo)

        fetched_defn, fetched_fn = registry.get("my_tool")
        assert fetched_defn.name == "my_tool"
        assert fetched_defn.description == "Search for things"
        assert fetched_fn is _echo

    def test_unknown_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ToolError) as exc_info:
            registry.get("does_not_exist")
        assert "does_not_exist" in str(exc_info.value)

    def test_list_all_tools(self):
        registry = ToolRegistry()
        registry.register(_make_tool_def("tool_a"), _echo)
        registry.register(_make_tool_def("tool_b"), _echo)
        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert {"tool_a", "tool_b"} <= names

    def test_filter_by_persona(self, sample_personas):
        registry = ToolRegistry()
        # researcher allows: knowledge_search, web_search, web_fetch, file_read
        for name in ("knowledge_search", "web_search", "send_email"):
            registry.register(_make_tool_def(name), _echo)

        researcher = sample_personas[0]
        allowed = registry.list_for_persona(researcher)
        names = {t.name for t in allowed}
        assert "knowledge_search" in names
        assert "web_search" in names
        assert "send_email" not in names  # not in researcher.allowed_tools

    def test_get_schema_for_llm(self, sample_personas):
        registry = ToolRegistry()
        registry.register(_make_tool_def("knowledge_search"), _echo)
        registry.register(_make_tool_def("send_email"), _echo)

        researcher = sample_personas[0]
        schemas = registry.get_schema_for_llm(researcher)
        schema_names = {s["function"]["name"] for s in schemas}
        assert "knowledge_search" in schema_names
        assert "send_email" not in schema_names
        # Each schema entry is in OpenAI/Anthropic tool format
        for schema in schemas:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "description" in schema["function"]


# ── TestPluginDecorator ────────────────────────────────────────────────────────

class TestPluginDecorator:

    def test_decorator_registers_tool(self):
        @tool(name="nexus_test_echo_001", description="Echo tool for testing")
        async def _test_echo(query: str) -> str:
            """Echo the query."""
            return query

        registered = get_registered_tools()
        assert "nexus_test_echo_001" in registered
        defn, fn = registered["nexus_test_echo_001"]
        assert defn.name == "nexus_test_echo_001"
        assert defn.description == "Echo tool for testing"

    def test_schema_from_type_hints(self):
        @tool(name="nexus_test_typed_001", description="Typed tool")
        async def _typed_fn(query: str, limit: int) -> str:
            return query

        defn = _typed_fn._nexus_tool
        props = defn.parameters["properties"]
        assert props["query"]["type"] == "string"
        assert props["limit"]["type"] == "integer"

    def test_optional_param_not_required(self):
        @tool(name="nexus_test_optional_001", description="Optional params")
        async def _optional_fn(query: str, limit: int = 10) -> str:
            return query

        defn = _optional_fn._nexus_tool
        required = defn.parameters.get("required", [])
        assert "query" in required
        assert "limit" not in required

    def test_risk_level_propagated(self):
        @tool(name="nexus_test_risky_001", description="High risk", risk_level=RiskLevel.HIGH)
        async def _risky_fn(target: str) -> str:
            return target

        defn = _risky_fn._nexus_tool
        assert defn.risk_level == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_decorated_function_still_callable(self):
        @tool(name="nexus_test_callable_001", description="Callable")
        async def _callable_fn(query: str) -> str:
            return f"result: {query}"

        result = await _callable_fn(query="hello")
        assert result == "result: hello"


# ── TestSandbox ────────────────────────────────────────────────────────────────

class TestSandbox:

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        sandbox = Sandbox()
        result = await sandbox.execute(_echo, {"query": "hello"})
        assert result == "echo: hello"

    @pytest.mark.asyncio
    async def test_timeout_raises_tool_error(self):
        sandbox = Sandbox()

        async def slow_fn(query: str) -> str:
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(ToolError) as exc_info:
            await sandbox.execute(slow_fn, {"query": "test"}, timeout=0.01)
        assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_exception_wrapped_as_tool_error(self):
        sandbox = Sandbox()

        async def failing_fn(query: str) -> str:
            raise ValueError("Something went wrong inside the tool")

        with pytest.raises(ToolError) as exc_info:
            await sandbox.execute(failing_fn, {"query": "test"})
        assert "Something went wrong" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tool_error_propagated_unchanged(self):
        """A ToolError raised inside the tool should pass through, not be double-wrapped."""
        sandbox = Sandbox()

        async def tool_error_fn(query: str) -> str:
            raise ToolError("direct tool error", tool_name="tool_error_fn")

        with pytest.raises(ToolError) as exc_info:
            await sandbox.execute(tool_error_fn, {"query": "x"})
        assert "direct tool error" in str(exc_info.value)
