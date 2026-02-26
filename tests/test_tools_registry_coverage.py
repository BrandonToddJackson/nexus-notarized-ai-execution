"""Tests for ToolRegistry.unregister() and get_by_source() — Gap 17.

Coverage:
  unregister() removes tool from all three internal dicts
  unregister() on nonexistent name is a no-op
  get_by_source() returns only tools with matching source label
  get_by_source() returns empty list after unregister
"""

import pytest
from nexus.tools.registry import ToolRegistry
from nexus.types import ToolDefinition, RiskLevel
from nexus.exceptions import ToolError


def _make_tool(name: str) -> tuple[ToolDefinition, object]:
    defn = ToolDefinition(
        name=name,
        description=f"Test tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
        risk_level=RiskLevel.LOW,
        resource_pattern="*",
    )

    async def _impl() -> str:
        return f"result of {name}"

    return defn, _impl


class TestUnregister:

    def test_unregister_removes_tool(self):
        reg = ToolRegistry()
        defn, impl = _make_tool("my_tool")
        reg.register(defn, impl, source="local")
        reg.unregister("my_tool")
        with pytest.raises(ToolError):
            reg.get("my_tool")

    def test_unregister_nonexistent_is_noop(self):
        """Unregistering a tool that was never registered must not raise."""
        reg = ToolRegistry()
        reg.unregister("does_not_exist")  # Must not raise

    def test_unregister_removes_from_list_tools(self):
        reg = ToolRegistry()
        defn, impl = _make_tool("temp_tool")
        reg.register(defn, impl)
        reg.unregister("temp_tool")
        names = [t.name for t in reg.list_tools()]
        assert "temp_tool" not in names

    def test_unregister_second_time_is_noop(self):
        """Double-unregister must not raise."""
        reg = ToolRegistry()
        defn, impl = _make_tool("one_shot")
        reg.register(defn, impl)
        reg.unregister("one_shot")
        reg.unregister("one_shot")  # second call — must not raise


class TestGetBySource:

    def test_get_by_source_returns_only_matching(self):
        reg = ToolRegistry()
        local_def, local_impl = _make_tool("local_tool")
        mcp_def, mcp_impl = _make_tool("mcp_tool")
        reg.register(local_def, local_impl, source="local")
        reg.register(mcp_def, mcp_impl, source="mcp")

        local = reg.get_by_source("local")
        mcp_tools = reg.get_by_source("mcp")

        assert len(local) == 1
        assert local[0].name == "local_tool"
        assert len(mcp_tools) == 1
        assert mcp_tools[0].name == "mcp_tool"

    def test_get_by_source_empty_when_none_registered(self):
        reg = ToolRegistry()
        assert reg.get_by_source("mcp") == []

    def test_get_by_source_empty_after_unregister(self):
        reg = ToolRegistry()
        defn, impl = _make_tool("mcp_tool")
        reg.register(defn, impl, source="mcp")
        reg.unregister("mcp_tool")
        assert reg.get_by_source("mcp") == []

    def test_get_by_source_unknown_source_returns_empty(self):
        reg = ToolRegistry()
        defn, impl = _make_tool("local_tool")
        reg.register(defn, impl, source="local")
        assert reg.get_by_source("nonexistent_source") == []

    def test_get_by_source_multiple_tools_same_source(self):
        reg = ToolRegistry()
        for name in ("tool_a", "tool_b", "tool_c"):
            defn, impl = _make_tool(name)
            reg.register(defn, impl, source="mcp")

        mcp_tools = reg.get_by_source("mcp")
        names = {t.name for t in mcp_tools}
        assert names == {"tool_a", "tool_b", "tool_c"}
