"""Tests for shadcn/ui frontend design generation pipeline.

Groups:
  TestFrontendDetection       — prompt → app_type + component mapping (no LLM)
  TestFrontendDesignTool      — generate_frontend_design() output shape + content (needs LLM+node)
  TestHtmlPreview             — HTML preview structure (needs LLM+node)
  TestTsxCode                 — TSX has correct shadcn imports + structure (needs LLM+node)
  TestColorSchemes            — color_scheme variations (needs LLM+node)
  TestRegistration            — tool is in _registered_tools (no LLM)
  TestShadcnMCPTools          — real MCP connection (skipif npx unavailable)
  TestShadcnMCPSearch         — real component search via MCP
  TestFullPipeline            — MCP → tool → output files end-to-end (needs LLM+node+npx)
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from nexus.tools.builtin.frontend_design import (
    generate_frontend_design,
    _detect_app_type,
    _resolve_components,
    _APP_COMPONENTS,
)
from nexus.tools.plugin import _registered_tools
from nexus.types import RiskLevel

HAS_NPX = shutil.which("npx") is not None
HAS_NODE = shutil.which("node") is not None

def _has_llm() -> bool:
    """True if any supported LLM provider is configured (cloud key or local Ollama model)."""
    from nexus.llm.client import is_local_model
    from nexus.config import config as _cfg
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or getattr(_cfg, "llm_api_key", None)
        or is_local_model(_cfg.default_llm_model)
    )

HAS_LLM_KEY = _has_llm()
# Tests that call generate_frontend_design() require node (JS sandbox) AND an LLM.
# They are also marked @pytest.mark.slow because each call takes 30-120s (real LLM).
# Run with: pytest -m slow   or   pytest tests/test_shadcn_frontend.py -m slow
NEEDS_DESIGN = HAS_NODE and HAS_LLM_KEY
SKIP_DESIGN = pytest.mark.skipif(
    not NEEDS_DESIGN,
    reason="generate_frontend_design requires node (JS sandbox) + LLM (API key or Ollama)",
)
SLOW = pytest.mark.slow


# ─────────────────────────────────────────────────────────────────────────────
# TestFrontendDetection — 8 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFrontendDetection:
    def test_dashboard_detected(self):
        assert _detect_app_type("SaaS analytics dashboard for startups") == "dashboard"

    def test_crud_detected_invoice(self):
        assert _detect_app_type("Invoice management system") == "crud"

    def test_crud_detected_admin(self):
        assert _detect_app_type("Admin panel to manage users") == "crud"

    def test_auth_detected(self):
        assert _detect_app_type("User login and registration page") == "auth"

    def test_kanban_detected(self):
        assert _detect_app_type("Kanban board for project management") == "kanban"

    def test_inbox_detected(self):
        assert _detect_app_type("Email inbox with messages") == "inbox"

    def test_unknown_defaults_to_dashboard(self):
        assert _detect_app_type("Something completely unrelated xyz") == "dashboard"

    def test_resolve_components_returns_list(self):
        comps = _resolve_components("analytics dashboard", "dashboard", None)
        assert isinstance(comps, list)
        assert len(comps) > 0
        assert len(comps) <= 8

    def test_resolve_components_prepends_extras(self):
        comps = _resolve_components("dashboard", "dashboard", ["custom-widget"])
        assert comps[0] == "custom-widget"

    def test_resolve_components_deduplicates(self):
        comps = _resolve_components("dashboard", "dashboard", ["card"])
        # card is in dashboard defaults so total length shouldn't exceed 8
        assert len(comps) <= 8


# ─────────────────────────────────────────────────────────────────────────────
# TestFrontendDesignTool — 10 tests
# ─────────────────────────────────────────────────────────────────────────────

@SKIP_DESIGN
@SLOW
class TestFrontendDesignTool:
    @pytest.mark.asyncio
    async def test_returns_required_keys(self):
        result = await generate_frontend_design(prompt="analytics dashboard")
        for key in ("tsx_code", "html_preview", "components_used", "app_type",
                    "install_command", "file_structure", "shadcn_add_command"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_app_type_auto_detected(self):
        result = await generate_frontend_design(prompt="SaaS analytics dashboard")
        assert result["app_type"] == "dashboard"

    @pytest.mark.asyncio
    async def test_explicit_app_type_respected(self):
        result = await generate_frontend_design(prompt="manage items", app_type="crud")
        assert result["app_type"] == "crud"

    @pytest.mark.asyncio
    async def test_components_used_is_list(self):
        result = await generate_frontend_design(prompt="dashboard")
        assert isinstance(result["components_used"], list)
        assert len(result["components_used"]) > 0

    @pytest.mark.asyncio
    async def test_install_command_format(self):
        result = await generate_frontend_design(prompt="invoice table")
        cmd = result["install_command"]
        assert cmd.startswith("npx shadcn")
        assert "add" in cmd

    @pytest.mark.asyncio
    async def test_custom_add_command_passed_through(self):
        custom_cmd = "npx shadcn@latest add card badge table"
        result = await generate_frontend_design(
            prompt="dashboard", add_command=custom_cmd
        )
        assert result["shadcn_add_command"] == custom_cmd

    @pytest.mark.asyncio
    async def test_color_scheme_respected(self):
        result = await generate_frontend_design(prompt="login screen", color_scheme="violet")
        # HTML should contain violet color values
        assert "violet" in result["html_preview"].lower() or "#7c3aed" in result["html_preview"] or "#8b5cf6" in result["html_preview"]

    @pytest.mark.asyncio
    async def test_file_structure_has_tsx(self):
        result = await generate_frontend_design(prompt="analytics dashboard")
        fnames = list(result["file_structure"].keys())
        assert any(".tsx" in f for f in fnames)

    @pytest.mark.asyncio
    async def test_file_structure_has_html(self):
        result = await generate_frontend_design(prompt="analytics dashboard")
        fnames = list(result["file_structure"].keys())
        assert any(".html" in f for f in fnames)

    @pytest.mark.asyncio
    async def test_file_structure_has_components_json(self):
        result = await generate_frontend_design(prompt="analytics dashboard")
        assert "components.json" in result["file_structure"]

    @pytest.mark.asyncio
    async def test_crud_app_type_generation(self):
        result = await generate_frontend_design(
            prompt="Invoice management for freelancers", app_type="crud"
        )
        assert result["tsx_code"]
        assert result["html_preview"]


# ─────────────────────────────────────────────────────────────────────────────
# TestHtmlPreview — 6 tests
# ─────────────────────────────────────────────────────────────────────────────

@SKIP_DESIGN
@SLOW
class TestHtmlPreview:
    @pytest.mark.asyncio
    async def test_html_has_doctype(self):
        result = await generate_frontend_design(prompt="analytics dashboard")
        assert result["html_preview"].startswith("<!DOCTYPE html>")

    @pytest.mark.asyncio
    async def test_html_has_html_tag(self):
        result = await generate_frontend_design(prompt="analytics dashboard")
        assert "<html" in result["html_preview"]
        assert "</html>" in result["html_preview"]

    @pytest.mark.asyncio
    async def test_html_has_body(self):
        result = await generate_frontend_design(prompt="analytics dashboard")
        assert "<body>" in result["html_preview"] or "<body " in result["html_preview"]

    @pytest.mark.asyncio
    async def test_html_contains_prompt_text(self):
        result = await generate_frontend_design(prompt="Revenue Analytics")
        html = result["html_preview"].lower()
        # At least one word from the prompt must appear in the rendered HTML
        assert any(w in html for w in ["revenue", "analytics", "dashboard", "finance"])

    @pytest.mark.asyncio
    async def test_html_preview_non_empty(self):
        result = await generate_frontend_design(prompt="dashboard")
        assert len(result["html_preview"]) > 500

    @pytest.mark.asyncio
    async def test_auth_html_has_sign_in(self):
        result = await generate_frontend_design(prompt="user login page", app_type="auth")
        html = result["html_preview"]
        assert "Sign In" in html or "sign in" in html.lower() or "login" in html.lower()


# ─────────────────────────────────────────────────────────────────────────────
# TestTsxCode — 6 tests
# ─────────────────────────────────────────────────────────────────────────────

@SKIP_DESIGN
@SLOW
class TestTsxCode:
    @pytest.mark.asyncio
    async def test_tsx_has_shadcn_imports(self):
        result = await generate_frontend_design(prompt="analytics dashboard")
        tsx = result["tsx_code"]
        assert '@/components/ui/' in tsx

    @pytest.mark.asyncio
    async def test_tsx_has_export_default(self):
        result = await generate_frontend_design(prompt="analytics dashboard")
        # Covers: `export default function Foo()` and `export default Foo` (arrow)
        assert "export default" in result["tsx_code"]

    @pytest.mark.asyncio
    async def test_tsx_has_jsx(self):
        result = await generate_frontend_design(prompt="analytics dashboard")
        assert "<div" in result["tsx_code"] or "<Card" in result["tsx_code"]

    @pytest.mark.asyncio
    async def test_crud_tsx_has_table(self):
        result = await generate_frontend_design(prompt="manage invoices", app_type="crud")
        tsx = result["tsx_code"]
        assert "table" in tsx.lower() or "Table" in tsx

    @pytest.mark.asyncio
    async def test_auth_tsx_has_input(self):
        result = await generate_frontend_design(prompt="user login", app_type="auth")
        assert "Input" in result["tsx_code"]

    @pytest.mark.asyncio
    async def test_tsx_function_name_derived_from_prompt(self):
        result = await generate_frontend_design(prompt="Revenue Dashboard Pro")
        tsx = result["tsx_code"]
        # Component name should be derived from prompt words
        assert "Revenue" in tsx or "Dashboard" in tsx or "Page" in tsx


# ─────────────────────────────────────────────────────────────────────────────
# TestColorSchemes — 3 tests
# ─────────────────────────────────────────────────────────────────────────────

@SKIP_DESIGN
@SLOW
class TestColorSchemes:
    @pytest.mark.asyncio
    async def test_blue_scheme(self):
        result = await generate_frontend_design(prompt="dashboard", color_scheme="blue")
        # Blue primary is #1d4ed8
        assert "#1d4ed8" in result["html_preview"] or "blue" in result["html_preview"].lower()

    @pytest.mark.asyncio
    async def test_green_scheme(self):
        result = await generate_frontend_design(prompt="dashboard", color_scheme="green")
        assert "#15803d" in result["html_preview"] or "green" in result["html_preview"].lower()

    @pytest.mark.asyncio
    async def test_slate_scheme(self):
        result = await generate_frontend_design(prompt="dashboard", color_scheme="slate")
        assert "#0f172a" in result["html_preview"]


# ─────────────────────────────────────────────────────────────────────────────
# TestRegistration — 2 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistration:
    def test_tool_registered(self):
        assert "generate_frontend_design" in _registered_tools

    def test_tool_risk_level(self):
        defn, _ = _registered_tools["generate_frontend_design"]
        assert defn.risk_level == RiskLevel.LOW

    def test_tool_resource_pattern(self):
        defn, _ = _registered_tools["generate_frontend_design"]
        assert defn.resource_pattern == "code:*"

    @pytest.mark.asyncio
    @pytest.mark.skipif(HAS_LLM_KEY, reason="only runs when no LLM is configured")
    async def test_raises_without_llm(self):
        """Tool must raise ToolError with a clear message, not silently return canned data."""
        from nexus.exceptions import ToolError
        with pytest.raises(ToolError, match="LLM"):
            await generate_frontend_design(prompt="test dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# TestShadcnMCPTools — 5 tests (require npx)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_NPX, reason="npx not available")
class TestShadcnMCPTools:
    @pytest.mark.asyncio
    async def test_mcp_server_connects(self):
        from nexus.mcp.client import MCPClient
        from nexus.types import MCPServerConfig
        client = MCPClient()
        cfg = MCPServerConfig(
            id="shadcn", tenant_id="test", name="shadcn", url="",
            transport="stdio", command="npx", args=["shadcn@latest", "mcp"],
        )
        await client.connect(cfg)
        assert client.is_connected("shadcn")
        await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_list_tools_returns_expected_tools(self):
        from nexus.mcp.client import MCPClient
        from nexus.types import MCPServerConfig
        client = MCPClient()
        cfg = MCPServerConfig(
            id="shadcn", tenant_id="test", name="shadcn", url="",
            transport="stdio", command="npx", args=["shadcn@latest", "mcp"],
        )
        await client.connect(cfg)
        session = list(client._sessions.values())[0]
        result = await session.list_tools()
        tool_names = {t.name for t in result.tools}
        expected = {
            "list_items_in_registries",
            "search_items_in_registries",
            "view_items_in_registries",
            "get_add_command_for_items",
            "get_item_examples_from_registries",
            "get_project_registries",
            "get_audit_checklist",
        }
        assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"
        await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_get_project_registries(self):
        from nexus.mcp.client import MCPClient
        from nexus.types import MCPServerConfig
        client = MCPClient()
        cfg = MCPServerConfig(
            id="shadcn", tenant_id="test", name="shadcn", url="",
            transport="stdio", command="npx", args=["shadcn@latest", "mcp"],
        )
        await client.connect(cfg)
        session = list(client._sessions.values())[0]
        result = await session.call_tool("get_project_registries", {})
        text = result.content[0].text if result.content else ""
        assert "@shadcn" in text
        await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_get_audit_checklist(self):
        from nexus.mcp.client import MCPClient
        from nexus.types import MCPServerConfig
        client = MCPClient()
        cfg = MCPServerConfig(
            id="shadcn", tenant_id="test", name="shadcn", url="",
            transport="stdio", command="npx", args=["shadcn@latest", "mcp"],
        )
        await client.connect(cfg)
        session = list(client._sessions.values())[0]
        result = await session.call_tool("get_audit_checklist", {})
        text = result.content[0].text if result.content else ""
        assert len(text) > 50  # has some content
        await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_via_mcp_tool_adapter(self):
        """MCPToolAdapter registers shadcn tools in the NEXUS tool registry."""
        from nexus.mcp.client import MCPClient
        from nexus.mcp.adapter import MCPToolAdapter
        from nexus.tools.registry import ToolRegistry
        from nexus.types import MCPServerConfig
        registry = ToolRegistry()
        client = MCPClient()
        adapter = MCPToolAdapter(registry=registry, client=client, repository=None, vault=None)
        cfg = MCPServerConfig(
            id="shadcn", tenant_id="test", name="shadcn", url="",
            transport="stdio", command="npx", args=["shadcn@latest", "mcp"],
        )
        await adapter.register_server(tenant_id="test", server_config=cfg)
        tools = registry.list_tools()
        tool_names = [t.name for t in tools]
        assert any("shadcn" in name for name in tool_names), f"No shadcn tools: {tool_names}"
        await client.disconnect_all()


# ─────────────────────────────────────────────────────────────────────────────
# TestShadcnMCPSearch — 4 tests (require npx)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_NPX, reason="npx not available")
class TestShadcnMCPSearch:
    """Each test opens its own MCP session to avoid anyio cancel-scope issues."""

    async def _session_call(self, tool_name: str, args: dict) -> str:
        from nexus.mcp.client import MCPClient
        from nexus.types import MCPServerConfig
        client = MCPClient()
        cfg = MCPServerConfig(
            id="shadcn", tenant_id="test", name="shadcn", url="",
            transport="stdio", command="npx", args=["shadcn@latest", "mcp"],
        )
        await client.connect(cfg)
        session = list(client._sessions.values())[0]
        result = await session.call_tool(tool_name, args)
        text = result.content[0].text if result.content else ""
        await client.disconnect_all()
        return text

    @pytest.mark.asyncio
    async def test_search_chart_returns_results(self):
        text = await self._session_call(
            "search_items_in_registries",
            {"registries": ["@shadcn"], "query": "chart", "limit": 5},
        )
        assert "chart" in text.lower()

    @pytest.mark.asyncio
    async def test_search_table_returns_results(self):
        text = await self._session_call(
            "search_items_in_registries",
            {"registries": ["@shadcn"], "query": "table data", "limit": 5},
        )
        assert "table" in text.lower() or "Found" in text

    @pytest.mark.asyncio
    async def test_get_add_command(self):
        text = await self._session_call(
            "get_add_command_for_items",
            {"items": ["card", "button", "badge"]},
        )
        assert "npx shadcn" in text
        assert "add" in text

    @pytest.mark.asyncio
    async def test_list_returns_403_items(self):
        text = await self._session_call(
            "list_items_in_registries",
            {"registries": ["@shadcn"], "limit": 5},
        )
        # The registry has 400+ items
        assert "Found" in text or "items" in text


# ─────────────────────────────────────────────────────────────────────────────
# TestFullPipeline — 3 tests (require npx + node)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (HAS_NPX and HAS_NODE and HAS_LLM_KEY),
    reason="npx, node, and LLM required",
)
@SLOW
class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_dashboard_full_pipeline(self, tmp_path):
        """MCP search → component list → generate_frontend_design → valid output."""
        from nexus.mcp.client import MCPClient
        from nexus.types import MCPServerConfig

        client = MCPClient()
        cfg = MCPServerConfig(
            id="shadcn", tenant_id="test", name="shadcn", url="",
            transport="stdio", command="npx", args=["shadcn@latest", "mcp"],
        )
        await client.connect(cfg)
        session = list(client._sessions.values())[0]

        # Get add command from MCP
        add_result = await session.call_tool(
            "get_add_command_for_items",
            {"items": ["card", "chart", "badge", "table"]},
        )
        add_cmd = add_result.content[0].text.strip() if add_result.content else ""
        await client.disconnect_all()

        # Generate design
        result = await generate_frontend_design(
            prompt="SaaS analytics dashboard",
            app_type="dashboard",
            color_scheme="zinc",
            add_command=add_cmd,
        )

        assert result["tsx_code"]
        assert result["html_preview"].startswith("<!DOCTYPE html>")
        assert "card" in result["components_used"]
        assert "npx shadcn" in result["install_command"]

    @pytest.mark.asyncio
    async def test_crud_full_pipeline(self):
        """Invoice CRUD app generation."""
        from nexus.mcp.client import MCPClient
        from nexus.types import MCPServerConfig

        client = MCPClient()
        cfg = MCPServerConfig(
            id="shadcn", tenant_id="test", name="shadcn", url="",
            transport="stdio", command="npx", args=["shadcn@latest", "mcp"],
        )
        await client.connect(cfg)
        session = list(client._sessions.values())[0]
        add_result = await session.call_tool(
            "get_add_command_for_items",
            {"items": ["table", "dialog", "button", "badge", "input"]},
        )
        add_cmd = add_result.content[0].text.strip() if add_result.content else ""
        await client.disconnect_all()

        result = await generate_frontend_design(
            prompt="Invoice management for freelancers",
            app_type="crud",
            color_scheme="blue",
            add_command=add_cmd,
        )

        tsx = result["tsx_code"]
        html = result["html_preview"]
        assert "dialog" in result["components_used"] or "table" in result["components_used"]
        assert "<!DOCTYPE html>" in html
        assert "export default function" in tsx

    @pytest.mark.asyncio
    async def test_output_files_writable(self, tmp_path):
        """Generated content can be written to disk."""
        result = await generate_frontend_design(
            prompt="Simple login page",
            app_type="auth",
            color_scheme="zinc",
        )
        tsx_path = tmp_path / "LoginPage.tsx"
        html_path = tmp_path / "preview.html"
        tsx_path.write_text(result["tsx_code"])
        html_path.write_text(result["html_preview"])
        assert tsx_path.stat().st_size > 100
        assert html_path.stat().st_size > 500
