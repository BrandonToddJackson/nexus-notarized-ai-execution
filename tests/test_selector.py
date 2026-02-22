"""Tests for ToolSelector: rule-based and LLM-based tool selection.

Coverage:
  rule-based selection  — keyword matching picks best-fit tool, fallback to first available
  no tools available    — returns empty-name IntentDeclaration with confidence 0.0
  LLM selection         — parses LLM JSON response, falls back to rule-based on parse error
  resource target       — correct derivation from resource_pattern (with/without wildcard)
  parameter extraction  — primary param inferred from schema (query, path, url)
"""

import json
import pytest

from nexus.types import PersonaContract, RiskLevel, RetrievedContext, ToolDefinition
from nexus.tools.registry import ToolRegistry
from nexus.tools.selector import ToolSelector


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _context() -> RetrievedContext:
    return RetrievedContext(
        query="test",
        documents=[],
        confidence=0.5,
        sources=[],
        namespace="default",
    )


def _researcher() -> PersonaContract:
    return PersonaContract(
        name="researcher",
        description="Searches information",
        allowed_tools=["knowledge_search", "web_search"],
        resource_scopes=["kb:*", "web:*"],
        intent_patterns=["search for information"],
        risk_tolerance=RiskLevel.LOW,
        max_ttl_seconds=60,
    )


def _operator() -> PersonaContract:
    return PersonaContract(
        name="operator",
        description="Executes operations",
        allowed_tools=["file_read", "file_write", "compute_stats"],
        resource_scopes=["file:*", "data:*"],
        intent_patterns=["read file", "write file", "compute"],
        risk_tolerance=RiskLevel.HIGH,
        max_ttl_seconds=180,
    )


def _tool_def(name: str, description: str, resource_pattern: str = "*", param_name: str = "query") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {param_name: {"type": "string"}},
            "required": [param_name],
        },
        risk_level=RiskLevel.LOW,
        resource_pattern=resource_pattern,
    )


def _registry_with(*tool_specs) -> ToolRegistry:
    registry = ToolRegistry()
    for defn, fn in tool_specs:
        registry.register(defn, fn)
    return registry


async def _noop(**kwargs) -> str:
    return "ok"


# ── No tools for persona ───────────────────────────────────────────────────────

class TestSelectorNoTools:

    @pytest.mark.asyncio
    async def test_no_tools_for_persona_returns_empty_intent(self):
        """Persona with no matching tools → returns IntentDeclaration with empty tool_name."""
        registry = ToolRegistry()
        # Register a tool that researcher can't use
        registry.register(
            _tool_def("send_email", "Send emails", resource_pattern="email:*"),
            _noop,
        )
        selector = ToolSelector(registry, llm_client=None)
        researcher = _researcher()  # allowed: knowledge_search, web_search — neither registered
        intent = await selector.select("find some data", researcher, _context())
        assert intent.tool_name == ""
        assert intent.confidence == 0.0
        assert "No tools available" in intent.reasoning


# ── Rule-based selection ───────────────────────────────────────────────────────

class TestSelectorRuleBased:

    @pytest.mark.asyncio
    async def test_picks_best_keyword_match(self):
        """'search' in task → knowledge_search beats web_search on keyword score."""
        ks_defn = _tool_def("knowledge_search", "Search the knowledge base", resource_pattern="kb:*")
        ws_defn = _tool_def("web_search", "Search the web", resource_pattern="web:*")
        registry = _registry_with((ks_defn, _noop), (ws_defn, _noop))
        selector = ToolSelector(registry, llm_client=None)
        researcher = _researcher()
        intent = await selector.select("search knowledge base for NEXUS docs", researcher, _context())
        # "knowledge" appears in task → knowledge_search has higher keyword score
        assert intent.tool_name == "knowledge_search"

    @pytest.mark.asyncio
    async def test_falls_back_to_first_tool_when_no_keywords_match(self):
        """Task has no keywords matching any tool → falls back to first available tool."""
        ks_defn = _tool_def("knowledge_search", "Search the knowledge base", resource_pattern="kb:*")
        ws_defn = _tool_def("web_search", "Search the web", resource_pattern="web:*")
        registry = _registry_with((ks_defn, _noop), (ws_defn, _noop))
        selector = ToolSelector(registry, llm_client=None)
        researcher = _researcher()
        intent = await selector.select("do something completely unrelated to any tool", researcher, _context())
        # Must pick one (no crash)
        assert intent.tool_name in ("knowledge_search", "web_search")

    @pytest.mark.asyncio
    async def test_produces_valid_intent_declaration(self):
        ks_defn = _tool_def("knowledge_search", "Search the knowledge base", resource_pattern="kb:*")
        registry = _registry_with((ks_defn, _noop))
        selector = ToolSelector(registry, llm_client=None)
        researcher = _researcher()
        intent = await selector.select("search for AI research", researcher, _context())
        # All required fields are non-empty
        assert intent.task_description
        assert intent.planned_action
        assert intent.tool_name
        assert intent.reasoning
        assert 0.0 <= intent.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_query_param_inferred_from_task(self):
        """For a tool with 'query' param, the task text is used as the query value."""
        ks_defn = _tool_def("knowledge_search", "Search knowledge base", resource_pattern="kb:*")
        registry = _registry_with((ks_defn, _noop))
        selector = ToolSelector(registry, llm_client=None)
        researcher = _researcher()
        task = "search for NEXUS documentation"
        intent = await selector.select(task, researcher, _context())
        assert "query" in intent.tool_params
        assert intent.tool_params["query"] == task

    @pytest.mark.asyncio
    async def test_path_param_inferred_from_task(self):
        """For a tool with 'path' param, a slash-containing word is used."""
        file_defn = _tool_def("file_read", "Read a file", resource_pattern="file:read:*", param_name="path")
        registry = _registry_with((file_defn, _noop))
        operator = _operator()
        selector = ToolSelector(registry, llm_client=None)
        task = "read /etc/config.yml for settings"
        intent = await selector.select(task, operator, _context())
        assert "path" in intent.tool_params

    @pytest.mark.asyncio
    async def test_only_persona_allowed_tools_considered(self):
        """Even if send_email is registered, researcher can't use it."""
        email_defn = _tool_def("send_email", "Send emails", resource_pattern="email:*")
        ks_defn = _tool_def("knowledge_search", "Search knowledge", resource_pattern="kb:*")
        registry = _registry_with((email_defn, _noop), (ks_defn, _noop))
        selector = ToolSelector(registry, llm_client=None)
        researcher = _researcher()
        intent = await selector.select("search for docs", researcher, _context())
        assert intent.tool_name == "knowledge_search"
        assert intent.tool_name != "send_email"


# ── Resource target derivation ─────────────────────────────────────────────────

class TestResourceTargetDerivation:

    def _selector(self) -> ToolSelector:
        return ToolSelector(ToolRegistry(), llm_client=None)

    def test_pattern_without_wildcard_returns_literal(self):
        selector = self._selector()
        defn = _tool_def("my_tool", "test", resource_pattern="kb:product_docs")
        targets = selector._derive_resource_targets(defn, {"query": "test"})
        assert targets == ["kb:product_docs"]

    def test_pattern_with_wildcard_uses_param_as_slug(self):
        selector = self._selector()
        defn = _tool_def("my_tool", "test", resource_pattern="kb:*")
        targets = selector._derive_resource_targets(defn, {"query": "My Important Query"})
        assert len(targets) == 1
        assert targets[0].startswith("kb:")
        # Slug is lowercased, spaces replaced with underscores, truncated to 30 chars
        assert "my_important_query" in targets[0]

    def test_pattern_with_wildcard_no_params_returns_pattern(self):
        selector = self._selector()
        defn = _tool_def("my_tool", "test", resource_pattern="kb:*")
        targets = selector._derive_resource_targets(defn, {})
        assert targets == ["kb:*"]

    def test_email_pattern_derivation(self):
        selector = self._selector()
        defn = _tool_def("send_email", "email", resource_pattern="email:*", param_name="to")
        targets = selector._derive_resource_targets(defn, {"to": "user@corp.com"})
        assert len(targets) == 1
        assert targets[0].startswith("email:")

    def test_wildcard_only_pattern(self):
        selector = self._selector()
        defn = _tool_def("generic", "generic", resource_pattern="*")
        targets = selector._derive_resource_targets(defn, {"query": "test"})
        assert len(targets) == 1
        # prefix before * is "" so result is just the slug
        assert targets[0] == "test"


# ── LLM-based selection ────────────────────────────────────────────────────────

class MockLLMForSelector:
    """Stub LLM that returns a valid tool selection JSON."""

    def __init__(self, tool_name: str, params: dict, reasoning: str = "LLM reasoning"):
        self._response = json.dumps({
            "tool": tool_name,
            "params": params,
            "reasoning": reasoning,
        })
        self._error = False

    async def complete(self, messages, **kwargs) -> dict:
        if self._error:
            raise RuntimeError("LLM failed")
        return {"content": self._response, "tool_calls": [], "usage": {}}


class MockLLMBroken:
    """LLM that returns unparseable content."""
    async def complete(self, messages, **kwargs) -> dict:
        return {"content": "NOT JSON AT ALL", "tool_calls": [], "usage": {}}


class TestSelectorLLMBased:

    @pytest.mark.asyncio
    async def test_llm_selection_picks_tool_from_response(self):
        ks_defn = _tool_def("knowledge_search", "Search knowledge", resource_pattern="kb:*")
        ws_defn = _tool_def("web_search", "Search web", resource_pattern="web:*")
        registry = _registry_with((ks_defn, _noop), (ws_defn, _noop))
        llm = MockLLMForSelector("web_search", {"query": "AI news"}, "web is better for this")
        selector = ToolSelector(registry, llm_client=llm)
        researcher = _researcher()
        intent = await selector.select("find latest AI news", researcher, _context())
        assert intent.tool_name == "web_search"

    @pytest.mark.asyncio
    async def test_llm_selection_uses_returned_params(self):
        ks_defn = _tool_def("knowledge_search", "Search knowledge", resource_pattern="kb:*")
        registry = _registry_with((ks_defn, _noop))
        llm = MockLLMForSelector("knowledge_search", {"query": "specific search query from LLM"})
        selector = ToolSelector(registry, llm_client=llm)
        researcher = _researcher()
        intent = await selector.select("task", researcher, _context())
        assert intent.tool_params.get("query") == "specific search query from LLM"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_rule_based(self):
        """If LLM returns invalid JSON → fall back to rule-based selection."""
        ks_defn = _tool_def("knowledge_search", "Search knowledge", resource_pattern="kb:*")
        registry = _registry_with((ks_defn, _noop))
        llm = MockLLMBroken()
        selector = ToolSelector(registry, llm_client=llm)
        researcher = _researcher()
        # Must not raise — falls back to rule-based
        intent = await selector.select("search for NEXUS", researcher, _context())
        assert intent.tool_name != ""

    @pytest.mark.asyncio
    async def test_llm_uses_tool_name_from_response(self):
        """Selector trusts the LLM's tool_name — Gate 1 (scope) is what blocks misuse.

        If LLM picks a tool not in the persona's allowed list, the selector still
        returns that tool_name (it is Gate 1 in AnomalyEngine that blocks execution).
        The tool_defn fallback only applies to resource target derivation.
        """
        ks_defn = _tool_def("knowledge_search", "Search knowledge", resource_pattern="kb:*")
        registry = _registry_with((ks_defn, _noop))
        llm = MockLLMForSelector("send_email", {"to": "user@example.com"})
        selector = ToolSelector(registry, llm_client=llm)
        researcher = _researcher()
        intent = await selector.select("notify user", researcher, _context())
        # Selector returns what LLM said — Gate 1 will BLOCK this at execution time
        assert intent.tool_name == "send_email"

    @pytest.mark.asyncio
    async def test_llm_confidence_set_to_point_eight(self):
        ks_defn = _tool_def("knowledge_search", "Search knowledge", resource_pattern="kb:*")
        registry = _registry_with((ks_defn, _noop))
        llm = MockLLMForSelector("knowledge_search", {"query": "test"})
        selector = ToolSelector(registry, llm_client=llm)
        researcher = _researcher()
        intent = await selector.select("search", researcher, _context())
        assert intent.confidence == 0.8

    @pytest.mark.asyncio
    async def test_llm_reasoning_preserved(self):
        ks_defn = _tool_def("knowledge_search", "Search knowledge", resource_pattern="kb:*")
        registry = _registry_with((ks_defn, _noop))
        llm = MockLLMForSelector("knowledge_search", {"query": "test"}, reasoning="Best tool for this")
        selector = ToolSelector(registry, llm_client=llm)
        researcher = _researcher()
        intent = await selector.select("search", researcher, _context())
        assert intent.reasoning == "Best tool for this"

    @pytest.mark.asyncio
    async def test_markdown_fenced_json_parsed_correctly(self):
        """LLM sometimes wraps JSON in ```json fences — must be stripped."""
        ks_defn = _tool_def("knowledge_search", "Search knowledge", resource_pattern="kb:*")
        registry = _registry_with((ks_defn, _noop))

        class FencedLLM:
            async def complete(self, messages, **kwargs):
                return {
                    "content": '```json\n{"tool": "knowledge_search", "params": {"query": "test"}, "reasoning": "fenced"}\n```',
                    "tool_calls": [],
                    "usage": {},
                }

        selector = ToolSelector(registry, llm_client=FencedLLM())
        researcher = _researcher()
        intent = await selector.select("search", researcher, _context())
        assert intent.tool_name == "knowledge_search"
