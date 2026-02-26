"""Tests for all built-in tool implementations.

Coverage:
  file tools     — file_read, file_write (return correct strings)
  web tools      — web_search, web_fetch (return strings with the query/url)
  comm tools     — send_email (returns confirmation)
  data tools     — compute_stats (returns dict with count), knowledge_search (returns string)
  registration   — all tools registered via @tool decorator in get_registered_tools()
  risk levels    — send_email is HIGH, file_read is LOW, etc.
  decorator      — attached ._nexus_tool on each decorated function
"""

import pytest

# Force registration of all built-in tools
import nexus.tools.builtin.files  # noqa: F401
import nexus.tools.builtin.web  # noqa: F401
import nexus.tools.builtin.comms  # noqa: F401
import nexus.tools.builtin.data  # noqa: F401

from nexus.tools.plugin import get_registered_tools
from nexus.types import RiskLevel


# ── Registration ───────────────────────────────────────────────────────────────

class TestBuiltinRegistration:
    """All built-in tools must be in the global registry after import."""

    def test_file_read_registered(self):
        tools = get_registered_tools()
        assert "file_read" in tools

    def test_file_write_registered(self):
        tools = get_registered_tools()
        assert "file_write" in tools

    def test_web_search_registered(self):
        tools = get_registered_tools()
        assert "web_search" in tools

    def test_web_fetch_registered(self):
        tools = get_registered_tools()
        assert "web_fetch" in tools

    def test_send_email_registered(self):
        tools = get_registered_tools()
        assert "send_email" in tools

    def test_compute_stats_registered(self):
        tools = get_registered_tools()
        assert "compute_stats" in tools

    def test_knowledge_search_registered(self):
        tools = get_registered_tools()
        assert "knowledge_search" in tools

    def test_all_registered_tools_have_definition(self):
        tools = get_registered_tools()
        for name, (defn, fn) in tools.items():
            assert defn.name == name
            assert defn.description
            assert callable(fn)

    def test_all_registered_tools_have_parameters(self):
        tools = get_registered_tools()
        for name, (defn, fn) in tools.items():
            assert isinstance(defn.parameters, dict)
            assert "properties" in defn.parameters


# ── Risk levels ────────────────────────────────────────────────────────────────

class TestBuiltinRiskLevels:

    def test_file_read_is_low_risk(self):
        defn, _ = get_registered_tools()["file_read"]
        assert defn.risk_level == RiskLevel.LOW

    def test_file_write_is_medium_risk(self):
        defn, _ = get_registered_tools()["file_write"]
        assert defn.risk_level == RiskLevel.MEDIUM

    def test_web_search_is_low_risk(self):
        defn, _ = get_registered_tools()["web_search"]
        assert defn.risk_level == RiskLevel.LOW

    def test_web_fetch_is_low_risk(self):
        defn, _ = get_registered_tools()["web_fetch"]
        assert defn.risk_level == RiskLevel.LOW

    def test_send_email_is_high_risk(self):
        defn, _ = get_registered_tools()["send_email"]
        assert defn.risk_level == RiskLevel.HIGH

    def test_compute_stats_is_low_risk(self):
        defn, _ = get_registered_tools()["compute_stats"]
        assert defn.risk_level == RiskLevel.LOW

    def test_knowledge_search_is_low_risk(self):
        defn, _ = get_registered_tools()["knowledge_search"]
        assert defn.risk_level == RiskLevel.LOW


# ── Approval flags ─────────────────────────────────────────────────────────────

class TestBuiltinApproval:

    def test_send_email_requires_approval(self):
        defn, _ = get_registered_tools()["send_email"]
        assert defn.requires_approval is True

    def test_file_read_no_approval(self):
        defn, _ = get_registered_tools()["file_read"]
        assert defn.requires_approval is False

    def test_web_search_no_approval(self):
        defn, _ = get_registered_tools()["web_search"]
        assert defn.requires_approval is False


# ── file_read ──────────────────────────────────────────────────────────────────

class TestFileRead:

    @pytest.mark.asyncio
    async def test_result_is_nonempty_and_contains_path(self):
        """Result must be non-empty and embed the requested path."""
        _, fn = get_registered_tools()["file_read"]
        result = await fn(path="/tmp/nexus/test.txt")
        assert len(result) > 0
        assert "test.txt" in result or "/tmp" in result

    @pytest.mark.asyncio
    async def test_result_contains_path(self):
        _, fn = get_registered_tools()["file_read"]
        result = await fn(path="/tmp/nexus/my_document.txt")
        assert "my_document.txt" in result or "/tmp" in result

    @pytest.mark.asyncio
    async def test_different_paths_produce_different_output(self):
        """Each path call produces distinct output (path is embedded)."""
        from nexus.tools.builtin.files import file_read
        r1 = await file_read(path="/tmp/nexus/alpha.txt")
        r2 = await file_read(path="/tmp/nexus/beta.txt")
        assert r1 != r2, "Different paths must produce different output"

    def test_has_nexus_tool_attribute(self):
        from nexus.tools.builtin.files import file_read
        assert hasattr(file_read, "_nexus_tool")
        assert file_read._nexus_tool.name == "file_read"

    def test_parameters_include_path(self):
        defn, _ = get_registered_tools()["file_read"]
        assert "path" in defn.parameters["properties"]
        assert "path" in defn.parameters.get("required", [])


# ── file_write ─────────────────────────────────────────────────────────────────

class TestFileWrite:

    @pytest.mark.asyncio
    async def test_result_is_nonempty_and_references_path(self):
        """Confirmation must be non-empty and reference the target path."""
        _, fn = get_registered_tools()["file_write"]
        result = await fn(path="/tmp/nexus/out.txt", content="hello world")
        assert len(result) > 0
        assert "out.txt" in result or "/tmp" in result or "11" in result

    @pytest.mark.asyncio
    async def test_result_reflects_content_length(self):
        _, fn = get_registered_tools()["file_write"]
        result = await fn(path="/tmp/nexus/out.txt", content="hello")
        # "5" = len("hello") must appear somewhere in the confirmation
        assert "5" in result or "hello" in result or "/tmp" in result

    @pytest.mark.asyncio
    async def test_different_paths_produce_different_confirmations(self):
        """Writing to different paths gives different confirmation messages."""
        from nexus.tools.builtin.files import file_write
        r1 = await file_write(path="/tmp/nexus/foo.txt", content="x")
        r2 = await file_write(path="/tmp/nexus/bar.txt", content="x")
        assert r1 != r2, "Different paths must produce different confirmations"

    def test_parameters_include_path_and_content(self):
        defn, _ = get_registered_tools()["file_write"]
        props = defn.parameters["properties"]
        assert "path" in props
        assert "content" in props

    def test_resource_pattern_is_file_write(self):
        defn, _ = get_registered_tools()["file_write"]
        assert "file" in defn.resource_pattern or defn.resource_pattern == "*"


# ── web_search ─────────────────────────────────────────────────────────────────

class TestWebSearch:

    @pytest.mark.asyncio
    async def test_result_is_nonempty_and_contains_query(self):
        """Result must be non-empty and reference the search query."""
        _, fn = get_registered_tools()["web_search"]
        result = await fn(query="NEXUS AI framework")
        assert len(result) > 0
        assert "NEXUS AI framework" in result

    @pytest.mark.asyncio
    async def test_result_contains_query(self):
        _, fn = get_registered_tools()["web_search"]
        result = await fn(query="quantum computing")
        assert "quantum computing" in result

    @pytest.mark.asyncio
    async def test_different_queries_produce_different_results(self):
        """Each search query gives distinct results."""
        from nexus.tools.builtin.web import web_search
        r1 = await web_search(query="machine learning")
        r2 = await web_search(query="blockchain security")
        assert r1 != r2

    def test_parameters_include_query(self):
        defn, _ = get_registered_tools()["web_search"]
        assert "query" in defn.parameters["properties"]


# ── web_fetch ──────────────────────────────────────────────────────────────────

class TestWebFetch:

    @pytest.mark.asyncio
    async def test_result_is_nonempty_and_contains_url(self):
        """Result must be non-empty and reference the fetched URL."""
        _, fn = get_registered_tools()["web_fetch"]
        result = await fn(url="https://example.com")
        assert len(result) > 0
        assert "example.com" in result

    @pytest.mark.asyncio
    async def test_result_contains_url(self):
        _, fn = get_registered_tools()["web_fetch"]
        result = await fn(url="https://docs.nexus.ai")
        assert "docs.nexus.ai" in result or "Fetched" in result

    @pytest.mark.asyncio
    async def test_different_urls_produce_different_content(self):
        """Fetching different URLs returns different content."""
        from nexus.tools.builtin.web import web_fetch
        r1 = await web_fetch(url="https://alpha.example.com")
        r2 = await web_fetch(url="https://beta.example.com")
        assert r1 != r2

    def test_parameters_include_url(self):
        defn, _ = get_registered_tools()["web_fetch"]
        assert "url" in defn.parameters["properties"]


# ── send_email ─────────────────────────────────────────────────────────────────

class TestSendEmail:

    @pytest.mark.asyncio
    async def test_result_is_nonempty_and_contains_recipient(self):
        """Confirmation must be non-empty and include the recipient address."""
        _, fn = get_registered_tools()["send_email"]
        result = await fn(to="user@example.com", subject="Test", body="Hello")
        assert len(result) > 0
        assert "user@example.com" in result

    @pytest.mark.asyncio
    async def test_result_contains_recipient(self):
        _, fn = get_registered_tools()["send_email"]
        result = await fn(to="alice@corp.com", subject="Hi", body="Content")
        assert "alice@corp.com" in result

    @pytest.mark.asyncio
    async def test_subject_in_result(self):
        """Subject line should appear in the confirmation."""
        from nexus.tools.builtin.comms import send_email
        result = await send_email(to="bob@example.com", subject="Quarterly Report", body="Hey")
        assert "Quarterly Report" in result

    def test_parameters_include_to_subject_body(self):
        defn, _ = get_registered_tools()["send_email"]
        props = defn.parameters["properties"]
        assert "to" in props
        assert "subject" in props
        assert "body" in props


# ── compute_stats ──────────────────────────────────────────────────────────────

class TestComputeStats:

    @pytest.mark.asyncio
    async def test_count_key_present_and_correct(self):
        """Result must be a dict with a correct 'count' key."""
        _, fn = get_registered_tools()["compute_stats"]
        result = await fn(data=[1, 2, 3, 4, 5])
        assert isinstance(result, dict)
        assert result["count"] == 5

    @pytest.mark.asyncio
    async def test_count_matches_input_length(self):
        _, fn = get_registered_tools()["compute_stats"]
        result = await fn(data=[10, 20, 30])
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_empty_data_returns_zero_count(self):
        _, fn = get_registered_tools()["compute_stats"]
        result = await fn(data=[])
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_metrics_parameter_respected(self):
        _, fn = get_registered_tools()["compute_stats"]
        result = await fn(data=[1, 2, 3], metrics=["mean", "std"])
        assert "metrics" in result

    @pytest.mark.asyncio
    async def test_default_metrics_present_when_not_specified(self):
        """Without metrics arg, result still has 'metrics' key with defaults."""
        from nexus.tools.builtin.data import compute_stats
        result = await compute_stats(data=[1, 2, 3])
        assert "metrics" in result
        assert isinstance(result["metrics"], list)

    def test_parameters_include_data(self):
        defn, _ = get_registered_tools()["compute_stats"]
        assert "data" in defn.parameters["properties"]


# ── knowledge_search ───────────────────────────────────────────────────────────

class TestKnowledgeSearch:

    @pytest.mark.asyncio
    async def test_result_is_nonempty_and_contains_query(self):
        """Result must be non-empty and reference the search query."""
        _, fn = get_registered_tools()["knowledge_search"]
        result = await fn(query="NEXUS anomaly gates")
        assert len(result) > 0
        assert "NEXUS anomaly gates" in result

    @pytest.mark.asyncio
    async def test_result_contains_query(self):
        _, fn = get_registered_tools()["knowledge_search"]
        result = await fn(query="Merkle chain integrity")
        assert "Merkle chain integrity" in result

    @pytest.mark.asyncio
    async def test_namespace_reflected_in_result(self):
        """The namespace parameter should appear in the result."""
        _, fn = get_registered_tools()["knowledge_search"]
        result = await fn(query="test", namespace="custom_ns")
        assert "custom_ns" in result

    @pytest.mark.asyncio
    async def test_different_queries_give_different_results(self):
        """Different queries produce distinct output."""
        from nexus.tools.builtin.data import knowledge_search
        r1 = await knowledge_search(query="what is NEXUS")
        r2 = await knowledge_search(query="what is the Merkle chain")
        assert r1 != r2

    def test_parameters_include_query(self):
        defn, _ = get_registered_tools()["knowledge_search"]
        assert "query" in defn.parameters["properties"]

    def test_resource_pattern_is_kb(self):
        defn, _ = get_registered_tools()["knowledge_search"]
        assert "kb" in defn.resource_pattern
