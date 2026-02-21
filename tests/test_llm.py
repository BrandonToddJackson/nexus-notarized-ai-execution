"""Tests for nexus/llm/cost_tracker.py and nexus/llm/client.py."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from nexus.exceptions import BudgetExceeded, NexusError
from nexus.llm.client import LLMClient
from nexus.llm.cost_tracker import CostTracker
from nexus.types import CostRecord


# ── Helpers ──────────────────────────────────────────────────────────────────

KNOWN_MODEL = "gpt-4"          # litellm has pricing for this
UNKNOWN_MODEL = "mock/unknown"  # no pricing → falls back to 0.0
TENANT = "tenant-test-001"
CHAIN = "chain-abc"
SEAL = "seal-xyz"

USAGE_SMALL = {"input_tokens": 100, "output_tokens": 50}
USAGE_LARGE = {"input_tokens": 1_000, "output_tokens": 500}


# ── CostTracker ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestCostTrackerRecord:

    async def test_returns_cost_record(self):
        tracker = CostTracker()
        record = await tracker.record(TENANT, CHAIN, SEAL, KNOWN_MODEL, USAGE_SMALL)
        assert isinstance(record, CostRecord)

    async def test_record_fields_populated(self):
        tracker = CostTracker()
        record = await tracker.record(TENANT, CHAIN, SEAL, KNOWN_MODEL, USAGE_SMALL)
        assert record.tenant_id == TENANT
        assert record.chain_id == CHAIN
        assert record.seal_id == SEAL
        assert record.model == KNOWN_MODEL
        assert record.input_tokens == USAGE_SMALL["input_tokens"]
        assert record.output_tokens == USAGE_SMALL["output_tokens"]

    async def test_seal_id_optional_none(self):
        tracker = CostTracker()
        record = await tracker.record(TENANT, CHAIN, None, KNOWN_MODEL, USAGE_SMALL)
        assert record.seal_id is None

    async def test_known_model_yields_nonzero_cost(self):
        """litellm has pricing for gpt-4; cost must be > 0."""
        tracker = CostTracker()
        record = await tracker.record(TENANT, CHAIN, None, KNOWN_MODEL, USAGE_SMALL)
        assert record.cost_usd > 0.0

    async def test_unknown_model_falls_back_to_zero(self):
        """Unknown model raises inside litellm; we silently default to 0.0."""
        tracker = CostTracker()
        record = await tracker.record(TENANT, CHAIN, None, UNKNOWN_MODEL, USAGE_SMALL)
        assert record.cost_usd == 0.0

    async def test_litellm_exception_falls_back_to_zero(self):
        """If litellm raises for any reason, cost_usd == 0.0 (never propagates)."""
        tracker = CostTracker()
        with patch("nexus.llm.cost_tracker.litellm.completion_cost", side_effect=RuntimeError("boom")):
            record = await tracker.record(TENANT, CHAIN, None, KNOWN_MODEL, USAGE_SMALL)
        assert record.cost_usd == 0.0

    async def test_cumulative_accumulation(self):
        """Second call adds to in-memory cumulative, not reset."""
        tracker = CostTracker()
        r1 = await tracker.record(TENANT, CHAIN, None, KNOWN_MODEL, USAGE_SMALL)
        r2 = await tracker.record(TENANT, CHAIN, None, KNOWN_MODEL, USAGE_SMALL)
        cumulative = tracker._tenant_costs[TENANT]
        assert cumulative == pytest.approx(r1.cost_usd + r2.cost_usd)

    async def test_different_tenants_isolated(self):
        """Each tenant tracks its own cumulative cost."""
        tracker = CostTracker()
        await tracker.record("tenant-A", CHAIN, None, KNOWN_MODEL, USAGE_SMALL)
        await tracker.record("tenant-B", CHAIN, None, KNOWN_MODEL, USAGE_SMALL)
        assert "tenant-A" in tracker._tenant_costs
        assert "tenant-B" in tracker._tenant_costs
        # They should be equal since same usage, but they're independent
        assert tracker._tenant_costs["tenant-A"] == pytest.approx(
            tracker._tenant_costs["tenant-B"]
        )

    async def test_budget_warning_logged_at_80_pct(self, caplog):
        """Warning logged when cumulative >= 80% of budget."""
        tracker = CostTracker()
        # Manually pre-seed cumulative cost to just below warning threshold
        budget = 50.0
        alert_pct = 0.8
        # Inject 39.99 USD already spent; next call with ~0 cost should tip to warning
        tracker._tenant_costs[TENANT] = budget * alert_pct - 0.001

        # Use a mock cost that tips us just past 80%
        with patch("nexus.llm.cost_tracker.litellm.completion_cost", return_value=0.01):
            with caplog.at_level(logging.WARNING, logger="nexus.llm.cost_tracker"):
                record = await tracker.record(TENANT, CHAIN, None, KNOWN_MODEL, USAGE_SMALL)

        assert record.cost_usd == 0.01
        assert any("budget" in msg.lower() or "%" in msg for msg in caplog.messages), \
            f"Expected a budget warning, got: {caplog.messages}"

    async def test_budget_exceeded_raises(self):
        """BudgetExceeded raised when cumulative >= 100% of budget."""
        tracker = CostTracker()
        budget = 50.0
        # Pre-seed so next call pushes us over
        tracker._tenant_costs[TENANT] = budget - 0.001

        with patch("nexus.llm.cost_tracker.litellm.completion_cost", return_value=0.01):
            with pytest.raises(BudgetExceeded) as exc_info:
                await tracker.record(TENANT, CHAIN, None, KNOWN_MODEL, USAGE_SMALL)

        assert TENANT in str(exc_info.value)
        assert exc_info.value.details["tenant_id"] == TENANT
        assert exc_info.value.details["cumulative_usd"] >= budget

    async def test_budget_exceeded_cost_still_tracked(self):
        """Even when BudgetExceeded is raised, the cost is accumulated in memory."""
        tracker = CostTracker()
        budget = 50.0
        tracker._tenant_costs[TENANT] = budget - 0.001

        with patch("nexus.llm.cost_tracker.litellm.completion_cost", return_value=0.01):
            with pytest.raises(BudgetExceeded):
                await tracker.record(TENANT, CHAIN, None, KNOWN_MODEL, USAGE_SMALL)

        # Cost must be accumulated before raising
        assert tracker._tenant_costs[TENANT] >= budget

    async def test_repository_add_cost_called(self):
        """When repository provided, add_cost is awaited with the CostRecord."""
        mock_repo = MagicMock()
        mock_repo.add_cost = AsyncMock()

        tracker = CostTracker(repository=mock_repo)
        record = await tracker.record(TENANT, CHAIN, SEAL, KNOWN_MODEL, USAGE_SMALL)

        mock_repo.add_cost.assert_awaited_once_with(record)

    async def test_repository_failure_does_not_raise(self):
        """Repository errors are silently swallowed; record is still returned."""
        mock_repo = MagicMock()
        mock_repo.add_cost = AsyncMock(side_effect=RuntimeError("db down"))

        tracker = CostTracker(repository=mock_repo)
        # Must not raise despite repo failure
        record = await tracker.record(TENANT, CHAIN, None, KNOWN_MODEL, USAGE_SMALL)
        assert isinstance(record, CostRecord)

    async def test_no_repository_no_crash(self):
        """CostTracker with no repository uses pure in-memory tracking."""
        tracker = CostTracker(repository=None)
        record = await tracker.record(TENANT, CHAIN, None, KNOWN_MODEL, USAGE_SMALL)
        assert isinstance(record, CostRecord)
        assert TENANT in tracker._tenant_costs


# ── LLMClient ─────────────────────────────────────────────────────────────────

def _make_litellm_response(content="hello", tool_calls=None, prompt_tokens=10, completion_tokens=5):
    """Build a mock litellm response object matching the real structure."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@pytest.mark.asyncio
class TestLLMClientComplete:

    async def test_returns_expected_shape(self):
        """complete() returns dict with content, tool_calls, usage."""
        mock_response = _make_litellm_response("world")
        with patch("nexus.llm.client.litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            client = LLMClient(model="openai/gpt-4o")
            result = await client.complete([{"role": "user", "content": "hi"}])

        assert result["content"] == "world"
        assert result["tool_calls"] == []           # tool_calls=None → []
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    async def test_tool_calls_returned_when_present(self):
        tool_call_obj = MagicMock()
        mock_response = _make_litellm_response("", tool_calls=[tool_call_obj])
        with patch("nexus.llm.client.litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            client = LLMClient(model="openai/gpt-4o")
            result = await client.complete([{"role": "user", "content": "do something"}])

        assert result["tool_calls"] == [tool_call_obj]

    async def test_content_none_becomes_empty_string(self):
        """When the LLM returns None content (tool-call only response), we return ''."""
        mock_response = _make_litellm_response(content=None, tool_calls=[MagicMock()])
        with patch("nexus.llm.client.litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            client = LLMClient(model="openai/gpt-4o")
            result = await client.complete([{"role": "user", "content": "hi"}])

        assert result["content"] == ""

    async def test_temperature_zero_not_overridden_by_config(self):
        """temperature=0.0 is falsy; must not fall back to config default."""
        mock_response = _make_litellm_response()
        captured = {}

        async def capture(**kwargs):
            captured.update(kwargs)
            return mock_response

        with patch("nexus.llm.client.litellm.acompletion", new=capture):
            client = LLMClient(model="openai/gpt-4o")
            await client.complete([{"role": "user", "content": "hi"}], temperature=0.0)

        assert captured["temperature"] == 0.0

    async def test_temperature_none_uses_config(self):
        """temperature=None → use config.llm_temperature."""
        from nexus.config import config as nexus_config
        mock_response = _make_litellm_response()
        captured = {}

        async def capture(**kwargs):
            captured.update(kwargs)
            return mock_response

        with patch("nexus.llm.client.litellm.acompletion", new=capture):
            client = LLMClient(model="openai/gpt-4o")
            await client.complete([{"role": "user", "content": "hi"}])

        assert captured["temperature"] == nexus_config.llm_temperature

    async def test_max_tokens_none_uses_config(self):
        from nexus.config import config as nexus_config
        mock_response = _make_litellm_response()
        captured = {}

        async def capture(**kwargs):
            captured.update(kwargs)
            return mock_response

        with patch("nexus.llm.client.litellm.acompletion", new=capture):
            client = LLMClient(model="openai/gpt-4o")
            await client.complete([{"role": "user", "content": "hi"}])

        assert captured["max_tokens"] == nexus_config.llm_max_tokens

    async def test_tools_kwarg_only_added_when_provided(self):
        mock_response = _make_litellm_response()
        captured_no_tools = {}
        captured_with_tools = {}

        async def capture_no(**kwargs):
            captured_no_tools.update(kwargs)
            return mock_response

        async def capture_with(**kwargs):
            captured_with_tools.update(kwargs)
            return mock_response

        tool_def = [{"type": "function", "function": {"name": "search"}}]

        with patch("nexus.llm.client.litellm.acompletion", new=capture_no):
            client = LLMClient(model="openai/gpt-4o")
            await client.complete([{"role": "user", "content": "hi"}])

        assert "tools" not in captured_no_tools

        with patch("nexus.llm.client.litellm.acompletion", new=capture_with):
            await client.complete([{"role": "user", "content": "hi"}], tools=tool_def)

        assert captured_with_tools["tools"] == tool_def

    async def test_response_format_only_added_when_provided(self):
        mock_response = _make_litellm_response()
        captured = {}

        async def capture(**kwargs):
            captured.update(kwargs)
            return mock_response

        with patch("nexus.llm.client.litellm.acompletion", new=capture):
            client = LLMClient(model="openai/gpt-4o")
            await client.complete([{"role": "user", "content": "hi"}])

        assert "response_format" not in captured

        with patch("nexus.llm.client.litellm.acompletion", new=capture):
            await client.complete(
                [{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            )

        assert captured["response_format"] == {"type": "json_object"}

    async def test_provider_error_raises_nexus_error(self):
        """Any exception from litellm becomes NexusError."""
        with patch(
            "nexus.llm.client.litellm.acompletion",
            new=AsyncMock(side_effect=ValueError("api key missing")),
        ):
            client = LLMClient(model="openai/gpt-4o")
            with pytest.raises(NexusError) as exc_info:
                await client.complete([{"role": "user", "content": "hi"}])

        assert "LLM call failed" in str(exc_info.value)
        assert exc_info.value.details["model"] == "openai/gpt-4o"

    async def test_default_model_from_config(self):
        """No model arg → uses config.default_llm_model."""
        from nexus.config import config as nexus_config
        client = LLMClient()
        assert client.model == nexus_config.default_llm_model
