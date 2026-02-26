"""Integration tests for SalesGrowthOrchestrator.

Uses a lightweight mock engine that returns ChainPlan-like objects so no
real AI calls or databases are needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from nexus.core.orchestrators.sales_growth import (
    SalesGrowthOrchestrator,
    _extract_list_from_chain,
    _extract_str_from_chain,
    _extract_dict_from_chain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chain(output=None):
    """Create a minimal mock ChainPlan with one seal."""
    seal = MagicMock()
    seal.output = output
    chain = MagicMock()
    chain.seals = [seal]
    return chain


def make_engine(output=None):
    """Create a mock NexusEngine whose run() returns a chain with given output."""
    engine = MagicMock()
    engine.run = AsyncMock(return_value=make_chain(output=output))
    return engine


DEFAULT_CONFIG = {
    "campaign_ids": ["campaign-abc"],
    "sheets_id": "sheet-xyz",
    "retell_from_number": "+15550000001",
    "interval_hours": 6,
    "tenant_id": "test-tenant",
}

# Stub lead: has linkedin_url so the linkedin step fires, but no phone_number
# so the voice step is skipped without any HTTP calls.
STUB_LEAD = {
    "email": "stub@example.com",
    "first_name": "Stub",
    "company": "StubCo",
    "linkedin_url": "https://linkedin.com/in/stub",
    "status": "Interested",
}


@pytest.fixture(autouse=True)
def mock_instantly_fallback(monkeypatch):
    """Prevent real HTTP calls: patch the direct-call fallback in _step_poll_leads."""
    async def stub_get_warm_leads(*args, **kwargs):
        return [STUB_LEAD]

    monkeypatch.setattr(
        "nexus.tools.builtin.sales_growth.instantly_get_warm_leads",
        stub_get_warm_leads,
    )


# ---------------------------------------------------------------------------
# Chain extraction helpers
# ---------------------------------------------------------------------------

class TestExtractHelpers:
    def test_extract_list_from_chain_direct_list(self):
        chain = make_chain(output=[{"email": "a@b.com"}])
        result = _extract_list_from_chain(chain, default=[])
        assert result == [{"email": "a@b.com"}]

    def test_extract_list_from_chain_json_string(self):
        chain = make_chain(output='[{"email": "a@b.com"}]')
        result = _extract_list_from_chain(chain, default=[])
        assert result == [{"email": "a@b.com"}]

    def test_extract_list_from_chain_empty_seals(self):
        chain = MagicMock()
        chain.seals = []
        result = _extract_list_from_chain(chain, default=["fallback"])
        assert result == ["fallback"]

    def test_extract_list_from_chain_bad_output(self):
        chain = make_chain(output="not a list")
        result = _extract_list_from_chain(chain, default=[])
        assert result == []

    def test_extract_str_from_chain(self):
        chain = make_chain(output="+15551234567")
        result = _extract_str_from_chain(chain, default="")
        assert result == "+15551234567"

    def test_extract_str_from_chain_no_seals(self):
        chain = MagicMock()
        chain.seals = []
        result = _extract_str_from_chain(chain, default="fallback")
        assert result == "fallback"

    def test_extract_dict_from_chain_direct_dict(self):
        chain = make_chain(output={"batch_call_id": "batch-001"})
        result = _extract_dict_from_chain(chain, default={})
        assert result == {"batch_call_id": "batch-001"}

    def test_extract_dict_from_chain_json_string(self):
        chain = make_chain(output='{"batch_call_id": "batch-001"}')
        result = _extract_dict_from_chain(chain, default={})
        assert result == {"batch_call_id": "batch-001"}

    def test_extract_dict_from_chain_fallback(self):
        chain = make_chain(output=None)
        result = _extract_dict_from_chain(chain, default={"default": True})
        assert result == {"default": True}


# ---------------------------------------------------------------------------
# Orchestrator construction
# ---------------------------------------------------------------------------

class TestOrchestratorConstruction:
    def test_init_sets_config(self):
        engine = make_engine()
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        assert orch.campaign_ids == ["campaign-abc"]
        assert orch.sheets_id == "sheet-xyz"
        assert orch.retell_from_number == "+15550000001"
        assert orch.interval_hours == 6
        assert orch.tenant_id == "test-tenant"

    def test_default_interval_is_6(self):
        engine = make_engine()
        config = {**DEFAULT_CONFIG}
        del config["interval_hours"]
        orch = SalesGrowthOrchestrator(engine, config)
        assert orch.interval_hours == 6

    def test_default_tenant_id(self):
        engine = make_engine()
        config = {**DEFAULT_CONFIG}
        del config["tenant_id"]
        orch = SalesGrowthOrchestrator(engine, config)
        assert orch.tenant_id == "cli-user"


# ---------------------------------------------------------------------------
# run_cycle — end-to-end with mocked engine
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_li_sleep(monkeypatch):
    """Prevent real asyncio.sleep calls from the LinkedIn inter-lead cooldown in cycle tests."""
    monkeypatch.setattr("nexus.core.orchestrators.sales_growth.asyncio.sleep", AsyncMock())
    monkeypatch.setattr("nexus.tools.builtin.sales_growth.asyncio.sleep", AsyncMock())


class TestRunCycle:
    @pytest.mark.asyncio
    async def test_run_cycle_returns_summary(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        result = await orch.run_cycle()
        assert "leads_count" in result
        assert "linkedin_result" in result
        assert "retell_result" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_run_cycle_calls_engine_multiple_times(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        await orch.run_cycle()
        # poll(1) + enrich×1 lead(1) + linkedin×1 lead(1) = 3
        # voice + sheets call tools directly (no LLM needed for deterministic ops)
        assert engine.run.call_count >= 3

    @pytest.mark.asyncio
    async def test_run_cycle_uses_sales_growth_agent_persona(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        await orch.run_cycle()
        for call in engine.run.call_args_list:
            args, kwargs = call
            persona = args[2] if len(args) > 2 else kwargs.get("persona_name")
            assert persona == "sales_growth_agent", f"Unexpected persona: {persona}"

    @pytest.mark.asyncio
    async def test_run_cycle_tenant_id_passed(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        await orch.run_cycle()
        for call in engine.run.call_args_list:
            args, _ = call
            assert args[1] == "test-tenant"

    @pytest.mark.asyncio
    async def test_run_cycle_no_leads_with_phones_skips_retell(self):
        """When no phone numbers are enriched, retell result should be skipped."""
        engine = make_engine(output=None)

        async def stub_enrich(email, company):
            return ""

        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        with patch("nexus.tools.builtin.sales_growth.enrich_lead_phone", side_effect=stub_enrich):
            result = await orch.run_cycle()
        assert result["retell_result"].get("status") == "skipped"


# ---------------------------------------------------------------------------
# Step isolation tests
# ---------------------------------------------------------------------------

class TestStepPollLeads:
    @pytest.mark.asyncio
    async def test_poll_leads_falls_back_to_stub(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        leads = await orch._step_poll_leads()
        assert isinstance(leads, list)
        assert len(leads) >= 1

    @pytest.mark.asyncio
    async def test_poll_leads_uses_chain_output_if_available(self):
        chain_leads = [{"email": "x@y.com", "first_name": "X", "company": "Y", "status": "Interested"}]
        engine = make_engine(output=chain_leads)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        leads = await orch._step_poll_leads()
        assert leads == chain_leads


class TestStepEnrichPhones:
    @pytest.mark.asyncio
    async def test_enrich_phones_adds_phone_number_field(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        leads = [{"email": "alice@example.com", "company": "Acme", "first_name": "Alice"}]
        enriched = await orch._step_enrich_phones(leads)
        assert all("phone_number" in lead for lead in enriched)

    @pytest.mark.asyncio
    async def test_enrich_phones_preserves_all_original_fields(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        leads = [{"email": "alice@example.com", "company": "Acme", "first_name": "Alice", "custom": "data"}]
        enriched = await orch._step_enrich_phones(leads)
        assert enriched[0]["custom"] == "data"


class TestStepLinkedInOutreach:
    @pytest.mark.asyncio
    async def test_linkedin_outreach_tracks_sent_count(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        leads = [
            {"email": "a@b.com", "first_name": "Alice", "linkedin_url": "https://linkedin.com/in/alice", "status": "Interested"},
            {"email": "b@c.com", "first_name": "Bob", "linkedin_url": "https://linkedin.com/in/bob", "status": "Interested"},
        ]
        result = await orch._step_linkedin_outreach(leads)
        assert result["sent"] == 2
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_linkedin_outreach_skips_leads_without_url(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        leads = [{"email": "a@b.com", "first_name": "Alice", "linkedin_url": "", "status": "Interested"}]
        result = await orch._step_linkedin_outreach(leads)
        assert result["failed"] == 1
        assert result["sent"] == 0


class TestStepVoiceCalls:
    @pytest.mark.asyncio
    async def test_voice_calls_skipped_when_no_phones(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        result = await orch._step_voice_calls([{"email": "a@b.com"}])
        assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_voice_calls_scheduled_for_leads_with_phones(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        leads = [{"email": "a@b.com", "phone_number": "+15551234567", "first_name": "Alice"}]
        result = await orch._step_voice_calls(leads)
        assert "batch_call_id" in result


class TestStepLogToSheets:
    @pytest.mark.asyncio
    async def test_log_to_sheets_does_not_raise(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        await orch._step_log_to_sheets(
            leads=[{"email": "a@b.com", "phone_number": "+15551234567"}],
            linkedin_result={"sent": 1, "failed": 0},
            retell_result={"batch_call_id": "batch-001", "scheduled_count": 1},
            timestamp="2026-02-25T00:00:00Z",
        )

    @pytest.mark.asyncio
    async def test_log_to_sheets_falls_back_on_engine_error(self):
        engine = MagicMock()
        engine.run = AsyncMock(side_effect=Exception("engine unavailable"))
        orch = SalesGrowthOrchestrator(engine, DEFAULT_CONFIG)
        # Should not raise — falls back to direct tool call
        await orch._step_log_to_sheets(
            leads=[],
            linkedin_result={},
            retell_result={},
            timestamp="2026-02-25T00:00:00Z",
        )


# ---------------------------------------------------------------------------
# run_forever — verifies it loops
# ---------------------------------------------------------------------------

class TestRunForever:
    @pytest.mark.asyncio
    async def test_run_forever_calls_run_cycle_repeatedly(self):
        engine = make_engine(output=None)
        orch = SalesGrowthOrchestrator(engine, {**DEFAULT_CONFIG, "interval_hours": 0})

        call_count = 0
        original_run_cycle = orch.run_cycle

        async def counting_run_cycle():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise KeyboardInterrupt
            return await original_run_cycle()

        orch.run_cycle = counting_run_cycle

        with pytest.raises(KeyboardInterrupt):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await orch.run_forever()

        assert call_count >= 2
