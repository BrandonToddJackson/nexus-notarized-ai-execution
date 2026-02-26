"""Unit tests for sales_growth built-in tools.

All external API calls are mocked so no network access is required.
"""

import pytest
import respx
import httpx
from nexus.tools.builtin.sales_growth import (
    instantly_get_warm_leads,
    instantly_get_campaign_analytics,
    instantly_mark_lead_as_contacted,
    enrich_lead_phone,
    linkedin_send_connection_request,
    linkedin_send_dm,
    linkedin_get_profile,
    craft_linkedin_message,
    retell_create_batch_call,
    retell_get_call_status,
    retell_get_batch_status,
    sheets_append_row,
    sheets_log_cycle,
    sheets_get_sheet_id,
    sheets_create_sheet,
)
from nexus.tools.plugin import _registered_tools
from nexus.types import RiskLevel


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_all_tools_registered(self):
        expected = [
            "instantly_get_warm_leads", "instantly_get_campaign_analytics", "instantly_mark_lead_as_contacted",
            "enrich_lead_phone",
            "linkedin_send_connection_request", "linkedin_send_dm", "linkedin_get_profile", "craft_linkedin_message",
            "retell_create_batch_call", "retell_get_call_status", "retell_get_batch_status",
            "sheets_append_row", "sheets_log_cycle", "sheets_get_sheet_id", "sheets_create_sheet",
        ]
        for name in expected:
            assert name in _registered_tools, f"Tool '{name}' not found in registry"

    def test_hunter_clearbit_are_private_not_registered(self):
        """hunter_find_phone and clearbit_find_phone are internal helpers, not tools."""
        assert "hunter_find_phone" not in _registered_tools
        assert "clearbit_find_phone" not in _registered_tools

    def test_risk_levels(self):
        high_tools = ["linkedin_send_dm", "retell_create_batch_call"]
        medium_tools = ["instantly_get_warm_leads", "instantly_mark_lead_as_contacted"]
        low_tools = [
            "instantly_get_campaign_analytics", "enrich_lead_phone",
            "linkedin_get_profile",
            "retell_get_call_status", "retell_get_batch_status",
            "sheets_append_row", "sheets_log_cycle", "sheets_get_sheet_id", "sheets_create_sheet",
        ]
        for name in high_tools:
            definition, _ = _registered_tools[name]
            assert definition.risk_level == RiskLevel.HIGH, f"{name} should be HIGH"
        for name in medium_tools:
            definition, _ = _registered_tools[name]
            assert definition.risk_level == RiskLevel.MEDIUM, f"{name} should be MEDIUM"
        for name in low_tools:
            definition, _ = _registered_tools[name]
            assert definition.risk_level == RiskLevel.LOW, f"{name} should be LOW"

    def test_resource_patterns(self):
        patterns = {
            "instantly_get_warm_leads": "crm:leads:*",
            "enrich_lead_phone": "enrichment:phone:*",
            "linkedin_send_dm": "social:linkedin:*",
            "retell_create_batch_call": "voice:calls:*",
            "sheets_append_row": "data:sheets:*",
        }
        for tool_name, expected_pattern in patterns.items():
            definition, _ = _registered_tools[tool_name]
            assert definition.resource_pattern == expected_pattern


# ---------------------------------------------------------------------------
# Instantly tools
# ---------------------------------------------------------------------------

_INSTANTLY_BASE = "https://api.instantly.ai/api/v2"

_FAKE_LEAD = {
    "email": "alice@example.com",
    "first_name": "Alice",
    "last_name": "Smith",
    "company": "Acme Corp",
    "linkedin_url": "https://linkedin.com/in/alice",
    "status": "Interested",
}


class TestInstantlyTools:
    @pytest.mark.asyncio
    @respx.mock
    async def test_get_warm_leads_returns_list(self, monkeypatch):
        monkeypatch.setenv("INSTANTLY_API_KEY", "fake-key")
        respx.post(f"{_INSTANTLY_BASE}/leads/list").mock(
            return_value=httpx.Response(200, json={"items": [_FAKE_LEAD]})
        )
        leads = await instantly_get_warm_leads(["campaign-123"])
        assert isinstance(leads, list)
        assert len(leads) >= 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_warm_leads_lead_has_required_fields(self, monkeypatch):
        monkeypatch.setenv("INSTANTLY_API_KEY", "fake-key")
        respx.post(f"{_INSTANTLY_BASE}/leads/list").mock(
            return_value=httpx.Response(200, json={"items": [_FAKE_LEAD]})
        )
        leads = await instantly_get_warm_leads(["campaign-abc"])
        lead = leads[0]
        assert "email" in lead
        assert "first_name" in lead
        assert "company" in lead

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_warm_leads_uses_campaign_id(self, monkeypatch):
        monkeypatch.setenv("INSTANTLY_API_KEY", "fake-key")
        respx.post(f"{_INSTANTLY_BASE}/leads/list").mock(
            return_value=httpx.Response(200, json={"items": [_FAKE_LEAD]})
        )
        leads = await instantly_get_warm_leads(["my-campaign"])
        assert any(lead.get("campaign_id") == "my-campaign" for lead in leads)

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_warm_leads_filters_already_contacted(self, monkeypatch):
        monkeypatch.setenv("INSTANTLY_API_KEY", "fake-key")
        contacted_lead = {**_FAKE_LEAD, "custom_variables": {"nexus_contacted": "true"}}
        respx.post(f"{_INSTANTLY_BASE}/leads/list").mock(
            return_value=httpx.Response(200, json={"items": [_FAKE_LEAD, contacted_lead]})
        )
        leads = await instantly_get_warm_leads(["campaign-abc"], filter_contacted=True)
        assert len(leads) == 1
        assert leads[0]["email"] == "alice@example.com"

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_campaign_analytics_returns_dict(self, monkeypatch):
        monkeypatch.setenv("INSTANTLY_API_KEY", "fake-key")
        respx.get(f"{_INSTANTLY_BASE}/campaigns/analytics").mock(
            return_value=httpx.Response(200, json={
                "id": "campaign-123",
                "leads_count": 10,
                "contacted_count": 5,
                "emails_sent_count": 5,
                "open_count": 2,
                "reply_count": 1,
                "bounced_count": 0,
                "unsubscribed_count": 0,
                "completed_count": 0,
            })
        )
        result = await instantly_get_campaign_analytics("campaign-123")
        assert isinstance(result, dict)
        assert result["campaign_id"] == "campaign-123"
        assert "open_rate" in result
        assert "reply_rate" in result

    @pytest.mark.asyncio
    @respx.mock
    async def test_mark_lead_as_contacted(self, monkeypatch):
        monkeypatch.setenv("INSTANTLY_API_KEY", "fake-key")
        respx.patch(f"{_INSTANTLY_BASE}/leads/test%40example.com").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        result = await instantly_mark_lead_as_contacted("test@example.com", "campaign-123")
        assert result["lead_email"] == "test@example.com"
        assert result["status"] == "updated"


# ---------------------------------------------------------------------------
# Phone enrichment
# ---------------------------------------------------------------------------

class TestPhoneEnrichmentTools:
    @pytest.mark.asyncio
    async def test_enrich_lead_phone_returns_string_no_keys(self):
        """Without API keys configured, returns empty string (not an error)."""
        phone = await enrich_lead_phone("alice@example.com", "Acme Corp")
        assert isinstance(phone, str)

    @pytest.mark.asyncio
    @respx.mock
    async def test_enrich_lead_phone_uses_hunter_when_key_set(self, monkeypatch):
        monkeypatch.setenv("HUNTER_API_KEY", "fake-hunter-key")
        respx.get("https://api.hunter.io/v2/phone-finder").mock(
            return_value=httpx.Response(200, json={"data": {"phone_number": "+15551234567"}})
        )
        phone = await enrich_lead_phone("alice@example.com", "Acme Corp")
        assert phone == "+15551234567"

    @pytest.mark.asyncio
    @respx.mock
    async def test_enrich_lead_phone_falls_back_to_clearbit(self, monkeypatch):
        monkeypatch.setenv("HUNTER_API_KEY", "fake-key")
        monkeypatch.setenv("CLEARBIT_API_KEY", "fake-clearbit-key")
        # Hunter returns no phone
        respx.get("https://api.hunter.io/v2/phone-finder").mock(
            return_value=httpx.Response(200, json={"data": {"phone_number": ""}})
        )
        # Clearbit returns a phone
        respx.get("https://person.clearbit.com/v2/combined/find").mock(
            return_value=httpx.Response(200, json={"person": {"employment": {"phone": "+15559876543"}}})
        )
        phone = await enrich_lead_phone("alice@example.com", "Acme Corp")
        assert phone == "+15559876543"


# ---------------------------------------------------------------------------
# LinkedIn tools
# ---------------------------------------------------------------------------

class TestLinkedInTools:
    @pytest.mark.asyncio
    async def test_linkedin_send_dm_no_credentials(self):
        """Without Unipile credentials, returns a stub sent response."""
        result = await linkedin_send_dm("https://linkedin.com/in/alice", "Hello Alice!")
        assert result["status"] == "sent"
        assert "conversation_id" in result
        assert result.get("_stub") is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_linkedin_send_dm_with_credentials(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LI_AT", "fake-li-at-token")
        monkeypatch.setenv("LI_CSRF", "fake-csrf")
        # Redirect state file to tmp so daily cap starts at 0 regardless of real state
        import nexus.tools.builtin.sales_growth as sg
        monkeypatch.setattr(sg, "_LI_STATE_FILE", tmp_path / "li_state.json")
        # Mock profile lookup via 2026 dash endpoint (returns fsd_profile entityUrn in included[0])
        respx.get("https://www.linkedin.com/voyager/api/identity/dash/profiles").mock(
            return_value=httpx.Response(200, json={"included": [{"entityUrn": "urn:li:fsd_profile:ACoAAAtest", "objectUrn": "urn:li:member:99999"}]})
        )
        # Mock conversation creation (returns conversation ID in header)
        respx.post("https://www.linkedin.com/voyager/api/messaging/conversations").mock(
            return_value=httpx.Response(201, headers={"x-restli-id": "conv-456"}, json={})
        )
        result = await linkedin_send_dm("https://linkedin.com/in/alice", "Hello!")
        assert result["status"] == "sent"
        assert result["conversation_id"] == "conv-456"
        assert "_stub" not in result

    @pytest.mark.asyncio
    async def test_linkedin_get_profile_no_credentials(self):
        result = await linkedin_get_profile("https://linkedin.com/in/alice-smith")
        assert isinstance(result, dict)
        assert "first_name" in result
        assert result.get("_stub") is True

    @pytest.mark.asyncio
    async def test_craft_linkedin_message_interested(self):
        """LLM unavailable in tests — falls back to template which must contain name."""
        from unittest.mock import AsyncMock, patch
        mock_llm = AsyncMock(return_value={"content": "Hi Alice, saw your work in SaaS — would swap notes anytime."})
        with patch("nexus.llm.client.LLMClient.complete", mock_llm):
            msg = await craft_linkedin_message("Alice", "Interested", industry="SaaS")
        assert "Alice" in msg
        assert len(msg) > 20
        assert len(msg) <= 300

    @pytest.mark.asyncio
    async def test_craft_linkedin_message_uses_fallback_on_llm_error(self):
        """When LLM raises, fallback template is returned — never hard-fails."""
        from unittest.mock import AsyncMock, patch
        with patch("nexus.llm.client.LLMClient.complete", AsyncMock(side_effect=Exception("LLM down"))):
            msg = await craft_linkedin_message("Bob", "Replied", headline="VP of Sales at Acme")
        assert "Bob" in msg
        assert len(msg) <= 300

    @pytest.mark.asyncio
    async def test_craft_linkedin_message_enforces_300_char_limit(self):
        """Message must never exceed LinkedIn's 300-char connection note limit."""
        from unittest.mock import AsyncMock, patch
        long_response = "x" * 500
        with patch("nexus.llm.client.LLMClient.complete", AsyncMock(return_value={"content": long_response})):
            msg = await craft_linkedin_message("Charlie", "Unknown")
        assert len(msg) <= 300

    @pytest.mark.asyncio
    async def test_craft_linkedin_message_uses_sender_context(self):
        """Prompt should include sender context so LLM writes peer-level outreach."""
        from unittest.mock import AsyncMock, patch, call
        mock_llm = AsyncMock(return_value={"content": "Hi Dana, working on similar automation challenges — would be good to swap notes."})
        with patch("nexus.llm.client.LLMClient.complete", mock_llm) as patched:
            msg = await craft_linkedin_message(
                "Dana", "Interested",
                summary="I focus on reducing ops overhead through AI",
                sender_context="founder building AI workflow automation for B2B sales teams",
            )
        # Sender context must appear in the user prompt (messages[1] — index 0 is system msg)
        prompt_sent = patched.call_args[1]["messages"][1]["content"]
        assert "founder building AI workflow automation" in prompt_sent
        assert "Dana" in msg


# ---------------------------------------------------------------------------
# Retell voice tools
# ---------------------------------------------------------------------------

class TestRetellTools:
    @pytest.mark.asyncio
    async def test_create_batch_call_no_credentials(self):
        """Without RETELL_API_KEY, returns skipped response with correct count."""
        leads = [{"phone_number": "+15551234567", "first_name": "Alice"}]
        result = await retell_create_batch_call(leads, "+15550000001")
        assert "batch_call_id" in result
        assert result["scheduled_count"] == 1

    @pytest.mark.asyncio
    async def test_create_batch_call_scheduled_count_matches_leads(self):
        leads = [
            {"phone_number": "+15551111111", "first_name": "Alice"},
            {"phone_number": "+15552222222", "first_name": "Bob"},
            {"phone_number": "+15553333333", "first_name": "Charlie"},
        ]
        result = await retell_create_batch_call(leads, "+15550000001")
        assert result["scheduled_count"] == 3

    @pytest.mark.asyncio
    async def test_create_batch_call_no_valid_numbers(self):
        result = await retell_create_batch_call([{"email": "x@y.com"}], "+15550000001")
        assert result["status"] == "no_valid_numbers"
        assert result["scheduled_count"] == 0

    @pytest.mark.asyncio
    async def test_get_call_status_returns_dict(self):
        result = await retell_get_call_status("call-123")
        assert result["call_id"] == "call-123"
        assert "status" in result
        assert "duration_seconds" in result

    @pytest.mark.asyncio
    async def test_get_batch_status_returns_dict(self):
        result = await retell_get_batch_status("batch-001")
        assert result["batch_call_id"] == "batch-001"
        assert "total" in result
        assert "completed" in result


# ---------------------------------------------------------------------------
# Google Sheets tools
# ---------------------------------------------------------------------------

class TestSheetsTools:
    @pytest.mark.asyncio
    async def test_append_row_no_credentials(self):
        """Without GOOGLE_ACCESS_TOKEN, returns stub response."""
        result = await sheets_append_row("spreadsheet-123", ["2026-02-25", "alice@example.com", "stub"])
        assert result["spreadsheet_id"] == "spreadsheet-123"
        assert result["updated_rows"] == 1

    @pytest.mark.asyncio
    async def test_log_cycle_with_leads(self):
        cycle_data = {
            "leads": [{"email": "alice@example.com", "phone_number": "+15551234567"}],
            "linkedin_result": {"status": "completed", "sent": 1, "failed": 0},
            "retell_result": {"batch_call_id": "batch-001", "scheduled_count": 1},
            "timestamp": "2026-02-25T00:00:00Z",
        }
        result = await sheets_log_cycle("spreadsheet-123", cycle_data)
        assert result["rows_written"] == 1
        assert result["spreadsheet_id"] == "spreadsheet-123"

    @pytest.mark.asyncio
    async def test_log_cycle_no_leads_writes_one_row(self):
        cycle_data = {
            "leads": [],
            "linkedin_result": {},
            "retell_result": {},
            "timestamp": "2026-02-25T00:00:00Z",
        }
        result = await sheets_log_cycle("spreadsheet-123", cycle_data)
        assert result["rows_written"] == 1

    @pytest.mark.asyncio
    async def test_get_sheet_id_returns_int(self):
        sheet_id = await sheets_get_sheet_id("spreadsheet-123", "Sheet1")
        assert isinstance(sheet_id, int)

    @pytest.mark.asyncio
    async def test_create_sheet_returns_dict(self):
        result = await sheets_create_sheet("spreadsheet-123", "Cycle Logs")
        assert result["sheet_name"] == "Cycle Logs"
        assert result["spreadsheet_id"] == "spreadsheet-123"
        assert isinstance(result["sheet_id"], int)
