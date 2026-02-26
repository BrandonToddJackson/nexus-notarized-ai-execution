"""Built-in tools package. Import to register all built-in tools."""

from nexus.tools.builtin.web import web_search, web_fetch
from nexus.tools.builtin.files import file_read, file_write
from nexus.tools.builtin.comms import send_email
from nexus.tools.builtin.data import compute_stats, knowledge_search
from nexus.tools.builtin.sales_growth import (
    # Path 1: Audit & Optimize
    instantly_list_campaigns, instantly_get_campaign_detail, instantly_get_campaign_analytics,
    instantly_audit_campaigns, instantly_get_leads, instantly_get_accounts,
    instantly_update_campaign_settings, instantly_update_campaign_sequence,
    # Path 2: Create
    instantly_create_campaign, instantly_activate_campaign, instantly_pause_campaign,
    instantly_get_lead_lists, instantly_add_lead, instantly_move_leads_to_campaign,
    # Outbound cycle
    instantly_get_warm_leads, instantly_mark_lead_as_contacted,
    # Phone enrichment
    enrich_lead_phone,
    # LinkedIn
    linkedin_send_dm, linkedin_get_profile, craft_linkedin_message,
    # Retell
    retell_create_batch_call, retell_get_call_status, retell_get_batch_status,
    # Sheets
    sheets_append_row, sheets_log_cycle, sheets_get_sheet_id, sheets_create_sheet,
)

__all__ = [
    "web_search", "web_fetch",
    "file_read", "file_write",
    "send_email",
    "compute_stats", "knowledge_search",
    # Instantly — path 1
    "instantly_list_campaigns", "instantly_get_campaign_detail", "instantly_get_campaign_analytics",
    "instantly_audit_campaigns", "instantly_get_leads", "instantly_get_accounts",
    "instantly_update_campaign_settings", "instantly_update_campaign_sequence",
    # Instantly — path 2
    "instantly_create_campaign", "instantly_activate_campaign", "instantly_pause_campaign",
    "instantly_get_lead_lists", "instantly_add_lead", "instantly_move_leads_to_campaign",
    # Instantly — outbound cycle
    "instantly_get_warm_leads", "instantly_mark_lead_as_contacted",
    # Phone enrichment
    "enrich_lead_phone",
    # LinkedIn
    "linkedin_send_dm", "linkedin_get_profile", "craft_linkedin_message",
    # Retell
    "retell_create_batch_call", "retell_get_call_status", "retell_get_batch_status",
    # Sheets
    "sheets_append_row", "sheets_log_cycle", "sheets_get_sheet_id", "sheets_create_sheet",
]
