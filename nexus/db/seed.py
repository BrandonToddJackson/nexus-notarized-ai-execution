"""Seed database with demo data.

Creates:
- 1 demo tenant with API key "nxs_demo_key_12345"
- 6 default personas: researcher, analyst, creator, communicator, operator,
  sales_growth_agent
- All built-in tools registered
"""

import hashlib
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from nexus.db.models import TenantModel, PersonaModel

DEFAULT_PERSONAS = [
    {
        "name": "researcher",
        "description": "Searches and retrieves information from knowledge bases and the web",
        "allowed_tools": ["knowledge_search", "web_search", "web_fetch", "file_read"],
        "resource_scopes": ["kb:*", "web:*", "file:read:*"],
        "intent_patterns": ["search for information", "find data about", "look up", "research"],
        "risk_tolerance": "low",
        "max_ttl_seconds": 60,
    },
    {
        "name": "analyst",
        "description": "Analyzes data and computes statistics",
        "allowed_tools": ["knowledge_search", "compute_stats", "file_read", "file_write"],
        "resource_scopes": ["kb:*", "file:*", "data:*"],
        "intent_patterns": ["analyze data", "compute statistics", "calculate", "summarize findings"],
        "risk_tolerance": "medium",
        "max_ttl_seconds": 120,
    },
    {
        "name": "creator",
        "description": "Creates content: documents, reports, summaries",
        "allowed_tools": ["knowledge_search", "file_write"],
        "resource_scopes": ["kb:*", "file:write:*"],
        "intent_patterns": ["write", "create", "draft", "generate content", "compose"],
        "risk_tolerance": "low",
        "max_ttl_seconds": 90,
    },
    {
        "name": "communicator",
        "description": "Sends emails and messages",
        "allowed_tools": ["knowledge_search", "send_email", "file_read"],
        "resource_scopes": ["kb:*", "email:*", "file:read:*"],
        "intent_patterns": ["send email", "notify", "communicate", "message"],
        "risk_tolerance": "high",
        "max_ttl_seconds": 60,
    },
    {
        "name": "operator",
        "description": "Executes code and system operations",
        "allowed_tools": ["knowledge_search", "file_read", "file_write", "compute_stats"],
        "resource_scopes": ["kb:*", "file:*", "system:*"],
        "intent_patterns": ["execute", "run", "deploy", "configure", "operate"],
        "risk_tolerance": "high",
        "max_ttl_seconds": 180,
    },
    {
        "name": "sales_growth_agent",
        "description": (
            "Manages Instantly email campaigns end-to-end as an always-on assistant: audits active campaigns, "
            "checks analytics and sender health, creates and activates new campaigns, sends LinkedIn DMs to warm leads, "
            "triggers Retell voice calls, and logs all activity to Google Sheets."
        ),
        # Gate 1 — Scope: must match every @tool registered in sales_growth.py
        "allowed_tools": [
            # Path 1 — Audit & Optimize
            "instantly_list_campaigns", "instantly_get_campaign_detail",
            "instantly_get_campaign_analytics", "instantly_audit_campaigns",
            "instantly_get_leads", "instantly_get_accounts",
            "instantly_update_campaign_settings", "instantly_update_campaign_sequence",
            # Path 2 — Create
            "instantly_create_campaign", "instantly_activate_campaign",
            "instantly_pause_campaign", "instantly_get_lead_lists",
            "instantly_add_lead", "instantly_move_leads_to_campaign",
            # Outbound cycle
            "instantly_get_warm_leads", "instantly_mark_lead_as_contacted",
            "enrich_lead_phone",
            "linkedin_send_connection_request", "linkedin_send_dm", "linkedin_get_profile", "craft_linkedin_message",
            "retell_create_batch_call", "retell_get_call_status", "retell_get_batch_status",
            "sheets_append_row", "sheets_log_cycle", "sheets_get_sheet_id", "sheets_create_sheet",
        ],
        # Gate 1 — Scope: must match resource_pattern of every allowed tool
        "resource_scopes": [
            "crm:campaigns:*", "crm:leads:*",
            "enrichment:phone:*",
            "social:linkedin:*", "voice:calls:*", "data:sheets:*",
        ],
        # Gate 2 — Intent: short phrases (3-6 words) that cosine-match orchestrator
        # task strings with all-MiniLM-L6-v2 at the 0.30 dev threshold.
        "intent_patterns": [
            # Path 1
            "audit email campaigns", "check campaign performance",
            "list active campaigns", "get campaign analytics",
            "review campaign health", "check sender warmup status",
            "get leads in campaign", "update campaign settings",
            "update email sequence",
            # Path 2
            "create email campaign", "build outbound campaign",
            "activate campaign sending", "pause email campaign",
            "add lead to campaign", "move leads to campaign",
            # Outbound cycle
            "poll warm leads",
            "send linkedin message", "schedule voice calls",
            "log sales cycle results", "run outbound sales cycle",
        ],
        "risk_tolerance": "high",
        # Gate 3 — TTL: 600s (10 min) covers full cycle (5 steps × up to 60s each)
        "max_ttl_seconds": 600,
    },
]


async def seed_database(session: AsyncSession):
    """Create demo data: tenant, personas, tools.

    Args:
        session: Async database session
    """
    # Create demo tenant if not already present
    result = await session.execute(
        select(TenantModel).where(TenantModel.id == "demo")
    )
    tenant = result.scalar_one_or_none()
    if tenant is None:
        api_key_hash = hashlib.sha256(b"nxs_demo_key_12345").hexdigest()
        tenant = TenantModel(
            id="demo",
            name="Demo Tenant",
            api_key_hash=api_key_hash,
        )
        session.add(tenant)
        await session.commit()

    # Create each persona if not already present
    for persona_data in DEFAULT_PERSONAS:
        result = await session.execute(
            select(PersonaModel).where(
                PersonaModel.tenant_id == "demo",
                PersonaModel.name == persona_data["name"],
            )
        )
        existing = result.scalar_one_or_none()
        if existing is None:
            persona = PersonaModel(tenant_id="demo", **persona_data)
            session.add(persona)

    await session.commit()
