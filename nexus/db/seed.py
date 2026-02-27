"""Seed database with demo data.

Creates:
- 1 demo tenant with API key "nxs_demo_key_12345"
- 5 default personas: researcher, analyst, creator, communicator, operator
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
        "name": "campaign_researcher",
        "description": "Researches leads, target audiences, and fetches external data via API and web tools",
        "allowed_tools": ["rag_query", "rag_ingest", "knowledge_search", "web_search", "web_fetch", "http_request", "instantly_get_campaigns", "instantly_get_campaign_analytics", "instantly_get_leads", "instantly_audit", "instantly_get_sender_health"],
        "resource_scopes": ["kb:*", "rag:*", "web:*", "http:*"],
        "intent_patterns": ["research leads", "find target audience", "analyze campaign data", "search for information", "fetch api data", "research optimization"],
        "risk_tolerance": "low",
        "max_ttl_seconds": 120,
    },
    {
        "name": "campaign_outreach",
        "description": "Executes personalized outreach using campaign knowledge",
        "allowed_tools": ["rag_query", "http_request", "send_email", "knowledge_search", "instantly_get_campaigns", "instantly_get_campaign_analytics", "instantly_get_leads", "instantly_add_leads", "instantly_move_leads", "instantly_create_campaign", "instantly_activate_campaign"],
        "resource_scopes": ["kb:*", "rag:*", "http:*"],
        "intent_patterns": ["send campaign message", "personalize outreach", "add lead to campaign", "execute campaign step", "create campaign", "launch campaign", "activate campaign", "build email sequence"],
        "risk_tolerance": "medium",
        "max_ttl_seconds": 60,
    },
    {
        "name": "campaign_optimizer",
        "description": "Optimizes campaign performance — pulls metrics from APIs, researches best practices, uses sequential reasoning to produce recommendations",
        "allowed_tools": [
            "rag_query", "compute_stats", "http_request", "knowledge_search",
            "web_search", "web_fetch", "mcp_sequential_thinking_sequentialthinking",
            "instantly_get_campaigns", "instantly_get_campaign_analytics", "instantly_get_leads",
            "instantly_audit", "instantly_get_sender_health",
        ],
        "resource_scopes": ["kb:*", "rag:*", "http:*", "web:*"],
        "intent_patterns": ["optimize campaign performance", "analyze metrics", "adjust targeting", "compute statistics", "recommend improvements", "pull campaign data", "reason through optimization", "check sender health", "audit campaigns", "sender warmup status", "deliverability check"],
        "risk_tolerance": "low",
        "max_ttl_seconds": 180,
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
        else:
            # Update mutable fields so re-seeding picks up tool/scope changes
            for field in ("allowed_tools", "resource_scopes", "intent_patterns",
                          "description", "risk_tolerance", "max_ttl_seconds"):
                if field in persona_data:
                    setattr(existing, field, persona_data[field])

    await session.commit()


async def seed_v2_data(session: AsyncSession):
    """Create Phase 28 demo data: workflow, trigger, credentials.

    Idempotent — checks for existing records before inserting.
    """
    import uuid
    from nexus.db.models import WorkflowModel, TriggerModel, CredentialModel
    from datetime import datetime, timezone

    # Demo workflow 1: Email Classify and Respond
    result = await session.execute(
        select(WorkflowModel).where(
            WorkflowModel.tenant_id == "demo",
            WorkflowModel.name == "Email Classify and Respond",
        )
    )
    if result.scalar_one_or_none() is None:
        workflow_id = str(uuid.uuid4())
        workflow = WorkflowModel(
            id=workflow_id,
            tenant_id="demo",
            name="Email Classify and Respond",
            description="Classify incoming emails and send appropriate responses",
            version=1,
            status="active",
            trigger_config={"type": "webhook"},
            steps=[
                {"id": "s1", "name": "Fetch Email", "step_type": "action", "tool_name": "http_request", "persona_name": "operator", "config": {}},
                {"id": "s2", "name": "Classify Intent", "step_type": "action", "tool_name": "knowledge_search", "persona_name": "analyst", "config": {}},
                {"id": "s3", "name": "Generate Reply", "step_type": "action", "tool_name": "knowledge_search", "persona_name": "creator", "config": {}},
                {"id": "s4", "name": "Send Response", "step_type": "action", "tool_name": "http_request", "persona_name": "communicator", "config": {}},
            ],
            edges=[
                {"id": "e1", "source_step_id": "s1", "target_step_id": "s2", "edge_type": "default"},
                {"id": "e2", "source_step_id": "s2", "target_step_id": "s3", "edge_type": "default"},
                {"id": "e3", "source_step_id": "s3", "target_step_id": "s4", "edge_type": "default"},
            ],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            created_by="seed",
            tags=["email", "automation"],
            settings={},
        )
        session.add(workflow)
        await session.flush()

        trigger = TriggerModel(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            tenant_id="demo",
            trigger_type="webhook",
            enabled=True,
            config={"method": "POST"},
            webhook_path="/webhooks/demo/email-inbound",
            last_triggered_at=None,
            created_at=datetime.now(timezone.utc),
        )
        session.add(trigger)

    # Demo workflow 2: Instantly Cold Email Campaign — AI PM Recruiters
    result = await session.execute(
        select(WorkflowModel).where(
            WorkflowModel.tenant_id == "demo",
            WorkflowModel.name == "Instantly Cold Email Campaign — AI PM Recruiters",
        )
    )
    if result.scalar_one_or_none() is None:
        wf2_id = str(uuid.uuid4())
        workflow2 = WorkflowModel(
            id=wf2_id,
            tenant_id="demo",
            name="Instantly Cold Email Campaign — AI PM Recruiters",
            description=(
                "Personalized cold email outreach to AI Product Manager recruiters "
                "via Instantly.ai. Queries RAG for ICP profile, fetches leads, "
                "generates a tailored intro showcasing your portfolio, and adds each "
                "lead to the active Instantly campaign. Sealed in the NEXUS ledger at "
                "every gate."
            ),
            version=1,
            status="active",
            trigger_config={"type": "manual"},
            steps=[
                {
                    "id": "step_rag_query",
                    "name": "Query ICP Profile",
                    "step_type": "action",
                    "tool_name": "rag_query",
                    "persona_name": "campaign_researcher",
                    "tool_params": {
                        "query": "Ideal customer profile: AI Product Manager recruiters, portfolio showcasing criteria",
                        "namespace": "campaign",
                        "mode": "hybrid",
                    },
                    "description": "Fetch ICP context from RAG knowledge base (gated: scope + intent + TTL + drift)",
                    "config": {},
                },
                {
                    "id": "step_fetch_leads",
                    "name": "Fetch Leads from Instantly",
                    "step_type": "action",
                    "tool_name": "http_request",
                    "persona_name": "campaign_outreach",
                    "tool_params": {
                        "method": "GET",
                        "url": "https://api.instantly.ai/api/v1/lead/list",
                        "headers": {"Authorization": "Bearer {{credentials.instantly_api_key}}"},
                        "params": {"campaign_id": "{{config.instantly_campaign_id}}", "limit": "50"},
                    },
                    "description": "Pull uncontacted leads from Instantly.ai campaign",
                    "config": {},
                },
                {
                    "id": "step_personalize",
                    "name": "Personalize Email Copy",
                    "step_type": "action",
                    "tool_name": "rag_query",
                    "persona_name": "campaign_researcher",
                    "tool_params": {
                        "query": "Write personalized cold email opening for AI PM recruiter using portfolio highlights",
                        "namespace": "campaign",
                        "mode": "hybrid",
                    },
                    "description": "Generate personalized intro line using RAG-enriched campaign context",
                    "config": {},
                },
                {
                    "id": "step_add_to_campaign",
                    "name": "Add Lead to Instantly Campaign",
                    "step_type": "action",
                    "tool_name": "http_request",
                    "persona_name": "campaign_outreach",
                    "tool_params": {
                        "method": "POST",
                        "url": "https://api.instantly.ai/api/v1/lead/add",
                        "headers": {
                            "Authorization": "Bearer {{credentials.instantly_api_key}}",
                            "Content-Type": "application/json",
                        },
                        "body": {
                            "campaign_id": "{{config.instantly_campaign_id}}",
                            "leads": [{"email": "{{step.fetch_leads.email}}", "personalization": "{{step.personalize.result}}"}],
                        },
                    },
                    "description": "Add personalized lead to the live Instantly campaign",
                    "config": {},
                },
                {
                    "id": "step_log_metrics",
                    "name": "Log Campaign Metrics",
                    "step_type": "action",
                    "tool_name": "compute_stats",
                    "persona_name": "campaign_optimizer",
                    "tool_params": {
                        "data": "{{step.add_to_campaign.result}}",
                        "operation": "count_added",
                    },
                    "description": "Track leads added, gate pass-rate, and campaign velocity",
                    "config": {},
                },
            ],
            edges=[
                {"id": "e_rq_fl", "source_step_id": "step_rag_query", "target_step_id": "step_fetch_leads", "edge_type": "default"},
                {"id": "e_fl_p", "source_step_id": "step_fetch_leads", "target_step_id": "step_personalize", "edge_type": "default"},
                {"id": "e_p_ac", "source_step_id": "step_personalize", "target_step_id": "step_add_to_campaign", "edge_type": "default"},
                {"id": "e_ac_lm", "source_step_id": "step_add_to_campaign", "target_step_id": "step_log_metrics", "edge_type": "default"},
            ],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            created_by="seed",
            tags=["campaign", "cold-email", "instantly", "ai-pm", "rag"],
            settings={},
        )
        session.add(workflow2)

    # Demo workflow 3: Instantly Campaign Intelligence & Optimizer
    result = await session.execute(
        select(WorkflowModel).where(
            WorkflowModel.tenant_id == "demo",
            WorkflowModel.name == "Instantly Campaign Intelligence",
        )
    )
    if result.scalar_one_or_none() is None:
        wf3_id = str(uuid.uuid4())
        workflow3 = WorkflowModel(
            id=wf3_id,
            tenant_id="demo",
            name="Instantly Campaign Intelligence",
            description=(
                "Pulls live campaign data from the Instantly.ai API, fetches "
                "community optimization tips from Reddit r/sales, reads the "
                "Instantly deliverability guide, and synthesizes 5 specific "
                "actionable recommendations to improve open rate and reply rate. "
                "Set your Instantly API key in NEXUS Credentials as 'instantly_api_key' "
                "before running."
            ),
            version=1,
            status="active",
            trigger_config={"type": "manual"},
            steps=[
                {
                    "id": "ci_list_campaigns",
                    "name": "List Active Campaigns",
                    "step_type": "action",
                    "tool_name": "http_request",
                    "persona_name": "campaign_optimizer",
                    "tool_params": {
                        "method": "GET",
                        "url": "https://api.instantly.ai/api/v1/campaign/list",
                        "params": {
                            "api_key": "{{config.instantly_api_key}}",
                            "limit": "10",
                            "skip": "0",
                        },
                        "extract": "campaigns",
                    },
                    "description": "Fetch all campaigns from Instantly.ai and extract the campaigns array",
                    "config": {},
                },
                {
                    "id": "ci_analytics",
                    "name": "Fetch Campaign Analytics",
                    "step_type": "action",
                    "tool_name": "http_request",
                    "persona_name": "campaign_optimizer",
                    "tool_params": {
                        "method": "GET",
                        "url": "https://api.instantly.ai/api/v1/analytics/overview",
                        "params": {
                            "api_key": "{{config.instantly_api_key}}",
                        },
                    },
                    "description": "Get aggregate analytics: open rate, reply rate, bounce rate across all campaigns",
                    "config": {},
                },
                {
                    "id": "ci_reddit",
                    "name": "Research Reddit: Cold Email Tips",
                    "step_type": "action",
                    "tool_name": "web_fetch",
                    "persona_name": "campaign_researcher",
                    "tool_params": {
                        "url": "https://www.reddit.com/r/sales/search.json?q=instantly+ai+cold+email+optimization&sort=relevance&limit=5&t=year&type=link",
                    },
                    "description": "Fetch top Reddit r/sales posts discussing Instantly.ai optimization strategies",
                    "config": {},
                },
                {
                    "id": "ci_docs",
                    "name": "Fetch Instantly Deliverability Guide",
                    "step_type": "action",
                    "tool_name": "web_fetch",
                    "persona_name": "campaign_researcher",
                    "tool_params": {
                        "url": "https://help.instantly.ai/en/articles/8636369-email-deliverability-guide",
                    },
                    "description": "Read the official Instantly.ai deliverability best practices guide",
                    "config": {},
                },
                {
                    "id": "ci_web_research",
                    "name": "Research Cold Email Best Practices",
                    "step_type": "action",
                    "tool_name": "web_search",
                    "persona_name": "campaign_researcher",
                    "tool_params": {
                        "query": "cold email open rate improvement best practices 2025 subject line personalization",
                    },
                    "description": "Search for general cold email optimization strategies and benchmarks",
                    "config": {},
                },
                {
                    "id": "ci_synthesize",
                    "name": "Sequential Reasoning: Optimization Report",
                    "step_type": "action",
                    "tool_name": "mcp_sequential_thinking_sequentialthinking",
                    "persona_name": "campaign_optimizer",
                    "tool_params": {
                        "thought": (
                            "I have gathered: (1) active Instantly.ai campaign list with statuses, "
                            "(2) aggregate analytics — open rate, reply rate, bounce rate, "
                            "(3) Reddit r/sales discussions on instantly.ai optimization, "
                            "(4) official Instantly deliverability guide content, "
                            "(5) general cold email best practices. "
                            "Now I will reason step by step through each metric, compare against benchmarks "
                            "(open rate >30%, reply rate >3%, bounce <2%), identify the top failure modes, "
                            "cross-reference community advice and official docs, and produce exactly 5 "
                            "prioritized, specific, actionable recommendations with implementation details."
                        ),
                        "thoughtNumber": 1,
                        "totalThoughts": 5,
                        "nextThoughtNeeded": True,
                    },
                    "description": "Use sequential thinking MCP to reason through the optimization problem step-by-step and produce a prioritized report",
                    "config": {},
                },
            ],
            edges=[
                {"id": "ci_e1", "source_step_id": "ci_list_campaigns", "target_step_id": "ci_analytics", "edge_type": "default"},
                {"id": "ci_e2", "source_step_id": "ci_analytics", "target_step_id": "ci_reddit", "edge_type": "default"},
                {"id": "ci_e3", "source_step_id": "ci_reddit", "target_step_id": "ci_docs", "edge_type": "default"},
                {"id": "ci_e4", "source_step_id": "ci_docs", "target_step_id": "ci_web_research", "edge_type": "default"},
                {"id": "ci_e5", "source_step_id": "ci_web_research", "target_step_id": "ci_synthesize", "edge_type": "default"},
            ],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            created_by="seed",
            tags=["instantly", "campaign-intelligence", "optimization", "cold-email", "analytics"],
            settings={
                "instantly_api_key_credential": "instantly_api_key",
                "instantly_api_docs": "https://developer.instantly.ai/",
                "quick_run_hint": (
                    "Go to Execute page, pick 'campaign_optimizer' persona, and type: "
                    "Fetch my active Instantly.ai campaigns using API key YOUR_KEY, "
                    "get their open/reply/bounce rates, research Reddit r/sales for optimization tips, "
                    "read https://help.instantly.ai/en/articles/8636369-email-deliverability-guide, "
                    "then give me 5 specific improvements."
                ),
            },
        )
        session.add(workflow3)

    # Demo workflow 4: Campaign Health Audit
    result = await session.execute(
        select(WorkflowModel).where(
            WorkflowModel.tenant_id == "demo",
            WorkflowModel.name == "Campaign Health Audit",
        )
    )
    if result.scalar_one_or_none() is None:
        wf4_id = str(uuid.uuid4())
        workflow4 = WorkflowModel(
            id=wf4_id,
            tenant_id="demo",
            name="Campaign Health Audit",
            description=(
                "Single-step full health report for all Instantly.ai campaigns: "
                "lead counts per campaign, cross-campaign duplicate detection, "
                "sender warmup status, daily sending capacity, and actionable "
                "recommendations. Requires INSTANTLY_API_KEY in .env."
            ),
            version=1,
            status="active",
            trigger_config={"type": "manual"},
            steps=[
                {
                    "id": "cha_audit",
                    "name": "Full Campaign Audit",
                    "step_type": "action",
                    "tool_name": "instantly_audit",
                    "persona_name": "campaign_optimizer",
                    "tool_params": {},
                    "description": "Run full Instantly.ai health report",
                    "config": {},
                },
            ],
            edges=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            created_by="seed",
            tags=["instantly", "audit", "campaign-health", "cold-email"],
            settings={},
        )
        session.add(workflow4)

    # Demo workflow 5: Sender Health Check
    result = await session.execute(
        select(WorkflowModel).where(
            WorkflowModel.tenant_id == "demo",
            WorkflowModel.name == "Sender Health Check",
        )
    )
    if result.scalar_one_or_none() is None:
        wf5_id = str(uuid.uuid4())
        workflow5 = WorkflowModel(
            id=wf5_id,
            tenant_id="demo",
            name="Sender Health Check",
            description=(
                "Checks warmup status and daily sending capacity for all "
                "Instantly.ai sender accounts. Shows which senders are warmed "
                "and ready, which are not, and total daily send capacity. "
                "Requires INSTANTLY_API_KEY in .env."
            ),
            version=1,
            status="active",
            trigger_config={"type": "manual"},
            steps=[
                {
                    "id": "shc_health",
                    "name": "Check Sender Health",
                    "step_type": "action",
                    "tool_name": "instantly_get_sender_health",
                    "persona_name": "campaign_optimizer",
                    "tool_params": {},
                    "description": "Get warmup status and daily capacity for all senders",
                    "config": {},
                },
            ],
            edges=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            created_by="seed",
            tags=["instantly", "sender-health", "deliverability", "warmup"],
            settings={},
        )
        session.add(workflow5)

    # Seed MCP Servers
    from nexus.db.models import MCPServerModel
    for mcp_data in [
        {
            "name": "sequential-thinking",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            "env": {},
            "enabled": True,
            "discovered_tools": ["mcp_sequential_thinking_sequentialthinking"],
        },
    ]:
        result = await session.execute(
            select(MCPServerModel).where(
                MCPServerModel.tenant_id == "demo",
                MCPServerModel.name == mcp_data["name"],
            )
        )
        if result.scalar_one_or_none() is None:
            mcp = MCPServerModel(
                id=str(uuid.uuid4()),
                tenant_id="demo",
                created_at=datetime.now(timezone.utc),
                **mcp_data,
            )
            session.add(mcp)

    # Demo credentials
    for cred_data in [
        {
            "name": "Demo OpenAI Key",
            "credential_type": "api_key",
            "service_name": "openai",
            "encrypted_data": "demo-encrypted-openai-key",
            "scoped_personas": ["researcher", "analyst"],
        },
        {
            "name": "Demo Slack Token",
            "credential_type": "bearer_token",
            "service_name": "slack",
            "encrypted_data": "demo-encrypted-slack-token",
            "scoped_personas": ["communicator"],
        },
    ]:
        result = await session.execute(
            select(CredentialModel).where(
                CredentialModel.tenant_id == "demo",
                CredentialModel.name == cred_data["name"],
            )
        )
        if result.scalar_one_or_none() is None:
            cred = CredentialModel(
                id=str(uuid.uuid4()),
                tenant_id="demo",
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                **cred_data,
            )
            session.add(cred)

    await session.commit()


async def run_seed(session: AsyncSession):
    """Run all seed functions: base data + v2 demo data."""
    await seed_database(session)
    await seed_v2_data(session)


if __name__ == "__main__":
    import asyncio
    from nexus.db.database import async_session, init_db

    async def main():
        await init_db()
        async with async_session() as session:
            await run_seed(session)
        print("Seed complete.")

    asyncio.run(main())
