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


async def seed_v2_data(session: AsyncSession):
    """Create Phase 28 demo data: workflow, trigger, credentials.

    Idempotent — checks for existing records before inserting.
    """
    import uuid
    from nexus.db.models import WorkflowModel, TriggerModel, CredentialModel
    from datetime import datetime, timezone

    # Demo workflow: Email Classify and Respond
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
                {"id": "s1", "name": "Fetch Email", "step_type": "tool", "tool_name": "http_request", "config": {}},
                {"id": "s2", "name": "Classify Intent", "step_type": "tool", "tool_name": "knowledge_search", "config": {}},
                {"id": "s3", "name": "Generate Reply", "step_type": "tool", "tool_name": "knowledge_search", "config": {}},
                {"id": "s4", "name": "Send Response", "step_type": "tool", "tool_name": "http_request", "config": {}},
            ],
            edges=[
                {"id": "e1", "source": "s1", "target": "s2", "edge_type": "default"},
                {"id": "e2", "source": "s2", "target": "s3", "edge_type": "default"},
                {"id": "e3", "source": "s3", "target": "s4", "edge_type": "default"},
            ],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            created_by="seed",
            tags=["email", "automation"],
            settings={},
        )
        session.add(workflow)
        await session.flush()  # get the ID before creating trigger

        # Webhook trigger for the demo workflow
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
