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
