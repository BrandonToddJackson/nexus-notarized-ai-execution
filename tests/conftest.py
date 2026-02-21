"""Test fixtures: mock LLM, mock Redis, test tenant, sample data.

All tests should use these fixtures for consistency.
"""

import pytest
import asyncio
from typing import AsyncGenerator

from nexus.types import PersonaContract, RiskLevel, TrustTier
from nexus.config import NexusConfig


@pytest.fixture
def config():
    """Test configuration with safe defaults."""
    return NexusConfig(
        debug=True,
        database_url="sqlite+aiosqlite:///test.db",
        redis_url="redis://localhost:6379/15",  # test DB
        default_llm_model="mock/test-model",
        secret_key="test-secret-key",
    )


@pytest.fixture
def sample_personas():
    """5 default personas for testing."""
    return [
        PersonaContract(
            name="researcher",
            description="Searches and retrieves information",
            allowed_tools=["knowledge_search", "web_search", "web_fetch", "file_read"],
            resource_scopes=["kb:*", "web:*", "file:read:*"],
            intent_patterns=["search for information", "find data about", "look up", "research"],
            risk_tolerance=RiskLevel.LOW,
            max_ttl_seconds=60,
        ),
        PersonaContract(
            name="analyst",
            description="Analyzes data and computes statistics",
            allowed_tools=["knowledge_search", "compute_stats", "file_read", "file_write"],
            resource_scopes=["kb:*", "file:*", "data:*"],
            intent_patterns=["analyze data", "compute statistics", "calculate"],
            risk_tolerance=RiskLevel.MEDIUM,
            max_ttl_seconds=120,
        ),
        PersonaContract(
            name="creator",
            description="Creates content",
            allowed_tools=["knowledge_search", "file_write"],
            resource_scopes=["kb:*", "file:write:*"],
            intent_patterns=["write", "create", "draft", "generate content"],
            risk_tolerance=RiskLevel.LOW,
            max_ttl_seconds=90,
        ),
        PersonaContract(
            name="communicator",
            description="Sends emails and messages",
            allowed_tools=["knowledge_search", "send_email", "file_read"],
            resource_scopes=["kb:*", "email:*", "file:read:*"],
            intent_patterns=["send email", "notify", "communicate"],
            risk_tolerance=RiskLevel.HIGH,
            max_ttl_seconds=60,
        ),
        PersonaContract(
            name="operator",
            description="Executes code and system operations",
            allowed_tools=["knowledge_search", "file_read", "file_write", "compute_stats"],
            resource_scopes=["kb:*", "file:*", "system:*"],
            intent_patterns=["execute", "run", "deploy", "configure"],
            risk_tolerance=RiskLevel.HIGH,
            max_ttl_seconds=180,
        ),
    ]


@pytest.fixture
def test_tenant_id():
    """Standard test tenant ID."""
    return "test-tenant-001"
