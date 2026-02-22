"""Test fixtures: mock LLM, mock Redis, test tenant, sample data.

All tests should use these fixtures for consistency.
"""

import json
import pytest

from nexus.types import PersonaContract, RiskLevel, ToolDefinition, RetrievedContext
from nexus.config import NexusConfig
from nexus.core.personas import PersonaManager
from nexus.core.anomaly import AnomalyEngine
from nexus.core.notary import Notary
from nexus.core.ledger import Ledger
from nexus.core.chain import ChainManager
from nexus.core.verifier import IntentVerifier
from nexus.core.output_validator import OutputValidator
from nexus.core.cot_logger import CoTLogger
from nexus.core.engine import NexusEngine
from nexus.knowledge.context import ContextBuilder
from nexus.tools.registry import ToolRegistry
from nexus.tools.selector import ToolSelector
from nexus.tools.executor import ToolExecutor
from nexus.tools.sandbox import Sandbox
from nexus.reasoning.think_act import ThinkActGate
from nexus.reasoning.continue_complete import ContinueCompleteGate
from nexus.reasoning.escalate import EscalateGate


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


# ── Mock LLM client ───────────────────────────────────────────────────────────

class _MockLLMClient:
    """Returns deterministic responses without hitting any LLM API.

    First call: returns a single-step decomposition using knowledge_search.
    Subsequent calls: return a minimal intent JSON so IntentVerifier doesn't
    need a live LLM.
    """
    def __init__(self, steps: list[dict] | None = None):
        self._steps = steps or [
            {
                "action": "search for information",
                "tool": "knowledge_search",
                "params": {"query": "test query"},
                "persona": "researcher",
            }
        ]
        self._call_count = 0

    async def complete(self, messages, **kwargs) -> dict:
        if self._call_count == 0:
            content = json.dumps(self._steps)
        else:
            # Minimal intent declaration JSON for IntentVerifier fallback
            content = json.dumps({
                "planned_action": "search for information",
                "tool_name": self._steps[0]["tool"],
                "resource_targets": ["kb:docs"],
                "reasoning": "mock reasoning",
                "confidence": 0.9,
            })
        self._call_count += 1
        return {
            "content": content,
            "tool_calls": [],
            "usage": {"input_tokens": 5, "output_tokens": 5},
        }


@pytest.fixture
def mock_llm_client():
    """Mock LLM client with deterministic single-step knowledge_search response."""
    return _MockLLMClient()


# ── Fully wired in-memory NexusEngine fixture ────────────────────────────────

class _MockKnowledgeStore:
    """No-op knowledge store: returns empty context. No ChromaDB needed."""
    def list_namespaces(self, tenant_id: str) -> list[str]:
        return []

    async def query(self, tenant_id, namespace, query, n_results=5, **kwargs) -> RetrievedContext:
        return RetrievedContext(
            query=query,
            documents=[],
            confidence=0.0,
            sources=[],
            namespace=namespace,
        )


@pytest.fixture
def engine(sample_personas, mock_llm_client):
    """Fully wired NexusEngine using all in-memory components.

    - No DB, no Redis, no ChromaDB
    - Mock LLM (deterministic single-step knowledge_search)
    - researcher persona pre-loaded with knowledge_search tool
    """
    cfg = NexusConfig(
        database_url="sqlite+aiosqlite:///test.db",
        redis_url="redis://localhost:6379/15",
        secret_key="test-secret-key-for-fixtures",
    )

    registry = ToolRegistry()
    ks_defn = ToolDefinition(
        name="knowledge_search",
        description="Search the knowledge base",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        risk_level=RiskLevel.LOW,
        resource_pattern="kb:*",
    )

    async def _knowledge_search(query: str) -> str:
        return f"Results for: {query}"

    registry.register(ks_defn, _knowledge_search)

    return NexusEngine(
        persona_manager=PersonaManager(sample_personas),
        anomaly_engine=AnomalyEngine(cfg),  # Gate 2 SKIP (no embeddings), Gate 4 SKIP (no history)
        notary=Notary(),
        ledger=Ledger(),
        chain_manager=ChainManager(),
        context_builder=ContextBuilder(_MockKnowledgeStore()),
        tool_registry=registry,
        tool_selector=ToolSelector(registry, llm_client=mock_llm_client),
        tool_executor=ToolExecutor(registry, Sandbox(), IntentVerifier()),
        output_validator=OutputValidator(),
        cot_logger=CoTLogger(),
        think_act_gate=ThinkActGate(),
        continue_complete_gate=ContinueCompleteGate(),
        escalate_gate=EscalateGate(),
        llm_client=mock_llm_client,
        config=cfg,
    )
