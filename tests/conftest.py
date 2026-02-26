"""Test fixtures: mock LLM, mock Redis, test tenant, sample data.

All tests should use these fixtures for consistency.
"""

import json
import pytest
from cryptography.fernet import Fernet

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


# ── Phase 30: v2 fixtures ─────────────────────────────────────────────────────

# Well-known tenant IDs for isolation tests
TENANT_A = "tenant-alpha-001"
TENANT_B = "tenant-beta-002"

# Stable Fernet key for the entire test session (ephemeral — regenerated each run)
_TEST_FERNET_KEY: str = Fernet.generate_key().decode()


@pytest.fixture
def test_config():
    """NexusConfig with encryption key and safe test defaults."""
    return NexusConfig(
        debug=True,
        database_url="sqlite+aiosqlite:///test.db",
        redis_url="redis://localhost:6379/15",
        default_llm_model="mock/test-model",
        secret_key="test-secret-key-phase30",
        credential_encryption_key=_TEST_FERNET_KEY,
    )


@pytest.fixture
def encryption():
    """CredentialEncryption backed by the session Fernet key."""
    from nexus.credentials.encryption import CredentialEncryption
    return CredentialEncryption(key=_TEST_FERNET_KEY)


@pytest.fixture
def vault(encryption):
    """In-memory CredentialVault — no DB required."""
    from nexus.credentials.vault import CredentialVault
    return CredentialVault(encryption=encryption)


@pytest.fixture
def linear_workflow_def():
    """Two-step linear workflow: step_a → step_b."""
    from nexus.types import WorkflowDefinition, WorkflowStep, WorkflowEdge, StepType, EdgeType
    wf_id = "wf-linear-test"
    step_a = WorkflowStep(id="step-a", workflow_id=wf_id, step_type=StepType.ACTION, name="step_a", tool_name="knowledge_search")
    step_b = WorkflowStep(id="step-b", workflow_id=wf_id, step_type=StepType.ACTION, name="step_b", tool_name="knowledge_search")
    edge = WorkflowEdge(id="edge-1", workflow_id=wf_id, source_step_id="step-a", target_step_id="step-b")
    return WorkflowDefinition(
        id=wf_id,
        tenant_id=TENANT_A,
        name="Linear Workflow",
        steps=[step_a, step_b],
        edges=[edge],
    )


@pytest.fixture
def cyclic_workflow_def():
    """Intentionally cyclic workflow — for validation failure tests."""
    from nexus.types import WorkflowDefinition, WorkflowStep, WorkflowEdge, StepType, EdgeType
    wf_id = "wf-cyclic-test"
    step_a = WorkflowStep(id="step-a", workflow_id=wf_id, step_type=StepType.ACTION, name="step_a")
    step_b = WorkflowStep(id="step-b", workflow_id=wf_id, step_type=StepType.ACTION, name="step_b")
    edge_ab = WorkflowEdge(id="edge-ab", workflow_id=wf_id, source_step_id="step-a", target_step_id="step-b")
    edge_ba = WorkflowEdge(id="edge-ba", workflow_id=wf_id, source_step_id="step-b", target_step_id="step-a")
    return WorkflowDefinition(
        id=wf_id,
        tenant_id=TENANT_A,
        name="Cyclic Workflow",
        steps=[step_a, step_b],
        edges=[edge_ab, edge_ba],
    )


@pytest.fixture
def branch_workflow_def():
    """Branch workflow: branch_node → [path_b (default), path_c (conditional)]."""
    from nexus.types import WorkflowDefinition, WorkflowStep, WorkflowEdge, StepType, EdgeType
    wf_id = "wf-branch-test"
    step_a = WorkflowStep(id="step-a", workflow_id=wf_id, step_type=StepType.BRANCH, name="branch_node")
    step_b = WorkflowStep(id="step-b", workflow_id=wf_id, step_type=StepType.ACTION, name="path_b")
    step_c = WorkflowStep(id="step-c", workflow_id=wf_id, step_type=StepType.ACTION, name="path_c")
    edge_ab = WorkflowEdge(
        id="e-ab", workflow_id=wf_id,
        source_step_id="step-a", target_step_id="step-b",
        edge_type=EdgeType.DEFAULT,
    )
    edge_ac = WorkflowEdge(
        id="e-ac", workflow_id=wf_id,
        source_step_id="step-a", target_step_id="step-c",
        edge_type=EdgeType.CONDITIONAL, condition="status == 'error'",
    )
    return WorkflowDefinition(
        id=wf_id,
        tenant_id=TENANT_A,
        name="Branch Workflow",
        steps=[step_a, step_b, step_c],
        edges=[edge_ab, edge_ac],
    )


@pytest.fixture
def loop_workflow_def():
    """Loop workflow: init → loop_node (loop_back to body, default to exit)."""
    from nexus.types import WorkflowDefinition, WorkflowStep, WorkflowEdge, StepType, EdgeType
    wf_id = "wf-loop-test"
    step_a = WorkflowStep(id="step-a", workflow_id=wf_id, step_type=StepType.ACTION, name="init")
    step_l = WorkflowStep(id="step-l", workflow_id=wf_id, step_type=StepType.LOOP, name="loop_node", config={"max_iterations": 5})
    step_b = WorkflowStep(id="step-b", workflow_id=wf_id, step_type=StepType.ACTION, name="loop_body")
    step_c = WorkflowStep(id="step-c", workflow_id=wf_id, step_type=StepType.ACTION, name="exit_step")
    edge_al = WorkflowEdge(id="e-al", workflow_id=wf_id, source_step_id="step-a", target_step_id="step-l")
    edge_lb = WorkflowEdge(id="e-lb", workflow_id=wf_id, source_step_id="step-l", target_step_id="step-b", edge_type=EdgeType.LOOP_BACK)
    edge_lc = WorkflowEdge(id="e-lc", workflow_id=wf_id, source_step_id="step-l", target_step_id="step-c", edge_type=EdgeType.DEFAULT)
    return WorkflowDefinition(
        id=wf_id,
        tenant_id=TENANT_A,
        name="Loop Workflow",
        steps=[step_a, step_l, step_b, step_c],
        edges=[edge_al, edge_lb, edge_lc],
    )


@pytest.fixture
def parallel_workflow_def():
    """Parallel workflow: entry → [parallel_a, parallel_b] → merge."""
    from nexus.types import WorkflowDefinition, WorkflowStep, WorkflowEdge, StepType, EdgeType
    wf_id = "wf-parallel-test"
    step_e = WorkflowStep(id="step-e", workflow_id=wf_id, step_type=StepType.ACTION, name="entry")
    step_pa = WorkflowStep(id="step-pa", workflow_id=wf_id, step_type=StepType.PARALLEL, name="parallel_a")
    step_pb = WorkflowStep(id="step-pb", workflow_id=wf_id, step_type=StepType.PARALLEL, name="parallel_b")
    step_m = WorkflowStep(id="step-m", workflow_id=wf_id, step_type=StepType.ACTION, name="merge")
    edge_epa = WorkflowEdge(id="e-epa", workflow_id=wf_id, source_step_id="step-e", target_step_id="step-pa")
    edge_epb = WorkflowEdge(id="e-epb", workflow_id=wf_id, source_step_id="step-e", target_step_id="step-pb")
    edge_pam = WorkflowEdge(id="e-pam", workflow_id=wf_id, source_step_id="step-pa", target_step_id="step-m")
    edge_pbm = WorkflowEdge(id="e-pbm", workflow_id=wf_id, source_step_id="step-pb", target_step_id="step-m")
    return WorkflowDefinition(
        id=wf_id,
        tenant_id=TENANT_A,
        name="Parallel Workflow",
        steps=[step_e, step_pa, step_pb, step_m],
        edges=[edge_epa, edge_epb, edge_pam, edge_pbm],
    )


@pytest.fixture
async def auth_headers():
    """JWT Authorization headers for TENANT_A (uses global config.secret_key)."""
    from nexus.auth.jwt import JWTManager
    token = await JWTManager().create_token(TENANT_A)
    return {"Authorization": f"Bearer {token}"}
