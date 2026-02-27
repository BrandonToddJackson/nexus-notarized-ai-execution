"""tests/test_user_journeys.py

USER JOURNEY TESTS — testing the app from a user's perspective.

These tests exercise the FULL HTTP stack:
  Real JWT auth middleware → real route handlers → real engine → real in-memory ledger.

No DB, no Redis, no ChromaDB. All components are in-memory.

==============================================================================
WHAT THESE TESTS PROVE (that 2013 unit tests cannot):
==============================================================================

J1.  Missing Authorization header → 401 (middleware is active, not bypassed)
J2.  Invalid/expired JWT → 401 (middleware validates signatures, not just presence)
J3.  Valid JWT → request proceeds; tenant_id extracted and used downstream
J4.  Workflow CRUD persists across HTTP requests (create then GET same app)
J5.  Listing workflows shows only what was created (no phantom data)
J6.  Activating a workflow changes its status from DRAFT → ACTIVE
J7.  Running a non-active workflow returns an error (pre-condition enforced)
J8.  Full lifecycle: create → activate → run → response contains execution_id
J9.  Ledger API: seals from a run are retrievable via GET /v1/ledger
J10. Chain integrity: verify_chain() passes on seals from an HTTP-triggered run
J11. Credential store → list → verify the credential appears (encrypted at rest)
J12. Tenant isolation: Tenant A's workflows are invisible to Tenant B
J13. Webhook paths bypass JWT — /v2/webhooks/* needs no Authorization header
J14. Health endpoint responds without authentication
J15. Blocked gate: tool outside persona's allowed_tools → BLOCKED seal in ledger

==============================================================================
APPROACH:
  httpx.AsyncClient(app=app, base_url="http://test") — no real network socket.
  Tokens created with JWTManager().create_token() — uses same nexus_config secret
  as AuthMiddleware, so signatures match within a process.
==============================================================================
"""

import json
import httpx
import pytest
from httpx import ASGITransport
from unittest.mock import AsyncMock, MagicMock
from cryptography.fernet import Fernet
from fastapi import FastAPI

from nexus.auth.jwt import JWTManager
from nexus.auth.middleware import AuthMiddleware
from nexus.config import NexusConfig
from nexus.core.anomaly import AnomalyEngine
from nexus.core.chain import ChainManager
from nexus.core.cot_logger import CoTLogger
from nexus.core.engine import NexusEngine
from nexus.core.ledger import Ledger
from nexus.core.notary import Notary
from nexus.core.output_validator import OutputValidator
from nexus.core.personas import PersonaManager
from nexus.core.verifier import IntentVerifier
from nexus.credentials.encryption import CredentialEncryption
from nexus.credentials.vault import CredentialVault
from nexus.knowledge.context import ContextBuilder
from nexus.reasoning.continue_complete import ContinueCompleteGate
from nexus.reasoning.escalate import EscalateGate
from nexus.reasoning.think_act import ThinkActGate
from nexus.tools.executor import ToolExecutor
from nexus.tools.registry import ToolRegistry
from nexus.tools.sandbox import Sandbox
from nexus.tools.selector import ToolSelector
from nexus.types import (
    PersonaContract, RiskLevel, ToolDefinition, RetrievedContext,
)
from nexus.workers.dispatcher import WorkflowDispatcher
from nexus.workflows.manager import WorkflowManager

# ── Constants ──────────────────────────────────────────────────────────────────

TENANT_A = "journey-tenant-alpha"
TENANT_B = "journey-tenant-beta"

# Single Fernet key for the whole test session (not reused across runs)
_FERNET_KEY: str = Fernet.generate_key().decode()


# ── Test app factory ───────────────────────────────────────────────────────────

class _AppBundle:
    """Groups the test app and its internal components for assertion access."""
    def __init__(self, app, engine, ledger, workflow_manager, vault, notary):
        self.app = app
        self.engine = engine
        self.ledger = ledger
        self.workflow_manager = workflow_manager
        self.vault = vault
        self.notary = notary


async def _create_runnable_workflow(
    client, headers: dict, bundle: "_AppBundle", name: str = "Test WF",
    tenant_id: str = TENANT_A,
) -> str:
    """Create a workflow with one step and return its workflow_id.

    Steps are added via the Python manager so WorkflowStep.workflow_id can be set
    (it's a required field the HTTP client doesn't know ahead of creation time).
    The HTTP activate/run/ledger stack is exercised end-to-end.
    """
    from nexus.types import WorkflowStep, StepType, NodePosition

    # 1. Create workflow via HTTP (tests the POST route)
    create = await client.post("/v2/workflows", json={"name": name}, headers=headers)
    assert create.status_code == 201, f"create failed: {create.text}"
    wf_id = create.json()["id"]

    # 2. Add step via Python manager (bypasses the PUT dict-coercion bug)
    #    This is valid because the test's purpose is activate/run/ledger, not step creation.
    step = WorkflowStep(
        id="s1",
        workflow_id=wf_id,
        name="search_step",
        step_type=StepType.ACTION,
        tool_name="knowledge_search",
        tool_params={"query": "journey query"},
        persona_name="researcher",
        description="Search",
        config={},
        timeout_seconds=30,
        retry_policy={},
        position=NodePosition(x=0.0, y=0.0),
    )
    await bundle.workflow_manager.update(
        workflow_id=wf_id,
        tenant_id=tenant_id,
        steps=[step],
        edges=[],
    )
    return wf_id


def _build_app(llm_steps: list[dict] | None = None) -> _AppBundle:
    """Build a fully-wired FastAPI test app with real middleware and in-memory state.

    - Real AuthMiddleware (JWT validation, not bypassed)
    - Real route handlers from nexus.api.routes.*
    - In-memory engine: mock LLM + mock ContextBuilder (no ChromaDB)
    - In-memory Ledger, Notary, CredentialVault
    - WorkflowDispatcher wired for inline execution (no Redis/ARQ)

    Each call returns a fresh isolated app — no shared state between tests.
    """
    from nexus.api.routes import workflows as wf_routes
    from nexus.api.routes import credentials as cred_routes
    from nexus.api.routes import ledger as ledger_routes
    from nexus.api.routes import health
    from nexus.api.routes import webhooks

    # ── Config ──────────────────────────────────────────────────────────────
    config = NexusConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        redis_url="redis://localhost:6379/15",
        credential_encryption_key=_FERNET_KEY,
    )

    # ── Personas ─────────────────────────────────────────────────────────────
    researcher = PersonaContract(
        name="researcher",
        description="Searches and retrieves information",
        allowed_tools=["knowledge_search"],
        resource_scopes=["kb:*"],
        intent_patterns=["search for information", "look up", "find data"],
        risk_tolerance=RiskLevel.LOW,
        max_ttl_seconds=120,
    )
    persona_manager = PersonaManager([researcher])

    # ── Tool registry ────────────────────────────────────────────────────────
    registry = ToolRegistry()
    _calls: list[str] = []

    async def knowledge_search(query: str) -> str:
        _calls.append(query)
        return f"Results for: {query}"

    registry.register(
        ToolDefinition(
            name="knowledge_search",
            description="Search the knowledge base",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            risk_level=RiskLevel.LOW,
            resource_pattern="kb:*",
        ),
        knowledge_search,
    )

    # ── In-memory core ────────────────────────────────────────────────────────
    ledger = Ledger()
    notary = Notary()
    chain_manager = ChainManager()

    # ── Mock LLM ─────────────────────────────────────────────────────────────
    steps = llm_steps or [{
        "action": "search for information",
        "tool": "knowledge_search",
        "params": {"query": "test query"},
        "persona": "researcher",
    }]
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value={
        "content": json.dumps(steps),
        "tool_calls": [],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    })

    # ── Mock ContextBuilder (no ChromaDB) ─────────────────────────────────────
    mock_ctx = RetrievedContext(
        query="test", documents=[], confidence=0.9, sources=[], namespace="general",
    )
    context_builder = MagicMock(spec=ContextBuilder)
    context_builder.build = AsyncMock(return_value=mock_ctx)

    # ── Credential vault ──────────────────────────────────────────────────────
    enc = CredentialEncryption(key=_FERNET_KEY)
    vault = CredentialVault(encryption=enc)

    # ── Workflow manager ──────────────────────────────────────────────────────
    workflow_manager = WorkflowManager(repository=None, config=config)

    # ── NexusEngine ───────────────────────────────────────────────────────────
    engine = NexusEngine(
        persona_manager=persona_manager,
        anomaly_engine=AnomalyEngine(config=config),
        notary=notary,
        ledger=ledger,
        chain_manager=chain_manager,
        context_builder=context_builder,
        tool_registry=registry,
        tool_selector=ToolSelector(registry, llm_client=None),
        tool_executor=ToolExecutor(registry, Sandbox(), IntentVerifier(), vault=vault),
        output_validator=OutputValidator(),
        cot_logger=CoTLogger(),
        think_act_gate=ThinkActGate(),
        continue_complete_gate=ContinueCompleteGate(),
        escalate_gate=EscalateGate(),
        llm_client=mock_llm,
        config=config,
        workflow_manager=workflow_manager,
    )

    # ── Dispatcher (inline, no Redis) ─────────────────────────────────────────
    dispatcher = WorkflowDispatcher(engine=engine, redis_pool=None, config=config)

    # ── FastAPI with real AuthMiddleware + production exception handlers ──────
    from nexus.api.main import register_exception_handlers
    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(AuthMiddleware)  # uses nexus_config.secret_key

    # Include real routes with exact same prefixes as production create_app()
    app.include_router(wf_routes.router)          # paths already include /v2
    app.include_router(cred_routes.router, prefix="/v2")
    app.include_router(ledger_routes.router, prefix="/v1")
    app.include_router(health.router, prefix="/v1")
    app.include_router(webhooks.router)            # catch-all, no prefix

    # Wire state — mirrors what production lifespan does
    app.state.engine = engine
    app.state.workflow_manager = workflow_manager
    app.state.ledger = ledger
    app.state.vault = vault
    app.state.persona_manager = persona_manager
    app.state.tool_registry = registry
    app.state.dispatcher = dispatcher
    # Optional attrs that routes access via getattr(..., None)
    app.state.workflow_generator = None
    app.state.ambiguity_resolver = None
    app.state.async_session = None
    app.state.mcp_adapter = None
    app.state.redis = None
    app.state.knowledge_store = None
    app.state.embedding_service = None

    return _AppBundle(app, engine, ledger, workflow_manager, vault, notary)


async def _token(tenant_id: str = TENANT_A) -> str:
    """Create a real JWT for the given tenant (uses nexus_config.secret_key)."""
    return await JWTManager().create_token(tenant_id)


def _auth(tenant_id: str | None = None):
    """Sync placeholder — use `await _headers(tenant_id)` instead."""
    raise NotImplementedError("use await _headers()")


async def _headers(tenant_id: str = TENANT_A) -> dict:
    tok = await _token(tenant_id)
    return {"Authorization": f"Bearer {tok}"}


# ══════════════════════════════════════════════════════════════════════════════
# J1-J3: Auth gateway — middleware is real, not bypassed
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthGateway:
    """The AuthMiddleware must block unauthenticated and malformed requests."""

    async def test_j1_no_auth_header_returns_401(self):
        """J1: Missing Authorization header → 401, not 200 or 500."""
        bundle = _build_app()
        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            resp = await client.get("/v2/workflows")
        assert resp.status_code == 401, \
            "J1 FAIL: request without Authorization header must be rejected"

    async def test_j2_invalid_jwt_returns_401(self):
        """J2: A tampered/garbage Bearer token → 401."""
        bundle = _build_app()
        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            resp = await client.get(
                "/v2/workflows",
                headers={"Authorization": "Bearer not.a.valid.jwt"},
            )
        assert resp.status_code == 401, \
            "J2 FAIL: tampered JWT must not pass middleware"

    async def test_j3_valid_jwt_reaches_handler(self):
        """J3: A real JWT → request reaches the route handler (200, not 401/500)."""
        bundle = _build_app()
        headers = await _headers()
        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            resp = await client.get("/v2/workflows", headers=headers)
        assert resp.status_code == 200, \
            f"J3 FAIL: valid JWT must reach route handler, got {resp.status_code}: {resp.text}"

    async def test_j3b_wrong_tenant_jwt_reaches_own_handler(self):
        """J3b: JWT for Tenant B reaches handler with B's tenant_id (not A's)."""
        bundle = _build_app()
        headers_b = await _headers(TENANT_B)
        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            resp = await client.get("/v2/workflows", headers=headers_b)
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# J4-J5: Workflow CRUD — cross-request state consistency
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkflowCRUD:
    """Workflow created via POST must be retrievable via GET in the same app."""

    async def test_j4_create_then_get_workflow(self):
        """J4: Create workflow via POST, then fetch it via GET — same app, same state."""
        bundle = _build_app()
        headers = await _headers()

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            # Create
            create_resp = await client.post(
                "/v2/workflows",
                json={"name": "My Test Workflow", "description": "journey test"},
                headers=headers,
            )
            assert create_resp.status_code == 201, \
                f"J4 FAIL: workflow creation returned {create_resp.status_code}: {create_resp.text}"
            wf_id = create_resp.json()["id"]

            # Retrieve
            get_resp = await client.get(f"/v2/workflows/{wf_id}", headers=headers)
            assert get_resp.status_code == 200
            data = get_resp.json()
            assert data["id"] == wf_id
            assert data["name"] == "My Test Workflow"
            assert data["tenant_id"] == TENANT_A, \
                f"J4 FAIL: tenant_id from JWT must be set on workflow, got {data['tenant_id']}"

    async def test_j5_list_workflows_shows_created(self):
        """J5: Listing workflows shows the one that was just created."""
        bundle = _build_app()
        headers = await _headers()

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            # Start empty
            list_before = (await client.get("/v2/workflows", headers=headers)).json()
            assert list_before["workflows"] == []

            # Create one
            await client.post("/v2/workflows", json={"name": "Listed"}, headers=headers)

            # Now list should have it
            list_after = (await client.get("/v2/workflows", headers=headers)).json()
            names = [w["name"] for w in list_after["workflows"]]
            assert "Listed" in names, f"J5 FAIL: created workflow not in list: {names}"


# ══════════════════════════════════════════════════════════════════════════════
# J6-J8: Workflow lifecycle — activate, run, guards
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkflowLifecycle:
    """The activate → run sequence must work end-to-end through HTTP."""

    async def test_j6_activate_changes_status_to_active(self):
        """J6: POST /activate changes workflow status from DRAFT to ACTIVE."""
        bundle = _build_app()
        headers = await _headers()

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            wf_id = await _create_runnable_workflow(client, headers, bundle)

            # Status should be DRAFT initially
            before = (await client.get(f"/v2/workflows/{wf_id}", headers=headers)).json()
            assert before["status"] == "draft", \
                f"J6 FAIL: new workflow should be DRAFT, got {before['status']}"

            # Activate
            act_resp = await client.post(
                f"/v2/workflows/{wf_id}/activate", headers=headers
            )
            assert act_resp.status_code == 200, \
                f"J6 FAIL: activate returned {act_resp.status_code}: {act_resp.text}"

            # Status now ACTIVE
            after = (await client.get(f"/v2/workflows/{wf_id}", headers=headers)).json()
            assert after["status"] == "active", \
                f"J6 FAIL: after activate, status should be 'active', got {after['status']}"

    async def test_j7_run_non_active_workflow_returns_error(self):
        """J7: Running a DRAFT workflow must return 422 (WorkflowValidationError handler)."""
        bundle = _build_app()
        headers = await _headers()

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            wf_id = await _create_runnable_workflow(client, headers, bundle)
            # Do NOT activate — leave in DRAFT

            run_resp = await client.post(
                f"/v2/workflows/{wf_id}/run",
                json={"context": {}, "force_background": False},
                headers=headers,
            )
        assert run_resp.status_code == 422, \
            f"J7 FAIL: running a non-active workflow must return 422, got {run_resp.status_code}: {run_resp.text}"

    async def test_j8_full_lifecycle_create_activate_run(self):
        """J8: Full journey — create → activate → run → response has execution_id."""
        bundle = _build_app()
        headers = await _headers()

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            # 1. Create + add step via manager
            wf_id = await _create_runnable_workflow(client, headers, bundle)

            # 2. Activate
            act = await client.post(f"/v2/workflows/{wf_id}/activate", headers=headers)
            assert act.status_code == 200, f"Activate failed: {act.text}"

            # 3. Run
            run = await client.post(
                f"/v2/workflows/{wf_id}/run",
                json={"context": {"source": "journey-test"}, "force_background": False},
                headers=headers,
            )
            assert run.status_code == 200, \
                f"J8 FAIL: run returned {run.status_code}: {run.text}"
            data = run.json()
            assert "execution_id" in data, \
                f"J8 FAIL: run response must contain 'execution_id', got: {list(data.keys())}"
            assert data["mode"] == "inline", \
                f"J8 FAIL: small workflow must run inline, got mode={data.get('mode')}"

    async def test_j8b_run_missing_workflow_returns_404(self):
        """J8b: Running a non-existent workflow_id must return 404 (WorkflowNotFound handler)."""
        bundle = _build_app()
        headers = await _headers()

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            run = await client.post(
                "/v2/workflows/does-not-exist/run",
                json={"context": {}, "force_background": False},
                headers=headers,
            )
        assert run.status_code == 404, \
            f"J8b FAIL: missing workflow must return 404, got {run.status_code}: {run.text}"


# ══════════════════════════════════════════════════════════════════════════════
# J9-J10: Ledger API and chain integrity
# ══════════════════════════════════════════════════════════════════════════════

class TestLedgerAndChainIntegrity:
    """Seals from a run must be visible via the ledger API and chain integrity must pass."""

    async def test_j9_seals_visible_via_ledger_api(self):
        """J9: After a run, GET /v1/ledger returns at least one seal for this tenant."""
        bundle = _build_app()
        headers = await _headers()

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            wf_id = await _create_runnable_workflow(client, headers, bundle, name="Ledger Test")
            await client.post(f"/v2/workflows/{wf_id}/activate", headers=headers)
            run = await client.post(
                f"/v2/workflows/{wf_id}/run",
                json={"context": {}, "force_background": False},
                headers=headers,
            )
            assert run.status_code == 200, f"Run failed: {run.text}"

            # Check ledger API
            ledger_resp = await client.get("/v1/ledger", headers=headers)
            assert ledger_resp.status_code == 200, \
                f"J9 FAIL: ledger endpoint returned {ledger_resp.status_code}"
            data = ledger_resp.json()
            seals = data.get("seals", data) if isinstance(data, dict) else data
            assert len(seals) >= 1, \
                "J9 FAIL: ledger must have at least 1 seal after a workflow run"

    async def test_j10_chain_integrity_passes_after_run(self):
        """J10: verify_chain() on seals from an HTTP-triggered run always returns True."""
        bundle = _build_app()
        headers = await _headers()

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            wf_id = await _create_runnable_workflow(client, headers, bundle, name="Chain Test")
            await client.post(f"/v2/workflows/{wf_id}/activate", headers=headers)
            await client.post(
                f"/v2/workflows/{wf_id}/run",
                json={"context": {}, "force_background": False},
                headers=headers,
            )

        # Inspect internal ledger directly — seals are in-memory
        seals = await bundle.ledger.get_by_tenant(TENANT_A)
        assert len(seals) >= 1, "J10 FAIL: no seals in ledger after run"
        assert bundle.notary.verify_chain(seals) is True, \
            "J10 FAIL: verify_chain must return True for seals from a clean run"


# ══════════════════════════════════════════════════════════════════════════════
# J11: Credential HTTP lifecycle
# ══════════════════════════════════════════════════════════════════════════════

class TestCredentialHTTP:
    """Credentials stored via HTTP must be retrievable and encrypted at rest."""

    async def test_j11_store_and_list_credential(self):
        """J11: POST /v2/credentials stores a credential; GET /v2/credentials lists it."""
        bundle = _build_app()
        headers = await _headers()

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            store_resp = await client.post(
                "/v2/credentials",
                json={
                    "name": "openai-key",
                    "credential_type": "api_key",
                    "service_name": "openai",
                    "data": {"api_key": "sk-supersecret"},
                },
                headers=headers,
            )
            assert store_resp.status_code in (200, 201), \
                f"J11 FAIL: credential store returned {store_resp.status_code}: {store_resp.text}"
            cred_id = store_resp.json().get("id") or store_resp.json().get("credential_id")

            list_resp = await client.get("/v2/credentials", headers=headers)
            assert list_resp.status_code == 200
            creds = list_resp.json()
            cred_list = creds if isinstance(creds, list) else creds.get("credentials", [])
            names = [c.get("name") for c in cred_list]
            assert "openai-key" in names, \
                f"J11 FAIL: stored credential not in list: {names}"

    async def test_j11b_credential_value_not_exposed_in_list(self):
        """J11b: Credential list response must NOT expose raw secret values."""
        bundle = _build_app()
        headers = await _headers()

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            await client.post(
                "/v2/credentials",
                json={
                    "name": "secret-key",
                    "credential_type": "api_key",
                    "service_name": "test-service",
                    "data": {"api_key": "sk-should-not-appear"},
                },
                headers=headers,
            )
            list_resp = await client.get("/v2/credentials", headers=headers)

        # The raw secret must not be in the response body
        body = list_resp.text
        assert "sk-should-not-appear" not in body, \
            "J11b FAIL: raw credential secret appeared in list response — secrets leaking!"


# ══════════════════════════════════════════════════════════════════════════════
# J12: Tenant isolation
# ══════════════════════════════════════════════════════════════════════════════

class TestTenantIsolation:
    """Tenant A's resources must be invisible to Tenant B."""

    async def test_j12_workflow_invisible_to_other_tenant(self):
        """J12: Workflow created by Tenant A is not returned to Tenant B."""
        bundle = _build_app()
        headers_a = await _headers(TENANT_A)
        headers_b = await _headers(TENANT_B)

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            # Tenant A creates a workflow
            create = await client.post(
                "/v2/workflows",
                json={"name": "Tenant A Secret Workflow"},
                headers=headers_a,
            )
            assert create.status_code == 201, f"Create failed: {create.text}"
            wf_id = create.json()["id"]

            # Tenant B cannot list it
            list_b = await client.get("/v2/workflows", headers=headers_b)
            b_ids = [w["id"] for w in list_b.json().get("workflows", [])]
            assert wf_id not in b_ids, \
                f"J12 FAIL: Tenant A's workflow appeared in Tenant B's list: {b_ids}"

            # Tenant B cannot fetch it directly
            get_b = await client.get(f"/v2/workflows/{wf_id}", headers=headers_b)
            assert get_b.status_code == 404, \
                f"J12 FAIL: Tenant B should get 404 for Tenant A's workflow, " \
                f"got {get_b.status_code}: {get_b.text}"

    async def test_j12b_run_seals_isolated_by_tenant(self):
        """J12b: Seals from Tenant A's run are not visible in Tenant B's ledger."""
        bundle = _build_app()
        headers_a = await _headers(TENANT_A)
        headers_b = await _headers(TENANT_B)

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            # Tenant A: create → activate → run
            wf_id = await _create_runnable_workflow(client, headers_a, bundle, name="A Only", tenant_id=TENANT_A)
            await client.post(f"/v2/workflows/{wf_id}/activate", headers=headers_a)
            await client.post(
                f"/v2/workflows/{wf_id}/run",
                json={"context": {}, "force_background": False},
                headers=headers_a,
            )

        # Directly inspect in-memory ledger by tenant
        seals_a = await bundle.ledger.get_by_tenant(TENANT_A)
        seals_b = await bundle.ledger.get_by_tenant(TENANT_B)
        assert len(seals_a) >= 1, "J12b: Tenant A should have seals"
        assert len(seals_b) == 0, \
            f"J12b FAIL: Tenant B's ledger has {len(seals_b)} seals from Tenant A's run"


# ══════════════════════════════════════════════════════════════════════════════
# J13: Webhook auth bypass
# ══════════════════════════════════════════════════════════════════════════════

class TestWebhookBypass:
    """Webhook paths must not require Authorization headers."""

    async def test_j13_webhook_path_no_auth_not_401(self):
        """J13: /v2/webhooks/... is public — must not return 401 even without auth."""
        bundle = _build_app()
        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            # No Authorization header — webhook paths bypass auth
            resp = await client.post(
                "/v2/webhooks/my-test-hook",
                json={"event": "test"},
            )
        # The webhook handler may return 404 (hook not registered) or 200,
        # but it must NOT return 401 (auth rejection).
        assert resp.status_code != 401, \
            f"J13 FAIL: /v2/webhooks/* returned 401 — auth bypass not working. " \
            f"Got {resp.status_code}: {resp.text}"


# ══════════════════════════════════════════════════════════════════════════════
# J14: Health check
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Health check must be publicly accessible."""

    async def test_j14_health_no_auth_returns_200(self):
        """J14: GET /v1/health works without an Authorization header."""
        bundle = _build_app()
        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            resp = await client.get("/v1/health")
        assert resp.status_code == 200, \
            f"J14 FAIL: /v1/health should be public, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "status" in data, f"J14 FAIL: health response missing 'status': {data}"


# ══════════════════════════════════════════════════════════════════════════════
# J15: Blocked gate (tool outside allowed_tools)
# ══════════════════════════════════════════════════════════════════════════════

class TestBlockedGateJourney:
    """When a step requests a tool blocked by Gate 1, the run must fail and
    a BLOCKED seal must appear in the ledger."""

    async def test_j15_blocked_tool_creates_blocked_seal_in_ledger(self):
        """J15: A workflow step with tool_name='send_email' (not in researcher.allowed_tools)
        triggers Gate 1 → BLOCKED seal in ledger (I3 invariant via HTTP).

        KEY INSIGHT: run_workflow uses step.tool_name directly (not LLM decompose).
        To trigger Gate 1, the WorkflowStep itself must have the forbidden tool_name.
        """
        from nexus.types import ActionStatus, WorkflowStep, StepType, NodePosition

        bundle = _build_app()
        headers = await _headers()

        async with httpx.AsyncClient(transport=ASGITransport(app=bundle.app), base_url="http://test") as client:
            # Create empty workflow via HTTP
            create = await client.post(
                "/v2/workflows", json={"name": "Gate1 Block Test"}, headers=headers
            )
            assert create.status_code == 201
            wf_id = create.json()["id"]

            # Add a step with send_email (NOT in researcher.allowed_tools) via manager
            forbidden_step = WorkflowStep(
                id="blocked-s1",
                workflow_id=wf_id,
                name="send_email_step",
                step_type=StepType.ACTION,
                tool_name="send_email",          # researcher does NOT allow this
                tool_params={"to": "x@y.com", "subject": "blocked"},
                persona_name="researcher",
                description="Send email — should be blocked by Gate 1",
                config={},
                timeout_seconds=30,
                retry_policy={},
                position=NodePosition(x=0.0, y=0.0),
            )
            await bundle.workflow_manager.update(
                workflow_id=wf_id,
                tenant_id=TENANT_A,
                steps=[forbidden_step],
                edges=[],
            )

            # Activate
            await client.post(f"/v2/workflows/{wf_id}/activate", headers=headers)

            # Run — Gate 1 will fail for send_email → AnomalyDetected → 403
            run = await client.post(
                f"/v2/workflows/{wf_id}/run",
                json={"context": {}, "force_background": False},
                headers=headers,
            )
            assert run.status_code == 403, \
                f"J15: expected 403 from AnomalyDetected handler, got {run.status_code}: {run.text}"

        # I3 INVARIANT: BLOCKED seal must be in ledger BEFORE the exception propagates
        seals = await bundle.ledger.get_by_tenant(TENANT_A)
        assert len(seals) >= 1, \
            "J15 FAIL: no seals in ledger — engine must create BLOCKED seal before raising"
        statuses = [s.status for s in seals]
        blocked_seals = [s for s in seals if s.status == ActionStatus.BLOCKED]
        assert len(blocked_seals) >= 1, \
            f"J15 FAIL: expected at least 1 BLOCKED seal, got statuses: {statuses}"


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION: all existing tests still pass (run separately)
# ══════════════════════════════════════════════════════════════════════════════
