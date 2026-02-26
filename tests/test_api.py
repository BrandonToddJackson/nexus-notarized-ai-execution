"""API integration tests: all endpoints, auth, full request/response validation.

Coverage:
  GET  /v1/health          — status, version, services dict
  POST /v1/auth/token      — valid key, invalid key, missing body, malformed key
  GET  /v1/ledger          — requires auth, pagination params, empty response
  GET  /v1/ledger/{id}     — chain-specific seals
  GET  /v1/personas        — requires auth, lists personas
  POST /v1/personas        — create persona, invalid risk_tolerance → 422
  GET  /v1/tools           — requires auth, lists tools
  POST /v1/execute         — requires auth (401 without), with auth returns chain_id/status/seals
  Validation               — missing required fields → 422
"""

import pytest
from fastapi.testclient import TestClient

from nexus.api.main import app
from nexus.core.ledger import Ledger
from nexus.core.personas import PersonaManager
from nexus.tools.registry import ToolRegistry
from nexus.types import PersonaContract, RiskLevel


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """Return Authorization header with a valid JWT for the demo tenant."""
    resp = client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
    assert resp.status_code == 200
    token = resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client_with_state(client):
    """Client with minimal app.state wired for routes that read from it."""
    client.app.state.ledger = Ledger()
    client.app.state.tool_registry = ToolRegistry()
    client.app.state.persona_manager = PersonaManager([
        PersonaContract(
            name="researcher",
            description="Searches information",
            allowed_tools=["knowledge_search"],
            resource_scopes=["kb:*"],
            intent_patterns=["search for information"],
            risk_tolerance=RiskLevel.LOW,
            max_ttl_seconds=60,
        )
    ])
    return client


# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200

    def test_health_has_status_field(self, client):
        data = client.get("/v1/health").json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_has_version_field(self, client):
        data = client.get("/v1/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert data["version"]  # non-empty

    def test_health_has_services_dict(self, client):
        data = client.get("/v1/health").json()
        assert "services" in data
        assert isinstance(data["services"], dict)

    def test_health_api_service_key_present(self, client):
        data = client.get("/v1/health").json()
        assert "api" in data["services"]
        assert data["services"]["api"] is True

    def test_health_no_auth_required(self, client):
        """Health check must be publicly accessible without auth."""
        resp = client.get("/v1/health")
        assert resp.status_code == 200


# ── Auth endpoint ──────────────────────────────────────────────────────────────

class TestAuthEndpoint:

    def test_valid_api_key_returns_200(self, client):
        resp = client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
        assert resp.status_code == 200

    def test_valid_api_key_returns_token(self, client):
        resp = client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
        data = resp.json()
        assert "token" in data
        assert isinstance(data["token"], str)
        assert len(data["token"]) > 20

    def test_valid_api_key_returns_tenant_id(self, client):
        resp = client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
        assert resp.json()["tenant_id"] == "demo"

    def test_valid_api_key_returns_expires_in(self, client):
        resp = client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
        assert resp.json()["expires_in"] > 0

    def test_invalid_api_key_returns_401(self, client):
        resp = client.post("/v1/auth/token", json={"api_key": "nxs_totally_wrong"})
        assert resp.status_code == 401

    def test_empty_api_key_returns_401(self, client):
        resp = client.post("/v1/auth/token", json={"api_key": ""})
        assert resp.status_code == 401

    def test_missing_api_key_field_returns_422(self, client):
        resp = client.post("/v1/auth/token", json={})
        assert resp.status_code == 422

    def test_auth_endpoint_public(self, client):
        """Auth endpoint must work without any Authorization header."""
        resp = client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
        assert resp.status_code == 200


# ── Auth protection ────────────────────────────────────────────────────────────

class TestAuthProtection:

    def test_execute_without_auth_returns_401(self, client):
        resp = client.post("/v1/execute", json={"task": "test"})
        assert resp.status_code == 401

    def test_ledger_without_auth_returns_401(self, client):
        resp = client.get("/v1/ledger")
        assert resp.status_code == 401

    def test_ledger_chain_without_auth_returns_401(self, client):
        resp = client.get("/v1/ledger/some-chain-id")
        assert resp.status_code == 401

    def test_personas_without_auth_returns_401(self, client):
        resp = client.get("/v1/personas")
        assert resp.status_code == 401

    def test_create_persona_without_auth_returns_401(self, client):
        resp = client.post("/v1/personas", json={
            "name": "test", "description": "Test persona",
            "allowed_tools": ["knowledge_search"],
            "resource_scopes": ["kb:*"],
            "intent_patterns": ["search"],
        })
        assert resp.status_code == 401

    def test_tools_without_auth_returns_401(self, client):
        resp = client.get("/v1/tools")
        assert resp.status_code == 401

    def test_knowledge_namespaces_without_auth_returns_401(self, client):
        resp = client.get("/v1/knowledge/namespaces")
        assert resp.status_code == 401


# ── Ledger endpoint ────────────────────────────────────────────────────────────

class TestLedgerEndpoint:

    def test_ledger_with_auth_returns_200(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/ledger", headers=auth_headers)
        assert resp.status_code == 200

    def test_ledger_response_has_seals_list(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/ledger", headers=auth_headers)
        data = resp.json()
        assert "seals" in data
        assert isinstance(data["seals"], list)

    def test_ledger_response_has_total(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/ledger", headers=auth_headers)
        data = resp.json()
        assert "total" in data
        assert isinstance(data["total"], int)

    def test_ledger_response_has_limit_and_offset(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/ledger?limit=50&offset=10", headers=auth_headers)
        data = resp.json()
        assert data["limit"] == 50
        assert data["offset"] == 10

    def test_ledger_limit_validation_min(self, client_with_state, auth_headers):
        """limit must be >= 1."""
        resp = client_with_state.get("/v1/ledger?limit=0", headers=auth_headers)
        assert resp.status_code == 422

    def test_ledger_limit_validation_max(self, client_with_state, auth_headers):
        """limit must be <= 500."""
        resp = client_with_state.get("/v1/ledger?limit=501", headers=auth_headers)
        assert resp.status_code == 422

    def test_ledger_offset_cannot_be_negative(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/ledger?offset=-1", headers=auth_headers)
        assert resp.status_code == 422

    def test_ledger_chain_endpoint_unknown_returns_404(self, client_with_state, auth_headers):
        """Unknown chain IDs return 404 — existence of another tenant's chain is not confirmed."""
        resp = client_with_state.get("/v1/ledger/nonexistent-chain", headers=auth_headers)
        assert resp.status_code == 404

    def test_ledger_chain_response_has_seals_when_found(self, client_with_state, auth_headers):
        """Chain detail returns seals list when the chain exists for this tenant."""
        from nexus.types import (
            Seal, IntentDeclaration, AnomalyResult, GateResult, GateVerdict,
            RiskLevel, ActionStatus,
        )
        import asyncio
        intent = IntentDeclaration(
            task_description="t", planned_action="p", tool_name="knowledge_search",
            tool_params={}, resource_targets=[], reasoning="",
        )
        gate = GateResult(gate_name="scope", verdict=GateVerdict.PASS, score=1.0, threshold=1.0, details="")
        anomaly = AnomalyResult(
            gates=[gate, gate, gate, gate], overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW, persona_uuid="researcher", action_fingerprint="fp",
        )
        seal = Seal(
            chain_id="my-chain-001", step_index=0, tenant_id="demo",
            persona_id="researcher", intent=intent, anomaly_result=anomaly,
            tool_name="knowledge_search", tool_params={}, status=ActionStatus.EXECUTED,
        )
        asyncio.run(client_with_state.app.state.ledger.append(seal))
        resp = client_with_state.get("/v1/ledger/my-chain-001", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["chain_id"] == "my-chain-001"
        assert "seals" in data
        assert isinstance(data["seals"], list)
        assert len(data["seals"]) == 1

    def test_ledger_empty_for_fresh_tenant(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/ledger", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 0
        assert data["seals"] == []


# ── Personas endpoint ──────────────────────────────────────────────────────────

class TestPersonasEndpoint:

    def test_list_personas_with_auth_returns_200(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/personas", headers=auth_headers)
        assert resp.status_code == 200

    def test_list_personas_response_has_personas_list(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/personas", headers=auth_headers)
        data = resp.json()
        assert "personas" in data
        assert isinstance(data["personas"], list)

    def test_list_personas_returns_loaded_personas(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/personas", headers=auth_headers)
        personas = resp.json()["personas"]
        assert len(personas) >= 1
        names = {p["name"] for p in personas}
        assert "researcher" in names

    def test_persona_response_has_required_fields(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/personas", headers=auth_headers)
        persona = resp.json()["personas"][0]
        for field in ("id", "name", "description", "allowed_tools", "resource_scopes",
                      "intent_patterns", "max_ttl_seconds", "risk_tolerance", "is_active"):
            assert field in persona, f"Missing field: {field}"

    def test_create_persona_invalid_risk_tolerance_returns_422(self, client_with_state, auth_headers):
        resp = client_with_state.post("/v1/personas", headers=auth_headers, json={
            "name": "bad_persona",
            "description": "Test",
            "allowed_tools": ["knowledge_search"],
            "resource_scopes": ["kb:*"],
            "intent_patterns": ["search"],
            "risk_tolerance": "ultra_mega_high",  # invalid
        })
        assert resp.status_code == 422

    def test_create_persona_missing_name_returns_422(self, client_with_state, auth_headers):
        resp = client_with_state.post("/v1/personas", headers=auth_headers, json={
            "description": "Missing name",
            "allowed_tools": ["knowledge_search"],
            "resource_scopes": ["kb:*"],
            "intent_patterns": ["search"],
        })
        assert resp.status_code == 422


# ── Tools endpoint ─────────────────────────────────────────────────────────────

class TestToolsEndpoint:

    def test_list_tools_with_auth_returns_200(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/tools", headers=auth_headers)
        assert resp.status_code == 200

    def test_list_tools_response_has_tools_list(self, client_with_state, auth_headers):
        resp = client_with_state.get("/v1/tools", headers=auth_headers)
        data = resp.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)

    def test_tool_response_has_required_fields(self, client_with_state, auth_headers):
        """If any tools are registered, they must have name/description/risk_level."""
        # Register a tool so the list is non-empty
        from nexus.types import ToolDefinition, RiskLevel
        defn = ToolDefinition(
            name="test_tool_api",
            description="API test tool",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            risk_level=RiskLevel.LOW,
            resource_pattern="*",
        )
        async def _fn(query: str) -> str:
            return query
        client_with_state.app.state.tool_registry.register(defn, _fn)

        resp = client_with_state.get("/v1/tools", headers=auth_headers)
        tools = resp.json()["tools"]
        tool = next(t for t in tools if t["name"] == "test_tool_api")
        for field in ("name", "description", "risk_level", "requires_approval"):
            assert field in tool, f"Missing field: {field}"

    def test_empty_registry_returns_empty_list(self, client, auth_headers):
        """A registry with no tools returns an empty list (not an error)."""
        client.app.state.tool_registry = ToolRegistry()
        resp = client.get("/v1/tools", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["tools"] == []


# ── Execute endpoint ───────────────────────────────────────────────────────────

class TestExecuteEndpoint:

    def test_execute_requires_auth(self, client):
        resp = client.post("/v1/execute", json={"task": "test"})
        assert resp.status_code == 401

    def test_execute_missing_task_returns_422(self, client, auth_headers):
        resp = client.post("/v1/execute", json={}, headers=auth_headers)
        assert resp.status_code == 422

    def test_execute_task_too_long_returns_422(self, client, auth_headers):
        """Task field has max_length=10000."""
        resp = client.post("/v1/execute", json={"task": "x" * 10001}, headers=auth_headers)
        assert resp.status_code == 422

    def test_execute_with_engine_returns_chain_response(self, client_with_state, auth_headers):
        """Wire a mock engine into app.state and verify response schema."""
        from unittest.mock import AsyncMock
        from nexus.types import ChainPlan, ChainStatus

        # Build a mock chain result
        mock_chain = ChainPlan(
            tenant_id="demo",
            task="search for NEXUS docs",
            steps=[{"action": "search", "tool": "knowledge_search", "params": {}, "persona": "researcher"}],
            status=ChainStatus.COMPLETED,
        )

        mock_engine = AsyncMock()
        mock_engine.run = AsyncMock(return_value=mock_chain)
        client_with_state.app.state.engine = mock_engine

        resp = client_with_state.post(
            "/v1/execute",
            json={"task": "search for NEXUS docs"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "chain_id" in data
        assert "status" in data
        assert "seals" in data
        assert isinstance(data["seals"], list)
        assert "duration_ms" in data
        assert "cost" in data

    def test_execute_response_cost_has_required_fields(self, client_with_state, auth_headers):
        from unittest.mock import AsyncMock
        from nexus.types import ChainPlan, ChainStatus

        mock_chain = ChainPlan(
            tenant_id="demo",
            task="test",
            steps=[],
            status=ChainStatus.COMPLETED,
        )
        mock_engine = AsyncMock()
        mock_engine.run = AsyncMock(return_value=mock_chain)
        client_with_state.app.state.engine = mock_engine

        resp = client_with_state.post(
            "/v1/execute",
            json={"task": "test"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        cost = resp.json()["cost"]
        assert "input_tokens" in cost
        assert "output_tokens" in cost
        assert "total_cost_usd" in cost

    def test_execute_with_persona_parameter(self, client_with_state, auth_headers):
        """persona field is optional — must be accepted without error."""
        from unittest.mock import AsyncMock
        from nexus.types import ChainPlan, ChainStatus

        mock_chain = ChainPlan(
            tenant_id="demo", task="test",
            steps=[], status=ChainStatus.COMPLETED,
        )
        mock_engine = AsyncMock()
        mock_engine.run = AsyncMock(return_value=mock_chain)
        client_with_state.app.state.engine = mock_engine

        resp = client_with_state.post(
            "/v1/execute",
            json={"task": "search for docs", "persona": "researcher"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        # Verify persona was forwarded to engine.run as persona_name kwarg
        call_kwargs = mock_engine.run.call_args
        assert call_kwargs is not None, "engine.run was never called"
        assert call_kwargs.kwargs.get("persona_name") == "researcher", (
            f"Route must forward body.persona → persona_name='researcher', got: {call_kwargs}"
        )


# ── Execute error paths (Gap 2) ────────────────────────────────────────────────

class TestExecuteErrorPaths:

    def test_anomaly_detected_returns_200_blocked(self, client_with_state, auth_headers):
        """AnomalyDetected → 200 with status='blocked' (chain was sealed but blocked)."""
        from unittest.mock import AsyncMock, MagicMock
        from nexus.exceptions import AnomalyDetected

        exc = AnomalyDetected("scope gate failed", gate_results=[], chain_id="blocked-chain-1")
        mock_engine = MagicMock()
        mock_engine.run = AsyncMock(side_effect=exc)
        client_with_state.app.state.engine = mock_engine

        resp = client_with_state.post(
            "/v1/execute", json={"task": "blocked task"}, headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "blocked"

    def test_chain_aborted_returns_422(self, client_with_state, auth_headers):
        from unittest.mock import AsyncMock, MagicMock
        from nexus.exceptions import ChainAborted

        mock_engine = MagicMock()
        mock_engine.run = AsyncMock(side_effect=ChainAborted("aborted"))
        client_with_state.app.state.engine = mock_engine

        resp = client_with_state.post(
            "/v1/execute", json={"task": "aborted task"}, headers=auth_headers
        )
        assert resp.status_code == 422

    def test_escalation_required_returns_422(self, client_with_state, auth_headers):
        from unittest.mock import AsyncMock, MagicMock
        from nexus.exceptions import EscalationRequired

        mock_engine = MagicMock()
        mock_engine.run = AsyncMock(side_effect=EscalationRequired("need human"))
        client_with_state.app.state.engine = mock_engine

        resp = client_with_state.post(
            "/v1/execute", json={"task": "escalate"}, headers=auth_headers
        )
        assert resp.status_code == 422

    def test_unexpected_exception_returns_500(self, client_with_state, auth_headers):
        from unittest.mock import AsyncMock, MagicMock

        mock_engine = MagicMock()
        mock_engine.run = AsyncMock(side_effect=RuntimeError("boom"))
        client_with_state.app.state.engine = mock_engine

        resp = client_with_state.post(
            "/v1/execute", json={"task": "crash"}, headers=auth_headers
        )
        assert resp.status_code == 500


# ── JWT expiry (Gap 14) ────────────────────────────────────────────────────────

class TestJWTExpiry:

    def test_expired_jwt_returns_401(self, client):
        """A JWT with exp in the past must be rejected with 401."""
        import jwt as pyjwt
        from datetime import datetime, timezone, timedelta
        from nexus.config import config as nexus_config

        expired_payload = {
            "tenant_id": "demo",
            "role": "user",
            "exp": datetime.now(timezone.utc) - timedelta(seconds=10),
            "iat": datetime.now(timezone.utc) - timedelta(hours=1),
        }
        expired_token = pyjwt.encode(
            expired_payload,
            nexus_config.secret_key,
            algorithm=nexus_config.jwt_algorithm,
        )
        resp = client.get(
            "/v1/ledger", headers={"Authorization": f"Bearer {expired_token}"}
        )
        assert resp.status_code == 401

    def test_malformed_jwt_returns_401(self, client):
        """A JWT with wrong signature must be rejected with 401."""
        resp = client.get(
            "/v1/ledger",
            headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJ0ZW5hbnRfaWQiOiJ4In0.bad_sig"},
        )
        assert resp.status_code == 401
