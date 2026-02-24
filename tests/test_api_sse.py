"""Tests for POST /v1/execute/stream — SSE real-time gate progression.

Covers Gap 1: chain_started, gate_result (×4), seal_created, step_completed,
chain_completed events emitted; error event on engine failure.
"""

import json
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from nexus.api.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_sse(text: str) -> list[dict]:
    """Parse raw SSE text into list of {'type': str, 'data': dict} dicts."""
    events = []
    current: dict = {}
    for line in text.split("\n"):
        if line.startswith("event: "):
            current["type"] = line[7:].strip()
        elif line.startswith("data: "):
            raw = line[6:].strip()
            try:
                current["data"] = json.loads(raw)
            except json.JSONDecodeError:
                current["data"] = raw
        elif line == "" and current:
            if "type" in current:
                events.append(current.copy())
            current = {}
    # flush any trailing event
    if current and "type" in current:
        events.append(current)
    return events


def _make_mock_engine(callback_sequence: list[tuple[str, dict]]) -> MagicMock:
    """Build a mock engine whose run() fires the given callback sequence."""
    mock_chain = MagicMock()
    mock_chain.id = "sse-test-chain-1"
    mock_chain.status = MagicMock()
    mock_chain.status.value = "completed"
    mock_chain.seals = []

    async def mock_run(task, tenant_id, persona_name=None, callbacks=None):
        cbs = callbacks or []
        for event, data in callback_sequence:
            for cb in cbs:
                await cb(event, data)
        return mock_chain

    mock_engine = MagicMock()
    mock_engine.run = mock_run
    return mock_engine


@pytest.fixture(scope="module")
def auth_headers():
    """Obtain a valid JWT for the demo tenant once per module."""
    client = TestClient(app)
    resp = client.post("/v1/auth/token", json={"api_key": "nxs_demo_key_12345"})
    if resp.status_code != 200:
        pytest.skip("Auth endpoint unavailable — cannot obtain JWT for SSE tests")
    token = resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSSEStreaming:

    def _full_sequence(self, chain_id: str = "sse-chain-1") -> list[tuple[str, dict]]:
        """A complete successful chain event sequence."""
        gates = [
            {"gate_name": "scope", "verdict": "pass", "score": 1.0, "threshold": 1.0, "details": "ok"},
            {"gate_name": "intent", "verdict": "skip", "score": 0.0, "threshold": 0.75, "details": "cold start"},
            {"gate_name": "ttl", "verdict": "pass", "score": 0.9, "threshold": 0.0, "details": "ok"},
            {"gate_name": "drift", "verdict": "skip", "score": 0.0, "threshold": 2.5, "details": "no baseline"},
        ]
        return [
            ("chain_created", {"chain_id": chain_id, "steps": 1, "task": "test task"}),
            ("step_started", {"chain_id": chain_id, "step_index": 0, "step": {"tool": "knowledge_search"}, "persona": "researcher"}),
            ("anomaly_checked", {"chain_id": chain_id, "step_index": 0, "verdict": "pass", "gates": gates}),
            ("step_completed", {"chain_id": chain_id, "step_index": 0, "seal_id": "seal-abc", "status": "executed", "tool": "knowledge_search", "result": "found it"}),
            ("chain_completed", {"chain_id": chain_id, "status": "completed"}),
        ]

    def test_chain_started_event_emitted(self, auth_headers):
        """chain_created callback → 'chain_started' SSE event."""
        app.state.engine = _make_mock_engine(self._full_sequence())

        with TestClient(app).stream(
            "POST", "/v1/execute/stream", json={"task": "test task"}, headers=auth_headers
        ) as resp:
            text = resp.read().decode()

        events = _parse_sse(text)
        assert any(e["type"] == "chain_started" for e in events), (
            f"Expected 'chain_started' in events, got: {[e['type'] for e in events]}"
        )

    def test_gate_result_events_emitted_per_gate(self, auth_headers):
        """anomaly_checked with 4 gates → 4 gate_result SSE events."""
        app.state.engine = _make_mock_engine(self._full_sequence())

        with TestClient(app).stream(
            "POST", "/v1/execute/stream", json={"task": "test task"}, headers=auth_headers
        ) as resp:
            text = resp.read().decode()

        events = _parse_sse(text)
        gate_events = [e for e in events if e["type"] == "gate_result"]
        assert len(gate_events) == 4, (
            f"Expected 4 gate_result events, got {len(gate_events)}. All events: {[e['type'] for e in events]}"
        )

    def test_gate_result_has_required_fields(self, auth_headers):
        """Each gate_result event must have step, gate, verdict, score."""
        app.state.engine = _make_mock_engine(self._full_sequence())

        with TestClient(app).stream(
            "POST", "/v1/execute/stream", json={"task": "test task"}, headers=auth_headers
        ) as resp:
            text = resp.read().decode()

        events = _parse_sse(text)
        gate_events = [e for e in events if e["type"] == "gate_result"]
        for gate_event in gate_events:
            data = gate_event["data"]
            assert "step" in data
            assert "gate" in data
            assert "verdict" in data
            assert "score" in data

    def test_chain_completed_event_emitted(self, auth_headers):
        """chain_completed callback → 'chain_completed' SSE event."""
        app.state.engine = _make_mock_engine(self._full_sequence())

        with TestClient(app).stream(
            "POST", "/v1/execute/stream", json={"task": "test task"}, headers=auth_headers
        ) as resp:
            text = resp.read().decode()

        events = _parse_sse(text)
        assert any(e["type"] == "chain_completed" for e in events)

    def test_error_during_engine_emits_error_event(self, auth_headers):
        """AnomalyDetected exception from engine → 'error' SSE event."""
        from nexus.exceptions import AnomalyDetected

        exc = AnomalyDetected("Gate 1 failed", gate_results=[], chain_id="err-chain")

        async def failing_run(task, tenant_id, persona_name=None, callbacks=None):
            raise exc

        mock_engine = MagicMock()
        mock_engine.run = failing_run
        app.state.engine = mock_engine

        with TestClient(app).stream(
            "POST", "/v1/execute/stream", json={"task": "test task"}, headers=auth_headers
        ) as resp:
            text = resp.read().decode()

        events = _parse_sse(text)
        assert any(e["type"] == "error" for e in events)

    def test_sse_response_has_correct_content_type(self, auth_headers):
        """SSE response must use text/event-stream content type."""
        app.state.engine = _make_mock_engine(self._full_sequence())

        with TestClient(app).stream(
            "POST", "/v1/execute/stream", json={"task": "test task"}, headers=auth_headers
        ) as resp:
            content_type = resp.headers.get("content-type", "")

        assert "text/event-stream" in content_type

    def test_seal_created_event_emitted(self, auth_headers):
        """step_completed callback → 'seal_created' SSE event before step_completed."""
        app.state.engine = _make_mock_engine(self._full_sequence())

        with TestClient(app).stream(
            "POST", "/v1/execute/stream", json={"task": "test task"}, headers=auth_headers
        ) as resp:
            text = resp.read().decode()

        events = _parse_sse(text)
        assert any(e["type"] == "seal_created" for e in events)
