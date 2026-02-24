"""Tests for the 4-gate anomaly engine.

Covers: each gate individually + combined, cold start, drift.
All tests use the real AnomalyEngine implementation — no stubs.
"""

import pytest
from datetime import datetime, timedelta, timezone

from nexus.types import PersonaContract, IntentDeclaration, GateVerdict, RiskLevel
from nexus.core.anomaly import AnomalyEngine
from nexus.config import NexusConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _config() -> NexusConfig:
    return NexusConfig(
        database_url="sqlite+aiosqlite:///test.db",
        redis_url="redis://localhost:6379/15",
        secret_key="test-secret",
        gate_intent_threshold=0.75,
        gate_drift_sigma=2.5,
    )


def _researcher() -> PersonaContract:
    return PersonaContract(
        name="researcher",
        description="Searches and retrieves information",
        allowed_tools=["knowledge_search", "web_search", "web_fetch", "file_read"],
        resource_scopes=["kb:*", "web:*", "file:read:*"],
        intent_patterns=["search for information", "find data about", "look up", "research"],
        risk_tolerance=RiskLevel.LOW,
        max_ttl_seconds=60,
    )


def _intent(
    tool: str = "knowledge_search",
    resources: list[str] = None,
    action: str = "search for information about NEXUS",
) -> IntentDeclaration:
    return IntentDeclaration(
        task_description="test task",
        planned_action=action,
        tool_name=tool,
        tool_params={"query": "test"},
        resource_targets=resources if resources is not None else ["kb:docs"],
        reasoning="test reasoning",
        confidence=0.9,
    )


class MockEmbeddingService:
    """Controllable stub for Gate 2 tests."""
    def __init__(self, similarity: float):
        self._similarity = similarity

    def similarities(self, query: str, candidates: list[str]) -> list[float]:
        return [self._similarity] * len(candidates)


# ── Gate 1: Scope ─────────────────────────────────────────────────────────────

class TestGate1Scope:

    def test_allowed_tool_passes(self, sample_personas):
        engine = AnomalyEngine(_config())
        researcher = sample_personas[0]  # researcher: allowed_tools includes knowledge_search
        intent = _intent(tool="knowledge_search", resources=["kb:docs"])
        result = engine._gate1_scope(researcher, intent)
        assert result.gate_name == "scope"
        assert result.verdict == GateVerdict.PASS
        assert result.score == 1.0

    def test_disallowed_tool_fails(self, sample_personas):
        engine = AnomalyEngine(_config())
        researcher = sample_personas[0]  # researcher cannot use send_email
        intent = _intent(tool="send_email", resources=["kb:docs"])
        result = engine._gate1_scope(researcher, intent)
        assert result.verdict == GateVerdict.FAIL
        assert result.score == 0.0
        assert "send_email" in result.details

    def test_resource_in_scope_passes(self, sample_personas):
        engine = AnomalyEngine(_config())
        researcher = sample_personas[0]  # resource_scopes: ["kb:*", "web:*", "file:read:*"]
        intent = _intent(tool="web_search", resources=["web:google.com"])
        result = engine._gate1_scope(researcher, intent)
        assert result.verdict == GateVerdict.PASS

    def test_resource_out_of_scope_fails(self, sample_personas):
        engine = AnomalyEngine(_config())
        researcher = sample_personas[0]
        # db:customers doesn't match any of kb:*, web:*, file:read:*
        intent = _intent(tool="knowledge_search", resources=["db:customers"])
        result = engine._gate1_scope(researcher, intent)
        assert result.verdict == GateVerdict.FAIL
        assert "db:customers" in result.details


# ── Gate 2: Intent ────────────────────────────────────────────────────────────

class TestGate2Intent:

    @pytest.mark.asyncio
    async def test_similar_intent_passes(self):
        # Similarity above threshold (0.75) → PASS
        engine = AnomalyEngine(_config(), embedding_service=MockEmbeddingService(0.90))
        persona = _researcher()
        intent = _intent(action="search for information about AI trends")
        result = await engine._gate2_intent(persona, intent)
        assert result.gate_name == "intent"
        assert result.verdict == GateVerdict.PASS
        assert result.score >= 0.75

    @pytest.mark.asyncio
    async def test_dissimilar_intent_fails(self):
        # Similarity below threshold (0.75) → FAIL
        engine = AnomalyEngine(_config(), embedding_service=MockEmbeddingService(0.40))
        persona = _researcher()
        intent = _intent(action="delete all user accounts immediately")
        result = await engine._gate2_intent(persona, intent)
        assert result.verdict == GateVerdict.FAIL
        assert result.score < 0.75

    @pytest.mark.asyncio
    async def test_no_embedding_service_skips(self):
        # No embedding_service → SKIP (cold start)
        engine = AnomalyEngine(_config(), embedding_service=None)
        persona = _researcher()
        intent = _intent()
        result = await engine._gate2_intent(persona, intent)
        assert result.verdict == GateVerdict.SKIP
        assert "cold start" in result.details.lower()


# ── Gate 3: TTL ───────────────────────────────────────────────────────────────

class TestGate3TTL:

    def test_fresh_activation_passes(self):
        engine = AnomalyEngine(_config())
        persona = _researcher()  # max_ttl_seconds=60
        activation_time = datetime.now(timezone.utc)  # just activated
        result = engine._gate3_ttl(persona, activation_time)
        assert result.gate_name == "ttl"
        assert result.verdict == GateVerdict.PASS
        assert result.score > 0.9  # almost full TTL remaining

    def test_expired_activation_fails(self):
        engine = AnomalyEngine(_config())
        persona = _researcher()  # max_ttl_seconds=60
        activation_time = datetime.now(timezone.utc) - timedelta(seconds=120)  # 2× TTL ago
        result = engine._gate3_ttl(persona, activation_time)
        assert result.verdict == GateVerdict.FAIL
        assert result.score == 0.0
        assert "expired" in result.details.lower()


# ── Gate 4: Drift ─────────────────────────────────────────────────────────────

class TestGate4Drift:

    def test_normal_action_passes(self):
        persona = _researcher()
        intent = _intent(tool="knowledge_search", resources=["kb:docs"])
        fp = AnomalyEngine.compute_fingerprint("knowledge_search", ["kb:docs"])
        # All 10 history entries use the same fingerprint → stdev=0 → sigma=0 → PASS
        store = {f"{persona.id}:fingerprints": [fp] * 10}
        engine = AnomalyEngine(_config(), fingerprint_store=store)
        result = engine._gate4_drift(persona, intent)
        assert result.gate_name == "drift"
        assert result.verdict == GateVerdict.PASS
        assert result.score == 0.0  # 0σ from baseline

    def test_anomalous_action_fails(self):
        persona = _researcher()
        # The tested action uses fp_test
        fp_test = AnomalyEngine.compute_fingerprint("knowledge_search", ["kb:docs"])
        # Build history: fp_test appears 100×, plus 9 other distinct fps each once
        history = [fp_test] * 100
        for i in range(9):
            other_fp = AnomalyEngine.compute_fingerprint(f"tool_{i}", [f"res:{i}"])
            history.append(other_fp)
        # freq: {fp_test: 100, other0: 1, ... other8: 1}
        # counts: [100, 1×9] → mean≈10.9, stdev≈31.3, sigma for fp_test ≈ 2.85 > 2.5 → FAIL
        store = {f"{persona.id}:fingerprints": history}
        engine = AnomalyEngine(_config(), fingerprint_store=store)
        intent = _intent(tool="knowledge_search", resources=["kb:docs"])
        result = engine._gate4_drift(persona, intent)
        assert result.verdict == GateVerdict.FAIL
        assert result.score > 2.5

    def test_insufficient_baseline_skips(self):
        persona = _researcher()
        fp = AnomalyEngine.compute_fingerprint("knowledge_search", ["kb:docs"])
        # Only 5 samples — below the 10-sample minimum
        store = {f"{persona.id}:fingerprints": [fp] * 5}
        engine = AnomalyEngine(_config(), fingerprint_store=store)
        intent = _intent(tool="knowledge_search", resources=["kb:docs"])
        result = engine._gate4_drift(persona, intent)
        assert result.verdict == GateVerdict.SKIP
        assert "5/10" in result.details


# ── Combined Gates ─────────────────────────────────────────────────────────────

class TestCombinedGates:

    @pytest.mark.asyncio
    async def test_all_pass(self):
        """All gates passing → overall PASS."""
        persona = _researcher()
        engine = AnomalyEngine(
            _config(),
            embedding_service=MockEmbeddingService(0.90),
        )
        activation_time = datetime.now(timezone.utc)
        intent = _intent(tool="knowledge_search", resources=["kb:docs"])
        result = await engine.check(persona, intent, activation_time)
        assert result.overall_verdict == GateVerdict.PASS
        assert len(result.gates) == 4

    @pytest.mark.asyncio
    async def test_one_fail_means_overall_fail(self):
        """Gate 1 FAIL (wrong tool) → overall verdict FAIL even if others pass."""
        persona = _researcher()
        engine = AnomalyEngine(
            _config(),
            embedding_service=MockEmbeddingService(0.90),
        )
        activation_time = datetime.now(timezone.utc)
        # send_email is NOT in researcher's allowed_tools → Gate 1 FAIL
        intent = _intent(tool="send_email", resources=["kb:docs"])
        result = await engine.check(persona, intent, activation_time)
        assert result.overall_verdict == GateVerdict.FAIL
        scope_gate = next(g for g in result.gates if g.gate_name == "scope")
        assert scope_gate.verdict == GateVerdict.FAIL

    @pytest.mark.asyncio
    async def test_skip_does_not_cause_fail(self):
        """Gate 2 SKIP (no embedding service) + Gate 4 SKIP (no baseline)
        must NOT cause overall FAIL — only FAILs count."""
        persona = _researcher()
        # No embedding service → Gate 2 SKIP
        # No fingerprint history → Gate 4 SKIP
        engine = AnomalyEngine(_config(), embedding_service=None)
        activation_time = datetime.now(timezone.utc)
        intent = _intent(tool="knowledge_search", resources=["kb:docs"])
        result = await engine.check(persona, intent, activation_time)
        assert result.overall_verdict == GateVerdict.PASS
        skipped = [g for g in result.gates if g.verdict == GateVerdict.SKIP]
        assert len(skipped) >= 2  # Gate 2 + Gate 4 both SKIP


# ── Gate 3: Exact boundary test ───────────────────────────────────────────────

class TestGate3TTLBoundary:

    def test_gate3_exact_boundary_fails(self):
        """Exactly at TTL boundary (remaining = 0.0) must FAIL."""
        engine = AnomalyEngine(_config())
        persona = _researcher()  # max_ttl_seconds=60
        # Activated exactly 60 seconds ago → remaining = 60 - 60 = 0.0 → FAIL
        activation_time = datetime.now(timezone.utc) - timedelta(seconds=60)
        result = engine._gate3_ttl(persona, activation_time)
        assert result.gate_name == "ttl"
        assert result.verdict == GateVerdict.FAIL
        assert result.score == 0.0


# ── Gate 1: Empty resource_targets design contract ────────────────────────────

class TestGate1EmptyTargets:

    def test_gate1_empty_targets_passes_when_tool_allowed(self):
        """Design contract: empty resource_targets = no resource scope check.

        If there are no resources to check, only the tool check applies.
        Allowed tool + no resources → PASS.
        """
        engine = AnomalyEngine(_config())
        persona = _researcher()  # allowed_tools includes knowledge_search
        intent = _intent(tool="knowledge_search", resources=[])
        result = engine._gate1_scope(persona, intent)
        assert result.verdict == GateVerdict.PASS

    def test_gate1_empty_targets_fails_when_tool_not_allowed(self):
        """Disallowed tool with empty resource_targets still FAILs (tool check runs)."""
        engine = AnomalyEngine(_config())
        persona = _researcher()
        intent = _intent(tool="send_email", resources=[])
        result = engine._gate1_scope(persona, intent)
        assert result.verdict == GateVerdict.FAIL
        assert "send_email" in result.details


# ── Gate 4: Async FingerprintCache interface ──────────────────────────────────

class TestGate4WithAsyncStore:
    """Gate 4 pre-fetches baseline asynchronously when store has get_baseline()."""

    async def test_gate4_uses_async_get_baseline_when_available(self):
        """When fingerprint_store has get_baseline(), it is awaited during check()."""
        from unittest.mock import AsyncMock

        # 12 identical fingerprints → freq={fp:12}, stdev=0 → sigma=0 → PASS
        fp = AnomalyEngine.compute_fingerprint("knowledge_search", ["kb:docs"])
        fingerprints = [fp] * 12

        mock_store = AsyncMock()
        mock_store.get_baseline = AsyncMock(return_value={
            "fingerprints": fingerprints,
            "sample_count": 12,
            "frequency_map": {},
        })

        engine = AnomalyEngine(_config(), fingerprint_store=mock_store)
        persona = _researcher()
        intent = _intent(tool="knowledge_search", resources=["kb:docs"])
        activation_time = datetime.now(timezone.utc) - timedelta(seconds=5)

        result = await engine.check(
            persona, intent, activation_time, tenant_id="tenant-1"
        )

        mock_store.get_baseline.assert_awaited_once_with("tenant-1", str(persona.id))
        # 12 samples → Gate 4 does NOT skip
        gate4 = next(g for g in result.gates if g.gate_name == "drift")
        assert gate4.verdict != GateVerdict.SKIP

    async def test_gate4_skips_when_get_baseline_returns_empty(self):
        """When get_baseline returns empty list, Gate 4 SKIPs (< 10 samples)."""
        from unittest.mock import AsyncMock

        mock_store = AsyncMock()
        mock_store.get_baseline = AsyncMock(return_value={
            "fingerprints": [],
            "sample_count": 0,
            "frequency_map": {},
        })

        engine = AnomalyEngine(_config(), fingerprint_store=mock_store)
        persona = _researcher()
        intent = _intent(tool="knowledge_search", resources=["kb:docs"])
        activation_time = datetime.now(timezone.utc) - timedelta(seconds=5)

        result = await engine.check(
            persona, intent, activation_time, tenant_id="tenant-1"
        )

        gate4 = next(g for g in result.gates if g.gate_name == "drift")
        assert gate4.verdict == GateVerdict.SKIP
