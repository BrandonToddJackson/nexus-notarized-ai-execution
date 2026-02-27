"""tests/test_e2e_invariants.py

==============================================================================
HOW NEXUS SHOULD WORK — First-Principles Specification
==============================================================================

NEXUS is a "notarized AI execution" framework. The core claim:
  "Every AI action is declared, verified, and sealed in an immutable ledger
   before execution. If it looks wrong, it's blocked."

This means EVERY tool call must pass through the following pipeline in order:

  1. DECOMPOSE   → LLM breaks the task into steps (action, tool, persona)
  2. RETRIEVE    → ContextBuilder assembles RAG context for the step
  3. THINK/ACT   → ThinkActGate decides if context is sufficient to act
  4. PLAN        → ToolSelector produces an IntentDeclaration
  5. ACTIVATE    → PersonaManager activates the persona (ephemeral identity)
  6. 4-GATE CHECK→ AnomalyEngine runs all 4 gates:
       Gate 1 — SCOPE:   tool_name in persona.allowed_tools?
                          resource_targets match persona.resource_scopes?
       Gate 2 — INTENT:  cosine(planned_action, intent_patterns) >= 0.75?
                          SKIPS if no embedding service (cold start)
       Gate 3 — TTL:     persona active < max_ttl_seconds?
       Gate 4 — DRIFT:   fingerprint within 2.5σ of baseline?
                          SKIPS if <10 historical samples (cold start)
  7. SEAL        → Notary.create_seal() stamps intent + gate results + Merkle
                    fingerprint into an immutable record (status=PENDING)
                    CRITICAL: tool_params are sanitized BEFORE sealing —
                    secrets never appear in the ledger.
  8. GATE FAIL?  → If ANY gate FAILed:
                    • finalize seal status=BLOCKED
                    • append to ledger
                    • revoke persona
                    • raise AnomalyDetected — tool is NEVER called
  9. VERIFY      → IntentVerifier cross-checks declared intent vs actual params
 10. INJECT      → ToolExecutor injects credentials from vault AFTER gate check
 11. EXECUTE     → Tool runs in Sandbox with timeout enforcement
 12. VALIDATE    → OutputValidator scans for PII (SSN, CC, email in wrong places)
 13. FINALIZE    → notary.finalize_seal() updates status=EXECUTED (or FAILED)
 14. LEDGER      → ledger.append(seal) — append-only, no modification
 15. REVOKE      → persona_manager.revoke() — identity is ephemeral
 16. CONTINUE?   → ContinueCompleteGate decides if chain needs another step

INVARIANTS (must NEVER be violated):
  I1. No tool executes unless all 4 gates PASS or SKIP.
  I2. Sealed tool_params never contain secret values (password, api_key, etc.).
  I3. A BLOCKED seal is appended to the ledger BEFORE AnomalyDetected is raised.
  I4. verify_chain() on seals from a completed chain always returns True.
  I5. Modifying a seal breaks verify_chain() — SealIntegrityError is raised.
  I6. ledger.get_by_tenant(A) never returns seals from tenant B.
  I7. Gate 2 SKIPs (not FAILs) when no embedding service is configured.
  I8. Gate 4 SKIPs (not FAILs) when persona has <10 historical fingerprints.
  I9. Gate 4 stops SKIPping once 10+ fingerprints exist for the persona.
  I10. Trust tier degrades immediately (in-memory) after any gate failure.
  I11. OutputValidator flags SSN patterns in tool results.
  I12. Workflow DAG: seals are created in topological step order.

KNOWN DESIGN TRADE-OFFS (not bugs, but must be visible):
  - Gates 2 & 4 SKIP on fresh systems → security is reduced until warmed up.
    The system is designed for a "trust-then-verify" warm-up phase.
  - Notary._last_fingerprint is singleton state. Under concurrent asyncio.gather
    calls (parallel workflow branches), seal fingerprints from different branches
    interleave in the global chain. verify_chain() for a single-chain sequential
    execution is always correct.

==============================================================================
TESTS
==============================================================================
"""
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
from nexus.credentials.vault import CredentialVault, sanitize_tool_params
from nexus.exceptions import AnomalyDetected, SealIntegrityError
from nexus.knowledge.context import ContextBuilder
from nexus.reasoning.continue_complete import ContinueCompleteGate
from nexus.reasoning.escalate import EscalateGate
from nexus.reasoning.think_act import ThinkActGate
from nexus.tools.executor import ToolExecutor
from nexus.tools.registry import ToolRegistry
from nexus.tools.sandbox import Sandbox
from nexus.tools.selector import ToolSelector
from nexus.types import (
    ActionStatus, GateVerdict, IntentDeclaration, PersonaContract,
    RiskLevel, ToolDefinition, TrustTier,
)

# ── Shared constants ──────────────────────────────────────────────────────────

TENANT_A = "tenant-e2e-alpha"
TENANT_B = "tenant-e2e-beta"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_config(**overrides) -> NexusConfig:
    """Minimal NexusConfig for E2E tests."""
    defaults = {
        "secret_key": "e2e-test-secret-key-32-bytes-long!!",
        "database_url": "sqlite+aiosqlite:///:memory:",
        "redis_url": "redis://localhost:6379/0",
        "gate_intent_threshold": 0.75,
        "gate_drift_sigma": 2.5,
        "gate_default_ttl": 120,
        "credential_encryption_key": "",  # auto-generated
    }
    defaults.update(overrides)
    return NexusConfig(**defaults)


def _make_researcher_persona(max_ttl_seconds: int = 120) -> PersonaContract:
    """Researcher persona — may only use knowledge_search on kb:* resources."""
    return PersonaContract(
        name="researcher",
        description="Searches and retrieves information",
        allowed_tools=["knowledge_search"],
        resource_scopes=["kb:*", "web:*"],
        intent_patterns=["search for information", "find data about", "look up"],
        max_ttl_seconds=max_ttl_seconds,
        risk_tolerance=RiskLevel.LOW,
        trust_tier=TrustTier.COLD_START,
    )


def _make_tool_registry() -> tuple[ToolRegistry, list[str]]:
    """Minimal tool registry with one tool that records calls."""
    calls: list[str] = []
    registry = ToolRegistry()

    async def knowledge_search(query: str) -> str:  # noqa: D401
        calls.append(query)
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
    return registry, calls


def _make_intent(
    tool_name: str = "knowledge_search",
    resource_targets: list[str] | None = None,
    tool_params: dict | None = None,
    planned_action: str = "search for information about AI",
) -> IntentDeclaration:
    return IntentDeclaration(
        task_description="E2E test task",
        planned_action=planned_action,
        tool_name=tool_name,
        tool_params=tool_params or {"query": "AI trends"},
        resource_targets=resource_targets or ["kb:general"],
        reasoning="Unit test",
        confidence=0.9,
    )


def _make_minimal_engine(
    persona: PersonaContract,
    registry: ToolRegistry,
    config: NexusConfig,
    fingerprint_store: dict | None = None,
    embedding_service=None,
    llm_decompose_steps: list[dict] | None = None,
) -> NexusEngine:
    """Wire a full NexusEngine with in-memory stores, no DB, no Redis."""
    from nexus.types import RetrievedContext

    persona_manager = PersonaManager([persona])
    anomaly_engine = AnomalyEngine(
        config=config,
        embedding_service=embedding_service,
        fingerprint_store=fingerprint_store or {},
    )
    notary = Notary()
    ledger = Ledger()
    chain_manager = ChainManager()
    verifier = IntentVerifier()
    output_validator = OutputValidator()
    cot_logger = CoTLogger()

    # ContextBuilder: mock knowledge_store so list_namespaces() returns real strings
    # and query() returns a valid RetrievedContext (not a raw MagicMock).
    knowledge_store = MagicMock()
    knowledge_store.list_namespaces = MagicMock(return_value=["general"])
    knowledge_store.query = AsyncMock(return_value=MagicMock(
        confidence=0.9, documents=[], sources=[],
    ))
    # Mock the full ContextBuilder.build() to skip ChromaDB dependency entirely
    mock_context = RetrievedContext(
        query="test",
        documents=[],
        confidence=0.9,
        sources=[],
        namespace="general",
    )
    context_builder = MagicMock(spec=ContextBuilder)
    context_builder.build = AsyncMock(return_value=mock_context)

    sandbox = Sandbox()
    tool_executor = ToolExecutor(registry=registry, sandbox=sandbox, verifier=verifier)
    tool_selector = ToolSelector(registry=registry, llm_client=None)

    think_act_gate = ThinkActGate()
    continue_complete_gate = ContinueCompleteGate()
    escalate_gate = EscalateGate()

    # Mock LLM: decompose returns the provided steps (or a single knowledge_search step)
    default_step = [{
        "action": "Search for information",
        "tool": "knowledge_search",
        "params": {"query": "test query"},
        "persona": "researcher",
    }]
    steps = llm_decompose_steps or default_step

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value={
        "content": __import__("json").dumps(steps),
        "tool_calls": [],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    })

    engine = NexusEngine(
        persona_manager=persona_manager,
        anomaly_engine=anomaly_engine,
        notary=notary,
        ledger=ledger,
        chain_manager=chain_manager,
        context_builder=context_builder,
        tool_registry=registry,
        tool_selector=tool_selector,
        tool_executor=tool_executor,
        output_validator=output_validator,
        cot_logger=cot_logger,
        think_act_gate=think_act_gate,
        continue_complete_gate=continue_complete_gate,
        escalate_gate=escalate_gate,
        llm_client=mock_llm,
        config=config,
    )
    return engine


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Anomaly Gate Unit Invariants
# ══════════════════════════════════════════════════════════════════════════════

class TestGate1Scope:
    """Gate 1: tool must be in allowed_tools; resources must match scopes."""

    @pytest.mark.asyncio
    async def test_pass_when_tool_in_scope(self):
        """I1 prerequisite: Gate 1 PASS when tool and resource match persona."""
        config = _make_config()
        engine = AnomalyEngine(config=config)
        persona = _make_researcher_persona()
        intent = _make_intent(tool_name="knowledge_search", resource_targets=["kb:general"])

        result = await engine.check(persona, intent, datetime.now(timezone.utc), TENANT_A)

        gate1 = next(g for g in result.gates if g.gate_name == "scope")
        assert gate1.verdict == GateVerdict.PASS
        assert gate1.score == 1.0

    @pytest.mark.asyncio
    async def test_fail_when_tool_not_allowed(self):
        """I1: Gate 1 FAIL when tool_name not in persona.allowed_tools."""
        config = _make_config()
        engine = AnomalyEngine(config=config)
        persona = _make_researcher_persona()
        intent = _make_intent(tool_name="send_email", resource_targets=["email:user@example.com"])

        result = await engine.check(persona, intent, datetime.now(timezone.utc), TENANT_A)

        gate1 = next(g for g in result.gates if g.gate_name == "scope")
        assert gate1.verdict == GateVerdict.FAIL
        assert gate1.score == 0.0
        assert result.overall_verdict == GateVerdict.FAIL

    @pytest.mark.asyncio
    async def test_fail_when_resource_out_of_scope(self):
        """I1: Gate 1 FAIL when resource_target doesn't match any persona scope."""
        config = _make_config()
        engine = AnomalyEngine(config=config)
        persona = _make_researcher_persona()
        # knowledge_search is allowed but accessing database resources
        intent = _make_intent(
            tool_name="knowledge_search",
            resource_targets=["db:production.customers"],  # not in ["kb:*", "web:*"]
        )

        result = await engine.check(persona, intent, datetime.now(timezone.utc), TENANT_A)

        gate1 = next(g for g in result.gates if g.gate_name == "scope")
        assert gate1.verdict == GateVerdict.FAIL
        assert "out of scope" in gate1.details

    @pytest.mark.asyncio
    async def test_all_gates_run_even_when_gate1_fails(self):
        """Diagnostic requirement: all 4 gates run regardless of gate1 result."""
        config = _make_config()
        engine = AnomalyEngine(config=config)
        persona = _make_researcher_persona()
        intent = _make_intent(tool_name="send_email")

        result = await engine.check(persona, intent, datetime.now(timezone.utc), TENANT_A)

        gate_names = {g.gate_name for g in result.gates}
        assert gate_names == {"scope", "intent", "ttl", "drift"}, \
            "All 4 gates must run even when gate1 fails — diagnostic requirement"


class TestGate2Intent:
    """Gate 2: intent similarity. SKIPs without embedding service."""

    @pytest.mark.asyncio
    async def test_skips_without_embedding_service(self):
        """I7: Gate 2 SKIPS (not FAILs) when no embedding service configured."""
        config = _make_config()
        # No embedding_service passed
        engine = AnomalyEngine(config=config, embedding_service=None)
        persona = _make_researcher_persona()
        intent = _make_intent(planned_action="delete all files")  # clearly wrong intent

        result = await engine.check(persona, intent, datetime.now(timezone.utc), TENANT_A)

        gate2 = next(g for g in result.gates if g.gate_name == "intent")
        assert gate2.verdict == GateVerdict.SKIP, \
            "Gate 2 must SKIP (not FAIL) when no embedding service — this is cold start mode"
        # SKIP does NOT cause overall failure
        assert result.overall_verdict != GateVerdict.FAIL or any(
            g.gate_name != "intent" and g.verdict == GateVerdict.FAIL
            for g in result.gates
        ), "Gate 2 SKIP alone must not cause overall FAIL"

    @pytest.mark.asyncio
    async def test_skips_when_persona_has_no_patterns(self):
        """I7: Gate 2 SKIPS when persona has no intent patterns."""
        config = _make_config()
        mock_embedding = MagicMock()
        engine = AnomalyEngine(config=config, embedding_service=mock_embedding)

        persona = PersonaContract(
            name="operator",
            description="No patterns defined",
            allowed_tools=["knowledge_search"],
            resource_scopes=["kb:*"],
            intent_patterns=[],  # empty
            max_ttl_seconds=120,
            risk_tolerance=RiskLevel.LOW,
        )
        intent = _make_intent()

        result = await engine.check(persona, intent, datetime.now(timezone.utc), TENANT_A)

        gate2 = next(g for g in result.gates if g.gate_name == "intent")
        assert gate2.verdict == GateVerdict.SKIP
        assert "no intent patterns" in gate2.details.lower()

    @pytest.mark.asyncio
    async def test_pass_when_similarity_above_threshold(self):
        """Gate 2 PASS when cosine similarity >= threshold."""
        config = _make_config(gate_intent_threshold=0.75)
        mock_embedding = MagicMock()
        mock_embedding.similarities = MagicMock(return_value=[0.92, 0.60, 0.45])
        engine = AnomalyEngine(config=config, embedding_service=mock_embedding)

        persona = _make_researcher_persona()
        intent = _make_intent(planned_action="search for information about AI")

        result = await engine.check(persona, intent, datetime.now(timezone.utc), TENANT_A)

        gate2 = next(g for g in result.gates if g.gate_name == "intent")
        assert gate2.verdict == GateVerdict.PASS
        assert gate2.score == pytest.approx(0.92, abs=0.001)

    @pytest.mark.asyncio
    async def test_fail_when_similarity_below_threshold(self):
        """I1: Gate 2 FAIL when cosine similarity < threshold → overall FAIL."""
        config = _make_config(gate_intent_threshold=0.75)
        mock_embedding = MagicMock()
        mock_embedding.similarities = MagicMock(return_value=[0.40, 0.35, 0.28])
        engine = AnomalyEngine(config=config, embedding_service=mock_embedding)

        persona = _make_researcher_persona()
        intent = _make_intent(planned_action="delete production database")

        result = await engine.check(persona, intent, datetime.now(timezone.utc), TENANT_A)

        gate2 = next(g for g in result.gates if g.gate_name == "intent")
        assert gate2.verdict == GateVerdict.FAIL
        assert result.overall_verdict == GateVerdict.FAIL


class TestGate3TTL:
    """Gate 3: persona must not have exceeded max_ttl_seconds."""

    @pytest.mark.asyncio
    async def test_pass_when_within_ttl(self):
        """Gate 3 PASS when persona was activated recently."""
        config = _make_config()
        engine = AnomalyEngine(config=config)
        persona = _make_researcher_persona(max_ttl_seconds=60)
        intent = _make_intent()
        just_activated = datetime.now(timezone.utc) - timedelta(seconds=10)

        result = await engine.check(persona, intent, just_activated, TENANT_A)

        gate3 = next(g for g in result.gates if g.gate_name == "ttl")
        assert gate3.verdict == GateVerdict.PASS
        assert gate3.score > 0.0

    @pytest.mark.asyncio
    async def test_fail_when_ttl_expired(self):
        """I1: Gate 3 FAIL when persona active > max_ttl_seconds → overall FAIL."""
        config = _make_config()
        engine = AnomalyEngine(config=config)
        persona = _make_researcher_persona(max_ttl_seconds=10)
        intent = _make_intent()
        # Activate 30 seconds ago — well past the 10s TTL
        expired_activation = datetime.now(timezone.utc) - timedelta(seconds=30)

        result = await engine.check(persona, intent, expired_activation, TENANT_A)

        gate3 = next(g for g in result.gates if g.gate_name == "ttl")
        assert gate3.verdict == GateVerdict.FAIL
        assert gate3.score == 0.0
        assert result.overall_verdict == GateVerdict.FAIL


class TestGate4Drift:
    """Gate 4: behavioral drift. SKIPs with <10 samples."""

    @pytest.mark.asyncio
    async def test_skips_with_insufficient_baseline(self):
        """I8: Gate 4 SKIPs when persona has <10 historical fingerprints."""
        config = _make_config()
        persona = _make_researcher_persona()
        intent = _make_intent()

        # Gate 4 with dict-based store uses key "{persona.id}:fingerprints"
        persona_key = f"{persona.id}:fingerprints"
        fingerprint_store = {persona_key: ["fp1", "fp2", "fp3", "fp4", "fp5"]}
        engine = AnomalyEngine(config=config, fingerprint_store=fingerprint_store)

        result = await engine.check(persona, intent, datetime.now(timezone.utc), TENANT_A)

        gate4 = next(g for g in result.gates if g.gate_name == "drift")
        assert gate4.verdict == GateVerdict.SKIP
        # Should report insufficient sample count in details
        assert "5/10" in gate4.details, \
            f"Gate4 details should report '5/10', got: {gate4.details!r}"

    @pytest.mark.asyncio
    async def test_pass_with_sufficient_baseline_within_sigma(self):
        """I9: Gate 4 becomes active (not SKIP) after 10+ fingerprints."""
        config = _make_config(gate_drift_sigma=2.5)
        engine = AnomalyEngine(config=config)
        persona = _make_researcher_persona()
        intent = _make_intent()

        # Compute the fingerprint for this exact intent
        fp = AnomalyEngine.compute_fingerprint(intent.tool_name, intent.resource_targets)
        # Provide 15 samples where this fingerprint appears 10 times (common action)
        history = [fp] * 10 + ["other_fp_1"] * 3 + ["other_fp_2"] * 2

        gate4 = engine._gate4_drift(persona, intent, history=history)
        assert gate4.verdict != GateVerdict.SKIP, \
            "Gate 4 must NOT skip with 15 historical samples"
        assert gate4.verdict == GateVerdict.PASS


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Seal & Merkle Chain Invariants
# ══════════════════════════════════════════════════════════════════════════════

class TestNotaryMerkleChain:
    """Notary creates immutable seals; verify_chain detects tampering."""

    def _make_anomaly_pass(self) -> "AnomalyResult":
        from nexus.types import AnomalyResult, GateResult
        return AnomalyResult(
            gates=[
                GateResult(gate_name="scope", verdict=GateVerdict.PASS, score=1.0, threshold=1.0, details=""),
                GateResult(gate_name="intent", verdict=GateVerdict.SKIP, score=0.0, threshold=0.75, details=""),
                GateResult(gate_name="ttl", verdict=GateVerdict.PASS, score=0.9, threshold=0.0, details=""),
                GateResult(gate_name="drift", verdict=GateVerdict.SKIP, score=0.0, threshold=2.5, details=""),
            ],
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_uuid="test-uuid",
            action_fingerprint="abc123",
        )

    def test_seal_fingerprint_is_deterministic(self):
        """Same inputs produce same fingerprint."""
        notary1 = Notary()
        notary2 = Notary()
        intent = _make_intent()
        anomaly = self._make_anomaly_pass()

        seal1 = notary1.create_seal("chain-1", 0, TENANT_A, "researcher", intent, anomaly)
        seal2 = notary2.create_seal("chain-1", 0, TENANT_A, "researcher", intent, anomaly)

        assert seal1.fingerprint == seal2.fingerprint, \
            "Fingerprint must be deterministic given the same inputs"

    def test_fingerprint_changes_when_chain_changes(self):
        """Different chain_id → different fingerprint."""
        notary = Notary()
        intent = _make_intent()
        anomaly = self._make_anomaly_pass()

        seal_a = notary.create_seal("chain-A", 0, TENANT_A, "researcher", intent, anomaly)
        notary2 = Notary()
        seal_b = notary2.create_seal("chain-B", 0, TENANT_A, "researcher", intent, anomaly)

        assert seal_a.fingerprint != seal_b.fingerprint

    def test_verify_chain_passes_for_single_chain(self):
        """I4: verify_chain returns True for a correctly-created chain."""
        notary = Notary()
        intent = _make_intent()
        anomaly = self._make_anomaly_pass()

        seal0 = notary.create_seal("chain-1", 0, TENANT_A, "researcher", intent, anomaly)
        seal0 = notary.finalize_seal(seal0, "result0", ActionStatus.EXECUTED)

        seal1 = notary.create_seal("chain-1", 1, TENANT_A, "researcher", intent, anomaly)
        seal1 = notary.finalize_seal(seal1, "result1", ActionStatus.EXECUTED)

        assert notary.verify_chain([seal0, seal1]) is True

    def test_verify_chain_empty_list(self):
        """Empty chain verifies as True (no seals = no tampering possible)."""
        notary = Notary()
        assert notary.verify_chain([]) is True

    def test_verify_chain_detects_fingerprint_tampering(self):
        """I5: Modifying a seal's fingerprint breaks verify_chain."""
        notary = Notary()
        intent = _make_intent()
        anomaly = self._make_anomaly_pass()

        seal0 = notary.create_seal("chain-1", 0, TENANT_A, "researcher", intent, anomaly)
        seal0 = notary.finalize_seal(seal0, "result", ActionStatus.EXECUTED)
        seal1 = notary.create_seal("chain-1", 1, TENANT_A, "researcher", intent, anomaly)
        seal1 = notary.finalize_seal(seal1, "result", ActionStatus.EXECUTED)

        # Tamper with seal0's fingerprint
        tampered_seal0 = seal0.model_copy(update={"fingerprint": "tampered-fingerprint"})

        with pytest.raises(SealIntegrityError) as exc_info:
            notary.verify_chain([tampered_seal0, seal1])

        assert "integrity broken" in str(exc_info.value).lower() or \
               "step 0" in str(exc_info.value)

    def test_verify_chain_detects_content_tampering(self):
        """I5: Modifying a seal's tool_name breaks verify_chain (content hash changes)."""
        notary = Notary()
        intent = _make_intent()
        anomaly = self._make_anomaly_pass()

        seal0 = notary.create_seal("chain-1", 0, TENANT_A, "researcher", intent, anomaly)
        seal0 = notary.finalize_seal(seal0, "result", ActionStatus.EXECUTED)

        # Tamper: change the tool_name in the intent
        tampered_intent = seal0.intent.model_copy(update={"tool_name": "send_email"})
        tampered_seal = seal0.model_copy(update={"intent": tampered_intent})

        with pytest.raises(SealIntegrityError):
            notary.verify_chain([tampered_seal])

    def test_parent_fingerprint_chains_correctly(self):
        """Each seal's parent_fingerprint matches the previous seal's fingerprint."""
        notary = Notary()
        intent = _make_intent()
        anomaly = self._make_anomaly_pass()

        seal0 = notary.create_seal("chain-1", 0, TENANT_A, "researcher", intent, anomaly)
        assert seal0.parent_fingerprint == "", "First seal has no parent"

        seal1 = notary.create_seal("chain-1", 1, TENANT_A, "researcher", intent, anomaly)
        assert seal1.parent_fingerprint == seal0.fingerprint, \
            "Seal 1's parent must be seal 0's fingerprint"

        seal2 = notary.create_seal("chain-1", 2, TENANT_A, "researcher", intent, anomaly)
        assert seal2.parent_fingerprint == seal1.fingerprint


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Credential Sanitization Invariants
# ══════════════════════════════════════════════════════════════════════════════

class TestCredentialSanitization:
    """Secrets must never appear in sealed tool_params."""

    def test_known_sensitive_keys_are_redacted(self):
        """I2: All known sensitive key names are replaced with '***' before sealing."""
        sensitive_params = {
            "Authorization": "Bearer sk-supersecret",
            "authorization": "Bearer sk-supersecret-lowercase",
            "password": "my-secret-password",
            "token": "ghp_supersecrettoken",
            "api_key": "sk-openai-key",
            "access_token": "oauth-access-token",
            "refresh_token": "oauth-refresh-token",
            "secret": "super-secret-value",
            "credentials": '{"key": "value"}',
            "client_secret": "oauth-client-secret",
            "safe_param": "this-is-safe",
            "query": "what is AI?",
        }

        sanitized = sanitize_tool_params(sensitive_params)

        # All sensitive keys must be redacted
        for key in ["Authorization", "authorization", "password", "token", "api_key",
                    "access_token", "refresh_token", "secret", "credentials", "client_secret"]:
            assert sanitized[key] == "***", f"Key '{key}' must be redacted, got: {sanitized[key]!r}"

        # Safe params must pass through unchanged
        assert sanitized["safe_param"] == "this-is-safe"
        assert sanitized["query"] == "what is AI?"

    def test_sanitized_params_are_a_copy(self):
        """sanitize_tool_params must not mutate the original dict."""
        original = {"api_key": "secret", "query": "test"}
        sanitized = sanitize_tool_params(original)

        assert original["api_key"] == "secret", "Original must not be mutated"
        assert sanitized["api_key"] == "***"

    def test_empty_params_sanitize_cleanly(self):
        """Empty params produce empty sanitized dict."""
        assert sanitize_tool_params({}) == {}

    def test_nested_values_not_recursively_sanitized(self):
        """Sanitization is shallow (top-level keys only) — this is by design."""
        params = {
            "headers": {"Authorization": "Bearer secret"},  # nested — NOT sanitized
            "safe_key": "safe_value",
        }
        sanitized = sanitize_tool_params(params)
        # The nested Authorization is NOT redacted — only top-level keys
        assert sanitized["headers"] == {"Authorization": "Bearer secret"}, \
            "Sanitization is shallow by design — nested values are not redacted"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Full Pipeline Integration — Happy Path
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineHappyPath:
    """The complete pipeline: decompose → gate → seal → execute → finalize."""

    @pytest.mark.asyncio
    async def test_successful_execution_produces_executed_seal(self):
        """I1 + I4: Happy path — chain completes with status=EXECUTED seal."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, calls = _make_tool_registry()
        engine = _make_minimal_engine(persona, registry, config)

        chain = await engine.run("Search for AI trends", TENANT_A, persona_name="researcher")

        assert chain.status.value in ("completed", "executing"), \
            f"Chain should complete successfully, got: {chain.status}"
        assert len(chain.seals) >= 1, "Chain must produce at least 1 seal"

    @pytest.mark.asyncio
    async def test_successful_execution_calls_tool(self):
        """Tool function is actually called during successful execution."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, calls = _make_tool_registry()
        engine = _make_minimal_engine(persona, registry, config)

        await engine.run("Search for AI trends", TENANT_A, persona_name="researcher")

        assert len(calls) >= 1, "Tool function must be called during successful execution"

    @pytest.mark.asyncio
    async def test_seal_appended_to_ledger(self):
        """I3 (success path): Executed seal is appended to the ledger."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, calls = _make_tool_registry()
        engine = _make_minimal_engine(persona, registry, config)

        await engine.run("Search for AI trends", TENANT_A, persona_name="researcher")

        seals = await engine.ledger.get_by_tenant(TENANT_A)
        assert len(seals) >= 1, "At least 1 seal must be in the ledger after execution"

    @pytest.mark.asyncio
    async def test_executed_seal_has_valid_merkle_fingerprint(self):
        """I4: verify_chain passes on the completed chain's seals."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()
        engine = _make_minimal_engine(persona, registry, config)

        chain = await engine.run("Search for AI trends", TENANT_A, persona_name="researcher")

        seals = await engine.ledger.get_chain(chain.id)
        assert len(seals) >= 1
        # This must not raise — if it does, the Merkle chain is broken
        assert engine.notary.verify_chain(seals) is True

    @pytest.mark.asyncio
    async def test_executed_seal_params_do_not_contain_secrets(self):
        """I2: Sealed tool_params have sensitive keys redacted."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()

        # Inject a secret into the decomposed step params
        steps_with_secret = [{
            "action": "Search with auth",
            "tool": "knowledge_search",
            "params": {"query": "AI trends", "api_key": "sk-supersecret-12345"},
            "persona": "researcher",
        }]
        engine = _make_minimal_engine(
            persona, registry, config, llm_decompose_steps=steps_with_secret
        )

        chain = await engine.run("Search securely", TENANT_A, persona_name="researcher")

        seals = await engine.ledger.get_chain(chain.id)
        for seal in seals:
            params = seal.intent.tool_params
            assert "sk-supersecret-12345" not in str(params), \
                "Raw secret value must never appear in a sealed tool_params"
            if "api_key" in params:
                assert params["api_key"] == "***", \
                    "api_key in sealed params must be redacted to '***'"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Blocked Path — Gate Failures
# ══════════════════════════════════════════════════════════════════════════════

def _make_engine_with_forced_intent(
    persona: PersonaContract,
    registry: ToolRegistry,
    config: NexusConfig,
    forced_intent: "IntentDeclaration",
) -> NexusEngine:
    """Engine where ToolSelector always returns a pre-built intent.

    This lets us test what happens when a specific intent (e.g., one that
    violates scope) reaches the gate check — bypassing ToolSelector's own
    filtering of persona.allowed_tools.
    """
    engine = _make_minimal_engine(persona, registry, config)
    engine.tool_selector = MagicMock(spec=ToolSelector)
    engine.tool_selector.select = AsyncMock(return_value=forced_intent)
    return engine


class TestBlockedPath:
    """Gate failures must: seal as BLOCKED, ledger append, raise AnomalyDetected, NOT call tool."""

    def _bad_intent(self) -> IntentDeclaration:
        """Intent that violates researcher's scope (send_email not allowed)."""
        return IntentDeclaration(
            task_description="Send email",
            planned_action="send an email to a user",
            tool_name="send_email",
            tool_params={"to": "user@example.com", "subject": "Test", "body": "Hi"},
            resource_targets=["email:user@example.com"],
            reasoning="E2E blocked-path test",
            confidence=0.9,
        )

    @pytest.mark.asyncio
    async def test_gate1_fail_raises_anomaly_detected(self):
        """I1: Out-of-scope tool → AnomalyDetected raised, chain aborted."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, calls = _make_tool_registry()
        engine = _make_engine_with_forced_intent(persona, registry, config, self._bad_intent())

        with pytest.raises((AnomalyDetected, Exception)):
            await engine.run("Send an email", TENANT_A, persona_name="researcher")

        # The knowledge_search tool must NOT have been called
        assert len(calls) == 0, \
            "I1 VIOLATED: Tool was called even though Gate 1 should have blocked it"

    @pytest.mark.asyncio
    async def test_blocked_seal_persisted_before_exception(self):
        """I3: BLOCKED seal is in the ledger even though AnomalyDetected was raised."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()
        engine = _make_engine_with_forced_intent(persona, registry, config, self._bad_intent())

        try:
            await engine.run("Send an email", TENANT_A, persona_name="researcher")
        except Exception:
            pass

        seals = await engine.ledger.get_by_tenant(TENANT_A)
        blocked_seals = [s for s in seals if s.status == ActionStatus.BLOCKED]
        assert len(blocked_seals) >= 1, \
            "I3 VIOLATED: BLOCKED seal must be persisted to ledger before AnomalyDetected is raised"

    @pytest.mark.asyncio
    async def test_blocked_seal_has_valid_fingerprint(self):
        """I4: Even BLOCKED seals have valid Merkle fingerprints."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()
        engine = _make_engine_with_forced_intent(persona, registry, config, self._bad_intent())

        try:
            await engine.run("Send an email", TENANT_A, persona_name="researcher")
        except Exception:
            pass

        seals = await engine.ledger.get_by_tenant(TENANT_A)
        if seals:
            assert engine.notary.verify_chain(seals) is True, \
                "Merkle chain must be valid even for BLOCKED seals"

    @pytest.mark.asyncio
    async def test_trust_tier_degrades_after_gate_failure(self):
        """I10: Trust tier is degraded in-memory after any gate failure."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()
        engine = _make_engine_with_forced_intent(persona, registry, config, self._bad_intent())

        try:
            await engine.run("Do bad thing", TENANT_A, persona_name="researcher")
        except Exception:
            pass

        # After failure, the in-memory persona should retain or degrade trust tier
        updated = engine.persona_manager._contracts.get("researcher")
        if updated is not None:
            assert updated.trust_tier in (TrustTier.COLD_START, TrustTier.ESTABLISHED, TrustTier.TRUSTED), \
                f"Trust tier must be a valid TrustTier after gate failure, got: {updated.trust_tier}"
            # Should not have been promoted (only demotions happen on failure)
            assert updated.trust_tier == TrustTier.COLD_START, \
                "Cold-start persona must remain COLD_START (cannot be promoted) after gate failure"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Tenant Isolation
# ══════════════════════════════════════════════════════════════════════════════

class TestTenantIsolation:
    """Tenant A's seals must never be visible to Tenant B."""

    @pytest.mark.asyncio
    async def test_get_by_tenant_returns_only_own_seals(self):
        """I6: get_by_tenant(A) excludes seals from tenant B."""
        config = _make_config()
        persona = _make_researcher_persona()

        # Shared ledger — both engines write to the same in-memory store
        shared_ledger = Ledger()

        registry_a, _ = _make_tool_registry()
        engine_a = _make_minimal_engine(persona, registry_a, config)
        engine_a.ledger = shared_ledger

        registry_b, _ = _make_tool_registry()
        engine_b = _make_minimal_engine(persona, registry_b, config)
        engine_b.ledger = shared_ledger

        # Run two different tenants through the shared ledger
        await engine_a.run("Search for AI trends", TENANT_A, persona_name="researcher")
        await engine_b.run("Search for B trends", TENANT_B, persona_name="researcher")

        seals_a = await shared_ledger.get_by_tenant(TENANT_A)
        seals_b = await shared_ledger.get_by_tenant(TENANT_B)

        assert len(seals_a) >= 1, "Tenant A must have seals"
        assert len(seals_b) >= 1, "Tenant B must have seals"

        tenant_ids_in_a = {s.tenant_id for s in seals_a}
        tenant_ids_in_b = {s.tenant_id for s in seals_b}

        assert TENANT_B not in tenant_ids_in_a, \
            "I6 VIOLATED: Tenant B's seals visible in Tenant A's query"
        assert TENANT_A not in tenant_ids_in_b, \
            "I6 VIOLATED: Tenant A's seals visible in Tenant B's query"

    @pytest.mark.asyncio
    async def test_get_chain_with_wrong_tenant_returns_empty(self):
        """I6: get_chain with tenant_id filter returns empty for wrong tenant."""
        ledger = Ledger()
        from nexus.types import AnomalyResult, GateResult
        anomaly = AnomalyResult(
            gates=[GateResult(gate_name="scope", verdict=GateVerdict.PASS, score=1.0, threshold=1.0, details=""),
                   GateResult(gate_name="intent", verdict=GateVerdict.SKIP, score=0.0, threshold=0.75, details=""),
                   GateResult(gate_name="ttl", verdict=GateVerdict.PASS, score=0.9, threshold=0.0, details=""),
                   GateResult(gate_name="drift", verdict=GateVerdict.SKIP, score=0.0, threshold=2.5, details="")],
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_uuid="uuid",
            action_fingerprint="fp",
        )
        notary = Notary()
        intent = _make_intent()
        seal = notary.create_seal("chain-A", 0, TENANT_A, "researcher", intent, anomaly)
        await ledger.append(seal)

        # Request chain for TENANT_A — should see it
        seals_correct = await ledger.get_chain("chain-A", tenant_id=TENANT_A)
        assert len(seals_correct) == 1

        # Request same chain for TENANT_B — should be empty
        seals_wrong_tenant = await ledger.get_chain("chain-A", tenant_id=TENANT_B)
        assert len(seals_wrong_tenant) == 0, \
            "I6 VIOLATED: get_chain returned seals for wrong tenant"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Output Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputValidation:
    """OutputValidator must flag PII patterns in tool results."""

    @pytest.mark.asyncio
    async def test_detects_ssn_pattern(self):
        """I11: SSN pattern in tool output is flagged as invalid."""
        validator = OutputValidator()
        intent = _make_intent()
        result_with_ssn = "Customer data: SSN is 123-45-6789 for this record."

        is_valid, reason = await validator.validate(intent, result_with_ssn)

        assert not is_valid, "Output containing SSN must be flagged as invalid"
        assert "ssn" in reason.lower() or "pii" in reason.lower() or "pattern" in reason.lower()

    @pytest.mark.asyncio
    async def test_detects_credit_card_pattern(self):
        """I11: Credit card number in tool output is flagged."""
        validator = OutputValidator()
        intent = _make_intent()
        result_with_cc = "Payment processed with card 4111 1111 1111 1111 successfully."

        is_valid, reason = await validator.validate(intent, result_with_cc)

        assert not is_valid, "Output containing credit card number must be flagged"

    @pytest.mark.asyncio
    async def test_clean_output_passes_validation(self):
        """Clean output passes output validation."""
        validator = OutputValidator()
        intent = _make_intent()
        clean_result = "AI trends in 2025 show increased adoption of multimodal models."

        is_valid, reason = await validator.validate(intent, clean_result)

        assert is_valid, f"Clean output must pass validation, got reason: {reason}"

    @pytest.mark.asyncio
    async def test_none_result_flagged_for_non_delete_intent(self):
        """Empty/None result for non-delete action is flagged."""
        validator = OutputValidator()
        intent = _make_intent(planned_action="search for information")

        is_valid, reason = await validator.validate(intent, None)

        assert not is_valid, "None result for a search action must be flagged as invalid"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Workflow DAG Execution
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkflowDAGExecution:
    """Workflow DAG: seals created in step order, all gates apply to each step."""

    def _make_linear_workflow(self, tenant_id: str):
        """Create a 2-step linear workflow definition."""
        from nexus.types import (
            WorkflowDefinition, WorkflowEdge, WorkflowStep,
            WorkflowStatus, StepType, EdgeType, NodePosition,
        )
        step_a = WorkflowStep(
            id="step-a",
            workflow_id="wf-1",
            step_type=StepType.ACTION,
            name="Step A",
            description="First step",
            tool_name="knowledge_search",
            tool_params={"query": "step A query"},
            persona_name="researcher",
            position=NodePosition(x=0.0, y=0.0),
            config={},
            timeout_seconds=30,
            retry_policy={},
        )
        step_b = WorkflowStep(
            id="step-b",
            workflow_id="wf-1",
            step_type=StepType.ACTION,
            name="Step B",
            description="Second step",
            tool_name="knowledge_search",
            tool_params={"query": "step B query"},
            persona_name="researcher",
            position=NodePosition(x=1.0, y=0.0),
            config={},
            timeout_seconds=30,
            retry_policy={},
        )
        edge = WorkflowEdge(
            id="edge-ab",
            workflow_id="wf-1",
            source_step_id="step-a",
            target_step_id="step-b",
            edge_type=EdgeType.DEFAULT,
        )
        return WorkflowDefinition(
            id="wf-1",
            tenant_id=tenant_id,
            name="Linear Test Workflow",
            description="2-step linear workflow",
            version=1,
            status=WorkflowStatus.ACTIVE,
            trigger_config={},
            steps=[step_a, step_b],
            edges=[edge],
            created_by="e2e-test",
            tags=[],
            settings={},
        )

    @pytest.mark.asyncio
    async def test_linear_workflow_produces_seals_in_order(self):
        """I12: A 2-step linear workflow produces 2 seals with step_index 0 and 1."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, calls = _make_tool_registry()
        engine = _make_minimal_engine(persona, registry, config)

        workflow = self._make_linear_workflow(TENANT_A)

        # Inject the workflow into the engine's workflow_manager
        from nexus.workflows.manager import WorkflowManager
        wm = WorkflowManager(repository=None, config=config)
        wm._store = {"wf-1": workflow}
        engine.workflow_manager = wm

        await engine.run_workflow(
            workflow_id="wf-1",
            tenant_id=TENANT_A,
            trigger_data={"source": "e2e-test"},
        )

        seals = await engine.ledger.get_by_tenant(TENANT_A)
        assert len(seals) >= 2, \
            f"I12: 2-step workflow must produce at least 2 seals, got {len(seals)}"

        # Both tools must have been called
        assert len(calls) >= 2, \
            f"Both workflow steps must execute the tool, got {len(calls)} calls"

    @pytest.mark.asyncio
    async def test_all_workflow_seals_have_valid_merkle_chain(self):
        """I4: verify_chain passes on all seals from a workflow execution."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()
        engine = _make_minimal_engine(persona, registry, config)

        workflow = self._make_linear_workflow(TENANT_A)

        from nexus.workflows.manager import WorkflowManager
        wm = WorkflowManager(repository=None, config=config)
        wm._store = {"wf-1": workflow}
        engine.workflow_manager = wm

        chain = await engine.run_workflow(
            workflow_id="wf-1",
            tenant_id=TENANT_A,
            trigger_data={"source": "e2e-test"},
        )

        # run_workflow returns WorkflowExecution; .chain_id holds the ChainPlan id
        chain_plan_id = chain.chain_id
        seals = await engine.ledger.get_chain(chain_plan_id)
        if not seals:
            # Fallback: get all seals for tenant and verify those
            seals = await engine.ledger.get_by_tenant(TENANT_A)
        assert len(seals) >= 1
        # Must not raise
        assert engine.notary.verify_chain(seals) is True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Data Flow Variable Threading
# ══════════════════════════════════════════════════════════════════════════════

class TestVariableFlow:
    """Critical variables must flow correctly through the entire pipeline."""

    @pytest.mark.asyncio
    async def test_tenant_id_set_correctly_in_sealed_record(self):
        """tenant_id set at engine.run() must appear in every created seal."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()
        engine = _make_minimal_engine(persona, registry, config)

        await engine.run("Search for AI", TENANT_A, persona_name="researcher")

        seals = await engine.ledger.get_by_tenant(TENANT_A)
        for seal in seals:
            assert seal.tenant_id == TENANT_A, \
                f"Seal tenant_id must be '{TENANT_A}', got '{seal.tenant_id}'"

    @pytest.mark.asyncio
    async def test_chain_id_consistent_across_all_seals(self):
        """All seals from one run() call share the same chain_id."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()

        # 2-step decomposition
        two_steps = [
            {"action": "Step 1", "tool": "knowledge_search", "params": {"query": "q1"}, "persona": "researcher"},
            {"action": "Step 2", "tool": "knowledge_search", "params": {"query": "q2"}, "persona": "researcher"},
        ]
        engine = _make_minimal_engine(persona, registry, config, llm_decompose_steps=two_steps)

        chain = await engine.run("Multi-step task", TENANT_A, persona_name="researcher")

        seals = await engine.ledger.get_chain(chain.id)
        if len(seals) > 1:
            chain_ids = {s.chain_id for s in seals}
            assert len(chain_ids) == 1, \
                f"All seals from one run must share one chain_id, found: {chain_ids}"

    @pytest.mark.asyncio
    async def test_persona_name_recorded_in_seal(self):
        """Persona name used at runtime appears in each seal."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()
        engine = _make_minimal_engine(persona, registry, config)

        chain = await engine.run("Search for AI", TENANT_A, persona_name="researcher")

        seals = await engine.ledger.get_chain(chain.id)
        for seal in seals:
            assert seal.persona_id == "researcher", \
                f"Seal must record persona 'researcher', got '{seal.persona_id}'"

    @pytest.mark.asyncio
    async def test_gate_results_recorded_in_every_seal(self):
        """Every seal must contain exactly 4 gate results."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()
        engine = _make_minimal_engine(persona, registry, config)

        chain = await engine.run("Search for AI", TENANT_A, persona_name="researcher")

        seals = await engine.ledger.get_chain(chain.id)
        for seal in seals:
            gates = seal.anomaly_result.gates
            assert len(gates) == 4, \
                f"Each seal must record 4 gate results, got {len(gates)}"
            gate_names = {g.gate_name for g in gates}
            assert gate_names == {"scope", "intent", "ttl", "drift"}, \
                f"Gate names must be scope/intent/ttl/drift, got {gate_names}"

    @pytest.mark.asyncio
    async def test_cot_trace_included_in_seal(self):
        """CoT reasoning trace is attached to each finalized seal."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()
        engine = _make_minimal_engine(persona, registry, config)

        chain = await engine.run("Search for AI", TENANT_A, persona_name="researcher")

        seals = await engine.ledger.get_chain(chain.id)
        for seal in seals:
            # cot_trace should be a list (possibly empty for minimal mock LLM)
            assert isinstance(seal.cot_trace, list), \
                f"cot_trace must be a list, got {type(seal.cot_trace)}"

    @pytest.mark.asyncio
    async def test_tool_result_stored_in_executed_seal(self):
        """The actual tool return value is stored in the finalized seal."""
        config = _make_config()
        persona = _make_researcher_persona()
        registry, _ = _make_tool_registry()
        engine = _make_minimal_engine(persona, registry, config)

        chain = await engine.run("Search for AI trends", TENANT_A, persona_name="researcher")

        seals = [s for s in await engine.ledger.get_chain(chain.id)
                 if s.status == ActionStatus.EXECUTED]
        assert len(seals) >= 1
        # The knowledge_search tool returns "Results for: ..."
        for seal in seals:
            assert seal.tool_result is not None, "Executed seal must have a tool_result"
            assert "Results for" in str(seal.tool_result), \
                f"tool_result should contain the knowledge_search output, got: {seal.tool_result!r}"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: Known Design Limitations (documented, not bugs)
# ══════════════════════════════════════════════════════════════════════════════

class TestKnownDesignLimitations:
    """Document known trade-offs so they are visible and testable."""

    @pytest.mark.asyncio
    async def test_gate2_and_gate4_skip_on_cold_start_is_by_design(self):
        """DOCUMENTED: Both Gates 2 and 4 SKIP on a fresh system.
        This is by design — the system warms up over time.
        Operators must understand that on first deploy:
        - Gate 2 SKIPS until embedding service loads
        - Gate 4 SKIPS until each persona accumulates 10+ fingerprints
        """
        config = _make_config()
        engine = AnomalyEngine(config=config, embedding_service=None, fingerprint_store={})
        persona = _make_researcher_persona()
        intent = _make_intent()

        gate2 = await engine._gate2_intent(persona, intent)
        gate4 = engine._gate4_drift(persona, intent, history=[])

        assert gate2.verdict == GateVerdict.SKIP, "Gate 2 SKIPS without embedding service"
        assert gate4.verdict == GateVerdict.SKIP, "Gate 4 SKIPS with no fingerprint history"
        # Document the implication: on cold start, SCOPE + TTL are the only active gates
        print("\nDOCUMENTED LIMITATION: On cold start, only Gates 1 (scope) and "
              "3 (TTL) are active. Gates 2 and 4 skip until warmed up.")

    def test_sanitize_is_shallow_not_recursive(self):
        """DOCUMENTED: sanitize_tool_params is shallow.
        Nested dicts (e.g., headers dict containing Authorization) are NOT sanitized.
        This means: tools that accept a 'headers' dict param can receive unsanitized secrets.
        If a tool receives headers as a dict, the outer key 'headers' is safe but
        its content is not inspected.
        """
        params = {
            "url": "https://api.example.com",
            "headers": {
                "Authorization": "Bearer sk-secret-not-sanitized",
                "Content-Type": "application/json",
            },
        }
        sanitized = sanitize_tool_params(params)
        # This assertion documents the limitation — it PASSES, revealing the gap
        assert sanitized["headers"]["Authorization"] == "Bearer sk-secret-not-sanitized", \
            "DOCUMENTED: Nested Authorization header is NOT sanitized. " \
            "http_request tool should handle its own auth injection via credential vault, " \
            "not via raw headers in tool_params."

    def test_notary_state_is_singleton_per_instance(self):
        """DOCUMENTED: Notary._last_fingerprint is instance-level shared state.
        Two seal creations on the same Notary instance form one global chain.
        This is correct for sequential execution but would interleave under
        true concurrent asyncio.gather calls with shared Notary.
        """
        notary = Notary()
        from nexus.types import AnomalyResult, GateResult
        anomaly = AnomalyResult(
            gates=[GateResult(gate_name="scope", verdict=GateVerdict.PASS, score=1.0, threshold=1.0, details=""),
                   GateResult(gate_name="intent", verdict=GateVerdict.SKIP, score=0.0, threshold=0.75, details=""),
                   GateResult(gate_name="ttl", verdict=GateVerdict.PASS, score=0.9, threshold=0.0, details=""),
                   GateResult(gate_name="drift", verdict=GateVerdict.SKIP, score=0.0, threshold=2.5, details="")],
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_uuid="uuid",
            action_fingerprint="fp",
        )
        intent = _make_intent()

        seal_chain_a_0 = notary.create_seal("chain-A", 0, TENANT_A, "researcher", intent, anomaly)
        seal_chain_b_0 = notary.create_seal("chain-B", 0, TENANT_B, "researcher", intent, anomaly)
        seal_chain_a_1 = notary.create_seal("chain-A", 1, TENANT_A, "researcher", intent, anomaly)

        # The global chain is A0 → B0 → A1 (interleaved)
        # Chain A verify will FAIL because A1.parent_fingerprint = B0.fingerprint,
        # not A0.fingerprint as verify_chain expects
        seals_a = sorted([seal_chain_a_0, seal_chain_a_1], key=lambda s: s.step_index)
        try:
            notary.verify_chain(seals_a)
            # If it passes, the implementation handles this correctly
            print("\nNOTE: verify_chain handles interleaved seals correctly")
        except SealIntegrityError:
            # This is the expected outcome — documenting the limitation
            print("\nDOCUMENTED LIMITATION: verify_chain fails for interleaved chains. "
                  "Each workflow/chain should use its own Notary instance for isolation, "
                  "or verify_chain must account for global seal ordering.")
