"""Tests verifying persona_id semantic consistency across the NEXUS pipeline.

Key invariants:
- Seal.persona_id == persona NAME (e.g. "researcher") — used for deterministic Merkle fingerprinting
- AnomalyResult.persona_uuid == persona UUID — UUID of the PersonaContract instance
- These two values are always different (name != UUID)
- Merkle fingerprint is stable regardless of persona UUID regeneration (UUID changes on restart)
"""

import uuid
import pytest
from datetime import datetime, timezone

from nexus.types import (
    AnomalyResult, GateResult, GateVerdict, RiskLevel,
    IntentDeclaration, PersonaContract, Seal, ActionStatus,
)
from nexus.core.anomaly import AnomalyEngine
from nexus.core.notary import Notary
from nexus.config import NexusConfig


def _make_gate(name: str, verdict: GateVerdict = GateVerdict.PASS) -> GateResult:
    return GateResult(
        gate_name=name, verdict=verdict, score=1.0, threshold=1.0, details="ok"
    )


def _make_intent(tool_name: str = "knowledge_search") -> IntentDeclaration:
    return IntentDeclaration(
        task_description="search for nexus docs",
        planned_action="search for information about nexus",
        tool_name=tool_name,
        tool_params={"query": "nexus"},
        resource_targets=["kb:*"],
        reasoning="user asked to search",
        confidence=0.9,
    )


def _make_persona(name: str = "researcher") -> PersonaContract:
    return PersonaContract(
        name=name,
        description="Research persona",
        allowed_tools=["knowledge_search", "web_search"],
        resource_scopes=["kb:*", "web:*"],
        intent_patterns=["search for information", "find documentation"],
        max_ttl_seconds=300,
    )


class TestAnomalyResultPersonaUUID:
    """AnomalyResult.persona_uuid must carry the UUID, not the name."""

    @pytest.mark.asyncio
    async def test_anomaly_result_persona_uuid_is_persona_id_value(self):
        """AnomalyResult.persona_uuid must equal persona.id (UUID)."""
        persona = _make_persona()
        engine = AnomalyEngine(config=NexusConfig())
        intent = _make_intent()
        result = await engine.check(
            persona=persona,
            intent=intent,
            activation_time=datetime.now(timezone.utc),
        )
        assert result.persona_uuid == persona.id

    @pytest.mark.asyncio
    async def test_anomaly_result_persona_uuid_is_valid_uuid4(self):
        """AnomalyResult.persona_uuid must parse as a valid UUID4."""
        persona = _make_persona()
        engine = AnomalyEngine(config=NexusConfig())
        intent = _make_intent()
        result = await engine.check(
            persona=persona,
            intent=intent,
            activation_time=datetime.now(timezone.utc),
        )
        # Must not raise ValueError
        parsed = uuid.UUID(result.persona_uuid)
        assert parsed.version == 4

    def test_anomaly_result_direct_construction_uses_persona_uuid_field(self):
        """Direct AnomalyResult construction must use persona_uuid, not persona_id."""
        anomaly = AnomalyResult(
            gates=[_make_gate(n) for n in ("scope", "intent", "ttl", "drift")],
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_uuid=str(uuid.uuid4()),
            action_fingerprint="abc123",
        )
        assert hasattr(anomaly, "persona_uuid")
        assert not hasattr(anomaly, "persona_id")


class TestSealPersonaIdIsName:
    """Seal.persona_id must be the persona NAME, not UUID."""

    def test_seal_persona_id_is_name(self):
        """Seal.persona_id must equal the persona name string."""
        notary = Notary()
        anomaly = AnomalyResult(
            gates=[_make_gate(n) for n in ("scope", "intent", "ttl", "drift")],
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_uuid=str(uuid.uuid4()),
            action_fingerprint="fp123",
        )
        seal = notary.create_seal(
            chain_id="chain-1",
            step_index=0,
            tenant_id="tenant-1",
            persona_id="researcher",  # passes the NAME
            intent=_make_intent(),
            anomaly_result=anomaly,
        )
        assert seal.persona_id == "researcher"
        assert seal.persona_id != seal.anomaly_result.persona_uuid


class TestPersonaIdAndUUIDAreDifferent:
    """Seal.persona_id (name) and AnomalyResult.persona_uuid (UUID) must differ."""

    @pytest.mark.asyncio
    async def test_name_and_uuid_are_different(self):
        """persona_id (name) != persona_uuid (UUID) in a real engine execution."""
        persona = _make_persona("researcher")
        engine = AnomalyEngine(config=NexusConfig())
        intent = _make_intent()
        result = await engine.check(
            persona=persona,
            intent=intent,
            activation_time=datetime.now(timezone.utc),
        )
        notary = Notary()
        seal = notary.create_seal(
            chain_id="chain-1",
            step_index=0,
            tenant_id="tenant-1",
            persona_id=persona.name,
            intent=intent,
            anomaly_result=result,
        )
        # Name and UUID are fundamentally different values
        assert seal.persona_id == "researcher"
        assert seal.persona_id != seal.anomaly_result.persona_uuid
        # UUID must be parseable, name is not a UUID
        uuid.UUID(seal.anomaly_result.persona_uuid)
        with pytest.raises(ValueError):
            uuid.UUID(seal.persona_id)


class TestMerkleChainStability:
    """Merkle fingerprint must be stable regardless of persona UUID regeneration."""

    def test_fingerprint_stable_across_persona_uuid_regeneration(self):
        """Two PersonaContracts with same name but different UUIDs produce identical fingerprints.

        The Merkle chain uses Seal.persona_id (name), NOT persona_uuid — so UUID
        regeneration on restart doesn't break chain continuity.
        """
        notary_a = Notary()
        notary_b = Notary()

        # Two personas with same name but different auto-generated UUIDs
        persona_a = _make_persona("researcher")
        persona_b = _make_persona("researcher")
        assert persona_a.id != persona_b.id  # different UUIDs

        anomaly_a = AnomalyResult(
            gates=[_make_gate(n) for n in ("scope", "intent", "ttl", "drift")],
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_uuid=persona_a.id,
            action_fingerprint="fp",
        )
        anomaly_b = AnomalyResult(
            gates=[_make_gate(n) for n in ("scope", "intent", "ttl", "drift")],
            overall_verdict=GateVerdict.PASS,
            risk_level=RiskLevel.LOW,
            persona_uuid=persona_b.id,
            action_fingerprint="fp",
        )

        intent = _make_intent()
        seal_a = notary_a.create_seal(
            chain_id="chain-stable",
            step_index=0,
            tenant_id="tenant-1",
            persona_id="researcher",  # always the name
            intent=intent,
            anomaly_result=anomaly_a,
        )
        seal_b = notary_b.create_seal(
            chain_id="chain-stable",
            step_index=0,
            tenant_id="tenant-1",
            persona_id="researcher",  # same name → same fingerprint
            intent=intent,
            anomaly_result=anomaly_b,
        )

        # Fingerprints must match despite different persona UUIDs
        assert seal_a.fingerprint == seal_b.fingerprint
