"""4-gate anomaly detection engine. The security core of NEXUS.

Gate 1 — SCOPE CHECK: Is tool in persona's allowed_tools? Are resources in scope?
Gate 2 — INTENT SIMILARITY: Cosine similarity of declared intent vs persona patterns.
Gate 3 — TTL CHECK: Has persona been active too long?
Gate 4 — BEHAVIORAL DRIFT: Is this action's fingerprint within N sigma of baseline?

Implementation notes:
- Gate 2 uses embedding_service which may be None at startup. If None, gate returns SKIP.
- Gate 4 needs historical fingerprints. If <10 samples, SKIP.
- AnomalyResult.overall_verdict = FAIL if ANY gate verdict is FAIL. SKIP gates don't cause failure.
- Fingerprint for drift: hashlib.sha256(f"{tool_name}:{sorted_targets}:{intent_category}".encode()).hexdigest()[:16]
- ALL 4 gates run even if Gate 1 fails (for diagnostics).
"""

import hashlib
from datetime import datetime, timezone
from typing import Optional

from nexus.types import (
    PersonaContract, IntentDeclaration, AnomalyResult, GateResult,
    GateVerdict, RiskLevel,
)
from nexus.config import NexusConfig


class AnomalyEngine:
    """4-gate anomaly detection. The security core of NEXUS."""

    def __init__(self, config: NexusConfig, embedding_service=None, fingerprint_store=None):
        """
        Args:
            config: Gate thresholds from NexusConfig
            embedding_service: For Gate 2 intent matching (injected from Phase 6).
                               Can be None for cold start / testing.
            fingerprint_store: Dict or Redis-backed store for Gate 4 drift baselines.
                               Can be None for cold start.
        """
        self.config = config
        self.embedding_service = embedding_service
        self.fingerprint_store = fingerprint_store or {}

    async def check(
        self,
        persona: PersonaContract,
        intent: IntentDeclaration,
        activation_time: datetime,
        tenant_id: str = "",
    ) -> AnomalyResult:
        """Run all 4 gates. Returns AnomalyResult.

        Gates run IN ORDER but ALL gates run even if earlier ones fail (for diagnostics).
        overall_verdict = FAIL if ANY gate verdict is FAIL.
        SKIP verdicts do NOT cause failure.

        Args:
            persona: Active persona contract
            intent: Declared intent for the action
            activation_time: When the persona was activated (for TTL check)
            tenant_id: Tenant scope (used for async FingerprintCache lookups)

        Returns:
            AnomalyResult with all 4 gate results and overall verdict
        """
        # Pre-fetch Gate 4 fingerprint history. FingerprintCache is async/Redis-backed;
        # _gate4_drift is synchronous so we resolve the history here before calling it.
        if self.fingerprint_store is not None and hasattr(self.fingerprint_store, "get_baseline"):
            baseline = await self.fingerprint_store.get_baseline(tenant_id, str(persona.id))
            drift_history: Optional[list] = baseline["fingerprints"]
        else:
            drift_history = None  # _gate4_drift will fall back to dict lookup

        gate1 = self._gate1_scope(persona, intent)
        gate2 = await self._gate2_intent(persona, intent)
        gate3 = self._gate3_ttl(persona, activation_time)
        gate4 = self._gate4_drift(persona, intent, drift_history)

        gates = [gate1, gate2, gate3, gate4]
        overall = GateVerdict.PASS
        for g in gates:
            if g.verdict == GateVerdict.FAIL:
                overall = GateVerdict.FAIL
                break

        fingerprint = self.compute_fingerprint(
            intent.tool_name,
            intent.resource_targets,
        )

        # Determine risk level from persona risk tolerance
        risk_map = {
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
            "critical": RiskLevel.CRITICAL,
        }
        risk_level = risk_map.get(str(persona.risk_tolerance.value).lower(), RiskLevel.MEDIUM)

        return AnomalyResult(
            gates=gates,
            overall_verdict=overall,
            risk_level=risk_level,
            persona_id=persona.id,
            action_fingerprint=fingerprint,
        )

    def _gate1_scope(self, persona: PersonaContract, intent: IntentDeclaration) -> GateResult:
        """SCOPE CHECK: Is tool_name in persona.allowed_tools?
        Are resource_targets within persona.resource_scopes?

        Score: 1.0 if all match, 0.0 if any don't.
        Threshold: 1.0 (binary pass/fail).
        """
        import fnmatch

        failures = []

        if intent.tool_name not in persona.allowed_tools:
            failures.append(f"tool '{intent.tool_name}' not in allowed_tools {persona.allowed_tools}")

        for target in intent.resource_targets:
            if not any(fnmatch.fnmatch(target, scope) for scope in persona.resource_scopes):
                failures.append(f"resource '{target}' out of scope {persona.resource_scopes}")

        if failures:
            return GateResult(
                gate_name="scope",
                verdict=GateVerdict.FAIL,
                score=0.0,
                threshold=1.0,
                details="; ".join(failures),
            )
        return GateResult(
            gate_name="scope",
            verdict=GateVerdict.PASS,
            score=1.0,
            threshold=1.0,
            details="All scope checks passed",
        )

    async def _gate2_intent(self, persona: PersonaContract, intent: IntentDeclaration) -> GateResult:
        """INTENT SIMILARITY: Embed intent.planned_action, compare cosine similarity
        against persona.intent_patterns embeddings.

        Score: max cosine similarity across all intent_patterns.
        Threshold: config.gate_intent_threshold (default 0.75).
        If no embedding_service, SKIP this gate (cold start mode).
        """
        if self.embedding_service is None:
            return GateResult(
                gate_name="intent",
                verdict=GateVerdict.SKIP,
                score=0.0,
                threshold=self.config.gate_intent_threshold,
                details="No embedding service available (cold start mode)",
            )

        if not persona.intent_patterns:
            return GateResult(
                gate_name="intent",
                verdict=GateVerdict.SKIP,
                score=0.0,
                threshold=self.config.gate_intent_threshold,
                details="Persona has no intent patterns to compare against",
            )

        scores = self.embedding_service.similarities(intent.planned_action, persona.intent_patterns)
        max_score = max(scores) if scores else 0.0

        verdict = GateVerdict.PASS if max_score >= self.config.gate_intent_threshold else GateVerdict.FAIL
        return GateResult(
            gate_name="intent",
            verdict=verdict,
            score=max_score,
            threshold=self.config.gate_intent_threshold,
            details=f"Max cosine similarity {max_score:.3f} vs threshold {self.config.gate_intent_threshold}",
        )

    def _gate3_ttl(self, persona: PersonaContract, activation_time: datetime) -> GateResult:
        """TTL CHECK: Has persona been active longer than max_ttl_seconds?

        Score: remaining_seconds / max_ttl_seconds (1.0 = just activated).
        Threshold: 0.0 (any time remaining = pass).
        """
        elapsed = (datetime.now(timezone.utc) - activation_time).total_seconds()
        remaining = persona.max_ttl_seconds - elapsed
        score = max(0.0, remaining / persona.max_ttl_seconds)

        if remaining <= 0:
            return GateResult(
                gate_name="ttl",
                verdict=GateVerdict.FAIL,
                score=0.0,
                threshold=0.0,
                details=f"Persona TTL expired: {abs(remaining):.1f}s overdue (max {persona.max_ttl_seconds}s)",
            )
        return GateResult(
            gate_name="ttl",
            verdict=GateVerdict.PASS,
            score=score,
            threshold=0.0,
            details=f"{remaining:.1f}s remaining of {persona.max_ttl_seconds}s TTL",
        )

    def _gate4_drift(
        self,
        persona: PersonaContract,
        intent: IntentDeclaration,
        history: Optional[list] = None,
    ) -> GateResult:
        """BEHAVIORAL DRIFT: Is this action's fingerprint within N sigma of
        the persona's historical baseline?

        Fingerprint = hash(tool_name + sorted(resource_targets) + intent_category).
        Score: how many sigma from mean. Lower is better.
        Threshold: config.gate_drift_sigma (default 2.5).
        If <10 historical samples, SKIP (insufficient baseline).

        Args:
            persona: Active persona contract
            intent: Declared intent for the action
            history: Pre-fetched fingerprint history. When None, falls back to
                     dict-based fingerprint_store (used in tests / cold start).
        """
        import statistics

        fingerprint = self.compute_fingerprint(intent.tool_name, intent.resource_targets)
        if history is None:
            history_key = f"{persona.id}:fingerprints"
            history = self.fingerprint_store.get(history_key, []) if isinstance(self.fingerprint_store, dict) else []

        if len(history) < 10:
            return GateResult(
                gate_name="drift",
                verdict=GateVerdict.SKIP,
                score=0.0,
                threshold=self.config.gate_drift_sigma,
                details=f"Insufficient baseline: {len(history)}/10 samples",
            )

        # Count frequency of each fingerprint in history
        freq: dict[str, int] = {}
        for fp in history:
            freq[fp] = freq.get(fp, 0) + 1

        counts = list(freq.values())
        mean = statistics.mean(counts)
        if len(counts) < 2:
            stdev = 0.0
        else:
            stdev = statistics.stdev(counts)

        current_count = freq.get(fingerprint, 0)
        if stdev == 0:
            sigma_distance = 0.0
        else:
            sigma_distance = abs(current_count - mean) / stdev

        verdict = GateVerdict.PASS if sigma_distance <= self.config.gate_drift_sigma else GateVerdict.FAIL
        return GateResult(
            gate_name="drift",
            verdict=verdict,
            score=sigma_distance,
            threshold=self.config.gate_drift_sigma,
            details=f"Action is {sigma_distance:.2f}σ from baseline (threshold {self.config.gate_drift_sigma}σ)",
        )

    @staticmethod
    def compute_fingerprint(tool_name: str, resource_targets: list[str], intent_category: str = "") -> str:
        """Compute behavioral fingerprint for drift tracking.

        Returns first 16 chars of SHA256 hash.
        """
        sorted_targets = ":".join(sorted(resource_targets))
        content = f"{tool_name}:{sorted_targets}:{intent_category}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
