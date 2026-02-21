"""Creates and verifies immutable seals. The audit backbone of NEXUS.

The Merkle chain ensures you can't delete or modify a seal without
breaking all subsequent fingerprints.

Implementation notes:
- seal_content for hashing = f"{seal.chain_id}:{seal.step_index}:{seal.tenant_id}:{seal.persona_id}:{seal.intent.tool_name}:{seal.intent.tool_params}:{seal.anomaly_result.overall_verdict}"
- Fingerprint = SHA256(previous_fingerprint + SHA256(seal_content))
"""

import hashlib
from datetime import datetime
from typing import Any

from nexus.types import (
    Seal, IntentDeclaration, AnomalyResult, ActionStatus,
)
from nexus.exceptions import SealIntegrityError


class Notary:
    """Creates and verifies immutable seals. The audit backbone."""

    def __init__(self):
        self._last_fingerprint: str = ""  # Merkle chain state

    def create_seal(
        self,
        chain_id: str,
        step_index: int,
        tenant_id: str,
        persona_id: str,
        intent: IntentDeclaration,
        anomaly_result: AnomalyResult,
    ) -> Seal:
        """Create a new seal in PENDING status.

        Fingerprint = SHA256(previous_fingerprint + seal_content_hash).
        This creates the Merkle chain — any tampering breaks the chain.

        Args:
            chain_id: Parent chain ID
            step_index: Position in the chain
            tenant_id: Tenant context
            persona_id: Active persona name
            intent: Declared intent
            anomaly_result: Result of 4-gate check

        Returns:
            New Seal in PENDING status with computed fingerprint
        """
        seal = Seal(
            chain_id=chain_id,
            step_index=step_index,
            tenant_id=tenant_id,
            persona_id=persona_id,
            intent=intent,
            anomaly_result=anomaly_result,
            tool_name=intent.tool_name,
            tool_params=intent.tool_params,
            status=ActionStatus.PENDING,
            parent_fingerprint=self._last_fingerprint,
        )
        content = self._seal_content_string(seal)
        seal.fingerprint = self._compute_fingerprint(self._last_fingerprint, content)
        self._last_fingerprint = seal.fingerprint
        return seal

    def finalize_seal(
        self, seal: Seal, tool_result: Any,
        status: ActionStatus, error: str = None
    ) -> Seal:
        """Finalize seal after execution.

        Updates status, result, completed_at.
        Does NOT recompute fingerprint — that was set at creation.

        Args:
            seal: The seal to finalize
            tool_result: Result from tool execution
            status: Final status (EXECUTED, FAILED, BLOCKED)
            error: Error message if failed

        Returns:
            Updated seal with final status
        """
        return seal.model_copy(update={
            "tool_result": tool_result,
            "status": status,
            "completed_at": datetime.utcnow(),
            "error": error,
        })

    def verify_chain(self, seals: list[Seal]) -> bool:
        """Verify Merkle chain integrity.

        Recompute each fingerprint and check it matches.

        Args:
            seals: Ordered list of seals in a chain

        Returns:
            True if chain is intact

        Raises:
            SealIntegrityError: with details on which seal broke the chain
        """
        if not seals:
            return True
        seals = sorted(seals, key=lambda s: s.step_index)
        prev_fingerprint = ""
        for seal in seals:
            content = self._seal_content_string(seal)
            expected = self._compute_fingerprint(prev_fingerprint, content)
            if seal.fingerprint != expected:
                raise SealIntegrityError(
                    f"Seal integrity broken at step {seal.step_index} (seal_id={seal.id}): "
                    f"expected fingerprint {expected!r}, got {seal.fingerprint!r}"
                )
            prev_fingerprint = seal.fingerprint
        return True

    @staticmethod
    def _compute_fingerprint(previous: str, seal_content: str) -> str:
        """SHA256(previous_fingerprint + SHA256(seal_content))"""
        content_hash = hashlib.sha256(seal_content.encode()).hexdigest()
        return hashlib.sha256(f"{previous}{content_hash}".encode()).hexdigest()

    @staticmethod
    def _seal_content_string(seal: Seal) -> str:
        """Generate deterministic content string for fingerprinting."""
        return (
            f"{seal.chain_id}:{seal.step_index}:{seal.tenant_id}:"
            f"{seal.persona_id}:{seal.intent.tool_name}:"
            f"{seal.intent.tool_params}:{seal.anomaly_result.overall_verdict}"
        )
