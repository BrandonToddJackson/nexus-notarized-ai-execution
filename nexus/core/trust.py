"""Trust tier management for persona behavioral contracts.

Trust tiers reflect cumulative track record:
  cold_start  → new persona, no history
  established → 50+ successful actions
  trusted     → 500+ successful, <1% anomaly rate

**Integration points:**

- ``maybe_degrade()`` is called by ``NexusEngine`` after any Gate failure.
  It runs in-memory: the engine updates ``persona_manager._contracts[name]``
  immediately so subsequent actions in the same session see the degraded tier.

- ``maybe_promote()`` requires cumulative action counts that are only available
  from the database (``Repository.count_persona_actions``). It is NOT called
  by the engine directly. Callers with DB access should call it after successful
  chains and persist the updated contract back to the DB.

  Example (in an API route or background task):
      counts = await repo.get_persona_action_counts(tenant_id, persona_name)
      updated = maybe_promote(contract, **counts)
      await repo.update_persona_trust_tier(updated)
"""

from nexus.types import PersonaContract, TrustTier


# Thresholds for tier promotion / degradation
ESTABLISHED_MIN_ACTIONS = 50
TRUSTED_MIN_ACTIONS = 500
TRUSTED_MAX_ANOMALY_RATE = 0.01  # 1%


def evaluate_trust_tier(
    successful_actions: int,
    total_actions: int,
    anomaly_count: int,
) -> TrustTier:
    """Compute the appropriate trust tier from lifetime action stats.

    Args:
        successful_actions: Number of non-blocked, non-failed actions
        total_actions: Total actions attempted (including blocked)
        anomaly_count: Actions that triggered anomaly gates

    Returns:
        The highest tier the persona qualifies for
    """
    if total_actions == 0:
        return TrustTier.COLD_START

    anomaly_rate = anomaly_count / total_actions if total_actions > 0 else 0.0

    if (
        successful_actions >= TRUSTED_MIN_ACTIONS
        and anomaly_rate <= TRUSTED_MAX_ANOMALY_RATE
    ):
        return TrustTier.TRUSTED

    if successful_actions >= ESTABLISHED_MIN_ACTIONS:
        return TrustTier.ESTABLISHED

    return TrustTier.COLD_START


def maybe_promote(
    contract: PersonaContract,
    successful_actions: int,
    total_actions: int,
    anomaly_count: int,
) -> PersonaContract:
    """Return updated contract with recalculated trust tier (immutable).

    Does not persist — caller is responsible for saving the updated contract.
    """
    new_tier = evaluate_trust_tier(successful_actions, total_actions, anomaly_count)
    if new_tier == contract.trust_tier:
        return contract
    return contract.model_copy(update={"trust_tier": new_tier})


def maybe_degrade(contract: PersonaContract) -> PersonaContract:
    """Degrade trust tier by one step on anomaly detection.

    cold_start  → cold_start (floor)
    established → cold_start
    trusted     → established
    """
    degradation_map = {
        TrustTier.TRUSTED: TrustTier.ESTABLISHED,
        TrustTier.ESTABLISHED: TrustTier.COLD_START,
        TrustTier.COLD_START: TrustTier.COLD_START,
    }
    new_tier = degradation_map[contract.trust_tier]
    if new_tier == contract.trust_tier:
        return contract
    return contract.model_copy(update={"trust_tier": new_tier})
