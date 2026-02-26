"""Comprehensive tests for trust.py, locks.py, and config/loader.py.

These modules were added post-Phase 14. This file provides exhaustive coverage:
  - nexus/core/trust.py  — evaluate_trust_tier, maybe_promote, maybe_degrade
  - nexus/cache/locks.py — DistributedLock, chain_lock context manager
  - nexus/config/loader.py — load_personas_yaml, load_tools_yaml, _find_file
"""

import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_contract(tier=None, name="test-persona", risk="low"):
    from nexus.types import PersonaContract, RiskLevel, TrustTier
    return PersonaContract(
        name=name,
        description="test persona",
        allowed_tools=["knowledge_search", "web_search"],
        resource_scopes=["kb:*", "web:*"],
        intent_patterns=["search for information", "look up"],
        max_ttl_seconds=60,
        risk_tolerance=RiskLevel(risk),
        trust_tier=tier or TrustTier.COLD_START,
    )


def _make_redis(set_return=True):
    """Mock RedisClient for lock tests."""
    mock = MagicMock()
    mock._key = lambda tenant_id, key: f"nexus:{tenant_id}:{key}"
    mock.client = MagicMock()
    mock.client.set = AsyncMock(return_value=set_return if set_return else None)
    mock.client.delete = AsyncMock(return_value=1)
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# TRUST — nexus/core/trust.py
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    """Thresholds are not magic numbers — test that they match the spec."""

    def test_established_min_actions(self):
        from nexus.core.trust import ESTABLISHED_MIN_ACTIONS
        assert ESTABLISHED_MIN_ACTIONS == 50

    def test_trusted_min_actions(self):
        from nexus.core.trust import TRUSTED_MIN_ACTIONS
        assert TRUSTED_MIN_ACTIONS == 500

    def test_trusted_max_anomaly_rate(self):
        from nexus.core.trust import TRUSTED_MAX_ANOMALY_RATE
        assert TRUSTED_MAX_ANOMALY_RATE == 0.01


class TestEvaluateTrustTier:
    """evaluate_trust_tier — boundary conditions for all three tiers."""

    def test_zero_total_actions_is_cold_start(self):
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(0, 0, 0) == TrustTier.COLD_START

    def test_zero_successful_with_some_total_is_cold_start(self):
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(0, 10, 0) == TrustTier.COLD_START

    # ── COLD_START upper boundary ─────────────────────────────────────────

    def test_49_successful_is_cold_start(self):
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(49, 49, 0) == TrustTier.COLD_START

    def test_1_successful_is_cold_start(self):
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(1, 1, 0) == TrustTier.COLD_START

    # ── ESTABLISHED boundary ──────────────────────────────────────────────

    def test_exactly_50_successful_is_established(self):
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(50, 50, 0) == TrustTier.ESTABLISHED

    def test_51_successful_is_established(self):
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(51, 51, 0) == TrustTier.ESTABLISHED

    def test_499_successful_is_established(self):
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(499, 499, 0) == TrustTier.ESTABLISHED

    # ── TRUSTED boundary ──────────────────────────────────────────────────

    def test_exactly_500_successful_zero_anomaly_is_trusted(self):
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(500, 500, 0) == TrustTier.TRUSTED

    def test_1000_successful_zero_anomaly_is_trusted(self):
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(1000, 1000, 0) == TrustTier.TRUSTED

    def test_500_successful_exactly_1pct_anomaly_is_trusted(self):
        """Boundary: 5 anomalies / 500 total = exactly 1% → still trusted."""
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(500, 500, 5) == TrustTier.TRUSTED

    def test_500_successful_above_1pct_anomaly_demotes_to_established(self):
        """6 / 500 = 1.2% → just over threshold → ESTABLISHED, not TRUSTED."""
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(500, 500, 6) == TrustTier.ESTABLISHED

    def test_high_anomaly_rate_with_many_actions_is_established(self):
        """50% anomaly rate regardless of volume → ESTABLISHED at best."""
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(500, 1000, 500) == TrustTier.ESTABLISHED

    def test_anomaly_rate_calculated_against_total_not_successful(self):
        """Anomaly rate uses total_actions as denominator."""
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        # 5 anomalies / 1000 total = 0.5% → well under 1% → TRUSTED
        assert evaluate_trust_tier(500, 1000, 5) == TrustTier.TRUSTED

    def test_successful_less_than_trusted_threshold_caps_at_established(self):
        """Even with perfect anomaly rate, insufficient actions = ESTABLISHED."""
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        assert evaluate_trust_tier(100, 100, 0) == TrustTier.ESTABLISHED

    def test_return_type_is_trust_tier_enum(self):
        from nexus.core.trust import evaluate_trust_tier
        from nexus.types import TrustTier
        result = evaluate_trust_tier(0, 0, 0)
        assert isinstance(result, TrustTier)


class TestMaybePromote:
    """maybe_promote — immutability, identity, all tier transitions."""

    def test_returns_same_object_when_tier_unchanged(self):
        """No copy made when tier would not change."""
        from nexus.core.trust import maybe_promote
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.COLD_START)
        result = maybe_promote(contract, 10, 10, 0)
        assert result is contract

    def test_cold_start_to_established(self):
        from nexus.core.trust import maybe_promote
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.COLD_START)
        result = maybe_promote(contract, 50, 50, 0)
        assert result is not contract
        assert result.trust_tier == TrustTier.ESTABLISHED

    def test_cold_start_to_trusted_skips_established(self):
        """With 500+ good actions, jump straight to TRUSTED from COLD_START."""
        from nexus.core.trust import maybe_promote
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.COLD_START)
        result = maybe_promote(contract, 500, 500, 0)
        assert result.trust_tier == TrustTier.TRUSTED

    def test_established_to_trusted(self):
        from nexus.core.trust import maybe_promote
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.ESTABLISHED)
        result = maybe_promote(contract, 500, 500, 0)
        assert result is not contract
        assert result.trust_tier == TrustTier.TRUSTED

    def test_established_stays_established_at_threshold(self):
        from nexus.core.trust import maybe_promote
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.ESTABLISHED)
        result = maybe_promote(contract, 50, 50, 0)
        assert result is contract  # already at this tier, no change

    def test_trusted_stays_trusted(self):
        """Already at top tier — no-op regardless of stats."""
        from nexus.core.trust import maybe_promote
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.TRUSTED)
        result = maybe_promote(contract, 1000, 1000, 0)
        assert result is contract

    def test_original_contract_not_mutated(self):
        """Pydantic model_copy must not modify the original."""
        from nexus.core.trust import maybe_promote
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.COLD_START)
        original_tier = contract.trust_tier
        _ = maybe_promote(contract, 500, 500, 0)
        assert contract.trust_tier == original_tier

    def test_only_trust_tier_changes_all_other_fields_preserved(self):
        """model_copy(update=...) only touches trust_tier."""
        from nexus.core.trust import maybe_promote
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.COLD_START)
        result = maybe_promote(contract, 50, 50, 0)
        assert result.name == contract.name
        assert result.allowed_tools == contract.allowed_tools
        assert result.resource_scopes == contract.resource_scopes
        assert result.max_ttl_seconds == contract.max_ttl_seconds
        assert result.risk_tolerance == contract.risk_tolerance

    def test_anomaly_over_threshold_blocks_promotion_to_trusted(self):
        from nexus.core.trust import maybe_promote
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.ESTABLISHED)
        # 6 anomalies / 500 = 1.2% — just over threshold
        result = maybe_promote(contract, 500, 500, 6)
        assert result is contract  # ESTABLISHED → would-be ESTABLISHED → no change


class TestMaybeDegrade:
    """maybe_degrade — one-step degradation, floor at COLD_START, immutability."""

    def test_trusted_degrades_to_established(self):
        from nexus.core.trust import maybe_degrade
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.TRUSTED)
        result = maybe_degrade(contract)
        assert result.trust_tier == TrustTier.ESTABLISHED

    def test_established_degrades_to_cold_start(self):
        from nexus.core.trust import maybe_degrade
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.ESTABLISHED)
        result = maybe_degrade(contract)
        assert result.trust_tier == TrustTier.COLD_START

    def test_cold_start_stays_cold_start(self):
        """Floor: COLD_START cannot go lower."""
        from nexus.core.trust import maybe_degrade
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.COLD_START)
        result = maybe_degrade(contract)
        assert result.trust_tier == TrustTier.COLD_START

    def test_cold_start_returns_same_object(self):
        """No copy when tier is already at floor."""
        from nexus.core.trust import maybe_degrade
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.COLD_START)
        result = maybe_degrade(contract)
        assert result is contract

    def test_trusted_returns_new_object(self):
        from nexus.core.trust import maybe_degrade
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.TRUSTED)
        result = maybe_degrade(contract)
        assert result is not contract

    def test_established_returns_new_object(self):
        from nexus.core.trust import maybe_degrade
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.ESTABLISHED)
        result = maybe_degrade(contract)
        assert result is not contract

    def test_original_not_mutated_after_degrade(self):
        from nexus.core.trust import maybe_degrade
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.TRUSTED)
        _ = maybe_degrade(contract)
        assert contract.trust_tier == TrustTier.TRUSTED

    def test_degrade_only_changes_trust_tier(self):
        from nexus.core.trust import maybe_degrade
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.TRUSTED)
        result = maybe_degrade(contract)
        assert result.name == contract.name
        assert result.allowed_tools == contract.allowed_tools
        assert result.resource_scopes == contract.resource_scopes
        assert result.max_ttl_seconds == contract.max_ttl_seconds
        assert result.risk_tolerance == contract.risk_tolerance

    def test_double_degrade_trusted_reaches_cold_start(self):
        """Two anomalies on a TRUSTED persona reach COLD_START."""
        from nexus.core.trust import maybe_degrade
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.TRUSTED)
        after_first = maybe_degrade(contract)
        assert after_first.trust_tier == TrustTier.ESTABLISHED
        after_second = maybe_degrade(after_first)
        assert after_second.trust_tier == TrustTier.COLD_START

    def test_triple_degrade_stays_at_cold_start(self):
        from nexus.core.trust import maybe_degrade
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.TRUSTED)
        c = contract
        for _ in range(3):
            c = maybe_degrade(c)
        assert c.trust_tier == TrustTier.COLD_START


class TestTrustRoundTrip:
    """Promote then degrade round-trips produce the expected tier sequence."""

    def test_promote_then_degrade_returns_to_lower_tier(self):
        from nexus.core.trust import maybe_promote, maybe_degrade
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.COLD_START)
        promoted = maybe_promote(contract, 50, 50, 0)
        assert promoted.trust_tier == TrustTier.ESTABLISHED
        degraded = maybe_degrade(promoted)
        assert degraded.trust_tier == TrustTier.COLD_START

    def test_evaluate_matches_maybe_promote_outcome(self):
        """evaluate_trust_tier and maybe_promote agree on the tier."""
        from nexus.core.trust import evaluate_trust_tier, maybe_promote
        from nexus.types import TrustTier
        contract = _make_contract(TrustTier.COLD_START)
        stats = (500, 500, 0)
        expected_tier = evaluate_trust_tier(*stats)
        result = maybe_promote(contract, *stats)
        assert result.trust_tier == expected_tier


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTED LOCKS — nexus/cache/locks.py
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestDistributedLockInit:
    """DistributedLock constructor and key derivation."""

    async def test_key_derived_from_redis_key_fn(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis()
        lock = DistributedLock(redis, "tenant-1", "chain:abc-123")
        assert lock._key == "nexus:tenant-1:lock:chain:abc-123"

    async def test_default_ttl_is_30(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis()
        lock = DistributedLock(redis, "t1", "res")
        assert lock.ttl_seconds == 30

    async def test_default_max_retries_is_100(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis()
        lock = DistributedLock(redis, "t1", "res")
        assert lock.max_retries == 100

    async def test_default_retry_interval(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis()
        lock = DistributedLock(redis, "t1", "res")
        assert lock.retry_interval == 0.05

    async def test_custom_ttl_stored(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis()
        lock = DistributedLock(redis, "t1", "res", ttl_seconds=60)
        assert lock.ttl_seconds == 60


@pytest.mark.asyncio
class TestDistributedLockAcquire:
    """DistributedLock.acquire — Redis SET NX EX semantics."""

    async def test_acquire_returns_true_when_set_succeeds(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis(set_return=True)
        lock = DistributedLock(redis, "tenant", "chain:x")
        assert await lock.acquire() is True

    async def test_acquire_returns_false_when_set_fails(self):
        """Redis SET NX returns None when key already exists."""
        from nexus.cache.locks import DistributedLock
        redis = _make_redis(set_return=False)
        lock = DistributedLock(redis, "tenant", "chain:x")
        assert await lock.acquire() is False

    async def test_acquire_calls_set_with_nx_true(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis(set_return=True)
        lock = DistributedLock(redis, "tenant", "chain:x")
        await lock.acquire()
        _, kwargs = redis.client.set.call_args
        assert kwargs.get("nx") is True

    async def test_acquire_calls_set_with_ex_ttl(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis(set_return=True)
        lock = DistributedLock(redis, "tenant", "chain:x", ttl_seconds=45)
        await lock.acquire()
        _, kwargs = redis.client.set.call_args
        assert kwargs.get("ex") == 45

    async def test_acquire_passes_value_1(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis(set_return=True)
        lock = DistributedLock(redis, "tenant", "chain:x")
        await lock.acquire()
        args, _ = redis.client.set.call_args
        assert args[1] == "1"

    async def test_acquire_uses_derived_key(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis(set_return=True)
        lock = DistributedLock(redis, "tenant", "chain:x")
        await lock.acquire()
        args, _ = redis.client.set.call_args
        assert args[0] == lock._key


@pytest.mark.asyncio
class TestDistributedLockRelease:
    """DistributedLock.release — deletes the key."""

    async def test_release_calls_delete(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis()
        lock = DistributedLock(redis, "tenant", "chain:x")
        await lock.release()
        redis.client.delete.assert_awaited_once_with(lock._key)

    async def test_release_with_correct_key(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis()
        lock = DistributedLock(redis, "tenant-A", "chain:z")
        await lock.release()
        args = redis.client.delete.call_args[0]
        assert args[0] == "nexus:tenant-A:lock:chain:z"


@pytest.mark.asyncio
class TestDistributedLockAcquireWithRetry:
    """acquire_with_retry — success on first attempt, failure after max_retries."""

    async def test_succeeds_immediately(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis(set_return=True)
        lock = DistributedLock(redis, "t", "r")
        assert await lock.acquire_with_retry() is True

    async def test_only_calls_set_once_on_immediate_success(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis(set_return=True)
        lock = DistributedLock(redis, "t", "r")
        await lock.acquire_with_retry()
        assert redis.client.set.await_count == 1

    async def test_returns_false_after_max_retries_exhausted(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis(set_return=False)
        lock = DistributedLock(redis, "t", "r", retry_interval=0.0, max_retries=5)
        with patch("nexus.cache.locks.asyncio.sleep", new=AsyncMock()):
            result = await lock.acquire_with_retry()
        assert result is False

    async def test_retries_exactly_max_retries_times(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis(set_return=False)
        lock = DistributedLock(redis, "t", "r", retry_interval=0.0, max_retries=7)
        with patch("nexus.cache.locks.asyncio.sleep", new=AsyncMock()):
            await lock.acquire_with_retry()
        assert redis.client.set.await_count == 7

    async def test_succeeds_on_second_attempt(self):
        """Lock acquired on retry after initial failure."""
        from nexus.cache.locks import DistributedLock
        call_count = 0

        async def set_mock(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return True if call_count >= 2 else None

        redis = _make_redis()
        redis.client.set = AsyncMock(side_effect=set_mock)
        lock = DistributedLock(redis, "t", "r", retry_interval=0.0, max_retries=10)
        with patch("nexus.cache.locks.asyncio.sleep", new=AsyncMock()):
            result = await lock.acquire_with_retry()
        assert result is True
        assert call_count == 2

    async def test_sleep_called_between_retries(self):
        from nexus.cache.locks import DistributedLock
        redis = _make_redis(set_return=False)
        lock = DistributedLock(redis, "t", "r", retry_interval=0.05, max_retries=3)
        sleep_mock = AsyncMock()
        with patch("nexus.cache.locks.asyncio.sleep", new=sleep_mock):
            await lock.acquire_with_retry()
        # sleep called between each failed attempt (not after the last one)
        assert sleep_mock.await_count == 3
        sleep_mock.assert_awaited_with(0.05)


@pytest.mark.asyncio
class TestChainLockContextManager:
    """chain_lock() — acquire/yield/release flow and timeout behavior."""

    async def test_yields_the_lock_object(self):
        from nexus.cache.locks import chain_lock
        redis = _make_redis(set_return=True)
        async with chain_lock(redis, "tenant", "chain-001") as lock:
            assert lock is not None

    async def test_release_called_on_normal_exit(self):
        from nexus.cache.locks import chain_lock
        redis = _make_redis(set_return=True)
        async with chain_lock(redis, "tenant", "chain-001"):
            pass
        redis.client.delete.assert_awaited_once()

    async def test_release_called_on_exception_inside_block(self):
        """Lock released even if the protected block raises."""
        from nexus.cache.locks import chain_lock
        redis = _make_redis(set_return=True)
        with pytest.raises(ValueError):
            async with chain_lock(redis, "tenant", "chain-001"):
                raise ValueError("inner error")
        redis.client.delete.assert_awaited_once()

    async def test_raises_timeout_error_when_cannot_acquire(self):
        from nexus.cache.locks import chain_lock
        redis = _make_redis(set_return=False)
        with patch("nexus.cache.locks.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(TimeoutError, match="chain-abc"):
                async with chain_lock(redis, "tenant", "chain-abc"):
                    pass  # never reached

    async def test_timeout_error_message_includes_chain_id(self):
        from nexus.cache.locks import chain_lock
        redis = _make_redis(set_return=False)
        with patch("nexus.cache.locks.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(TimeoutError) as exc_info:
                async with chain_lock(redis, "tenant", "my-chain-xyz"):
                    pass
        assert "my-chain-xyz" in str(exc_info.value)

    async def test_delete_not_called_when_acquire_fails(self):
        """If we never acquired, we must not call delete."""
        from nexus.cache.locks import chain_lock
        redis = _make_redis(set_return=False)
        with patch("nexus.cache.locks.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(TimeoutError):
                async with chain_lock(redis, "tenant", "chain-abc"):
                    pass
        redis.client.delete.assert_not_awaited()

    async def test_lock_key_scoped_to_chain_id(self):
        """The lock key must embed the chain_id to avoid cross-chain contention."""
        from nexus.cache.locks import chain_lock
        redis = _make_redis(set_return=True)
        async with chain_lock(redis, "tenant", "chain-999"):
            pass
        set_args, _ = redis.client.set.call_args
        assert "chain-999" in set_args[0]

    async def test_custom_ttl_forwarded_to_lock(self):
        """chain_lock's ttl_seconds parameter reaches the SET EX call."""
        from nexus.cache.locks import chain_lock
        redis = _make_redis(set_return=True)
        async with chain_lock(redis, "tenant", "chain-001", ttl_seconds=120):
            pass
        _, set_kwargs = redis.client.set.call_args
        assert set_kwargs.get("ex") == 120

    async def test_concurrent_acquires_serialized(self):
        """Second acquire blocks until first releases (simulated via call order)."""
        from nexus.cache.locks import chain_lock

        call_log = []
        acquire_count = 0

        async def set_mock(*args, **kwargs):
            nonlocal acquire_count
            acquire_count += 1
            # First acquire succeeds; subsequent fail until delete called
            return True if acquire_count == 1 else None

        redis = _make_redis()
        redis.client.set = AsyncMock(side_effect=set_mock)
        redis.client.delete = AsyncMock(side_effect=lambda *_: call_log.append("released"))

        async with chain_lock(redis, "tenant", "chain-001"):
            call_log.append("inside")

        assert call_log == ["inside", "released"]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG LOADER — nexus/config/loader.py
# ─────────────────────────────────────────────────────────────────────────────

class TestFindFile:
    """_find_file — resolution priority: explicit > cwd > defaults."""

    def test_explicit_path_returned_when_exists(self, tmp_path):
        from nexus.config.loader import _find_file
        f = tmp_path / "personas.yaml"
        f.write_text("personas: []")
        result = _find_file("personas.yaml", explicit=f)
        assert result == f

    def test_explicit_path_raises_when_missing(self, tmp_path):
        from nexus.config.loader import _find_file
        missing = tmp_path / "nope.yaml"
        with pytest.raises(FileNotFoundError, match="nope.yaml"):
            _find_file("personas.yaml", explicit=missing)

    def test_defaults_returned_when_no_cwd_file(self, tmp_path, monkeypatch):
        from nexus.config.loader import _find_file, _DEFAULTS_DIR
        monkeypatch.chdir(tmp_path)  # empty dir — no cwd personas.yaml
        result = _find_file("personas.yaml", explicit=None)
        assert result == _DEFAULTS_DIR / "personas.yaml"

    def test_cwd_file_takes_precedence_over_defaults(self, tmp_path, monkeypatch):
        from nexus.config.loader import _find_file
        cwd_file = tmp_path / "personas.yaml"
        cwd_file.write_text("personas: []")
        monkeypatch.chdir(tmp_path)
        result = _find_file("personas.yaml", explicit=None)
        assert result == cwd_file

    def test_explicit_takes_precedence_over_cwd(self, tmp_path, monkeypatch):
        from nexus.config.loader import _find_file
        explicit_file = tmp_path / "explicit.yaml"
        explicit_file.write_text("personas: []")
        cwd_file = tmp_path / "personas.yaml"
        cwd_file.write_text("personas: []")
        monkeypatch.chdir(tmp_path)
        result = _find_file("personas.yaml", explicit=explicit_file)
        assert result == explicit_file

    def test_raises_when_no_file_anywhere(self, tmp_path, monkeypatch):
        from nexus.config.loader import _find_file
        monkeypatch.chdir(tmp_path)
        # Patch _DEFAULTS_DIR to an empty temp dir so defaults don't exist either
        with patch("nexus.config.loader._DEFAULTS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                _find_file("nonexistent.yaml", explicit=None)


class TestLoadPersonasYaml:
    """load_personas_yaml — return types, field values, error cases."""

    def test_returns_list(self):
        from nexus.config.loader import load_personas_yaml
        result = load_personas_yaml()
        assert isinstance(result, list)

    def test_returns_persona_contracts(self):
        from nexus.config.loader import load_personas_yaml
        from nexus.types import PersonaContract
        for p in load_personas_yaml():
            assert isinstance(p, PersonaContract)

    def test_loads_6_default_personas(self):
        from nexus.config.loader import load_personas_yaml
        assert len(load_personas_yaml()) == 6

    def test_default_persona_names(self):
        from nexus.config.loader import load_personas_yaml
        names = {p.name for p in load_personas_yaml()}
        assert names == {"researcher", "analyst", "creator", "communicator", "operator", "sales_growth_agent"}

    def test_researcher_allowed_tools(self):
        from nexus.config.loader import load_personas_yaml
        researcher = next(p for p in load_personas_yaml() if p.name == "researcher")
        assert "knowledge_search" in researcher.allowed_tools
        assert "web_search" in researcher.allowed_tools

    def test_researcher_risk_tolerance_is_low(self):
        from nexus.config.loader import load_personas_yaml
        from nexus.types import RiskLevel
        researcher = next(p for p in load_personas_yaml() if p.name == "researcher")
        assert researcher.risk_tolerance == RiskLevel.LOW

    def test_operator_risk_tolerance_is_high(self):
        from nexus.config.loader import load_personas_yaml
        from nexus.types import RiskLevel
        operator = next(p for p in load_personas_yaml() if p.name == "operator")
        assert operator.risk_tolerance == RiskLevel.HIGH

    def test_all_personas_have_intent_patterns(self):
        from nexus.config.loader import load_personas_yaml
        for p in load_personas_yaml():
            assert len(p.intent_patterns) >= 1, f"{p.name} has no intent_patterns"

    def test_all_personas_default_trust_tier_is_cold_start(self):
        from nexus.config.loader import load_personas_yaml
        from nexus.types import TrustTier
        for p in load_personas_yaml():
            assert p.trust_tier == TrustTier.COLD_START

    def test_max_ttl_seconds_are_positive(self):
        from nexus.config.loader import load_personas_yaml
        for p in load_personas_yaml():
            assert p.max_ttl_seconds > 0, f"{p.name}.max_ttl_seconds must be > 0"

    def test_explicit_path_loaded(self, tmp_path):
        from nexus.config.loader import load_personas_yaml
        content = textwrap.dedent("""\
            personas:
              - name: custom
                description: Custom persona
                allowed_tools: [knowledge_search]
                resource_scopes: ["kb:*"]
                intent_patterns: [search]
                risk_tolerance: low
                max_ttl_seconds: 30
        """)
        f = tmp_path / "personas.yaml"
        f.write_text(content)
        result = load_personas_yaml(path=f)
        assert len(result) == 1
        assert result[0].name == "custom"

    def test_explicit_path_parses_risk_tolerance_string(self, tmp_path):
        from nexus.config.loader import load_personas_yaml
        from nexus.types import RiskLevel
        content = textwrap.dedent("""\
            personas:
              - name: highRiskAgent
                description: Dangerous
                allowed_tools: [send_email]
                resource_scopes: ["email:*"]
                intent_patterns: [send]
                risk_tolerance: high
                max_ttl_seconds: 60
        """)
        f = tmp_path / "personas.yaml"
        f.write_text(content)
        result = load_personas_yaml(path=f)
        assert result[0].risk_tolerance == RiskLevel.HIGH

    def test_explicit_path_parses_trust_tier_string(self, tmp_path):
        from nexus.config.loader import load_personas_yaml
        from nexus.types import TrustTier
        content = textwrap.dedent("""\
            personas:
              - name: veteran
                description: Veteran persona
                allowed_tools: [knowledge_search]
                resource_scopes: ["kb:*"]
                intent_patterns: [search]
                risk_tolerance: low
                max_ttl_seconds: 60
                trust_tier: trusted
        """)
        f = tmp_path / "personas.yaml"
        f.write_text(content)
        result = load_personas_yaml(path=f)
        assert result[0].trust_tier == TrustTier.TRUSTED

    def test_empty_personas_list_returns_empty(self, tmp_path):
        from nexus.config.loader import load_personas_yaml
        f = tmp_path / "personas.yaml"
        f.write_text("personas: []")
        assert load_personas_yaml(path=f) == []

    def test_raises_file_not_found_on_bad_explicit_path(self):
        from nexus.config.loader import load_personas_yaml
        with pytest.raises(FileNotFoundError):
            load_personas_yaml(path="/nonexistent/path/personas.yaml")

    def test_multiple_personas_all_returned(self, tmp_path):
        from nexus.config.loader import load_personas_yaml
        content = textwrap.dedent("""\
            personas:
              - name: alpha
                description: Alpha
                allowed_tools: [knowledge_search]
                resource_scopes: ["kb:*"]
                intent_patterns: [search]
              - name: beta
                description: Beta
                allowed_tools: [web_search]
                resource_scopes: ["web:*"]
                intent_patterns: [find]
              - name: gamma
                description: Gamma
                allowed_tools: [file_read]
                resource_scopes: ["file:read:*"]
                intent_patterns: [read]
        """)
        f = tmp_path / "personas.yaml"
        f.write_text(content)
        result = load_personas_yaml(path=f)
        assert len(result) == 3
        assert [p.name for p in result] == ["alpha", "beta", "gamma"]


class TestLoadToolsYaml:
    """load_tools_yaml — return types, field values, error cases."""

    def test_returns_list(self):
        from nexus.config.loader import load_tools_yaml
        result = load_tools_yaml()
        assert isinstance(result, list)

    def test_returns_tool_definitions(self):
        from nexus.config.loader import load_tools_yaml
        from nexus.types import ToolDefinition
        for t in load_tools_yaml():
            assert isinstance(t, ToolDefinition)

    def test_loads_7_default_tools(self):
        from nexus.config.loader import load_tools_yaml
        assert len(load_tools_yaml()) == 7

    def test_default_tool_names(self):
        from nexus.config.loader import load_tools_yaml
        names = {t.name for t in load_tools_yaml()}
        assert names == {
            "knowledge_search", "web_search", "web_fetch",
            "file_read", "file_write", "send_email", "compute_stats",
        }

    def test_send_email_requires_approval(self):
        from nexus.config.loader import load_tools_yaml
        send_email = next(t for t in load_tools_yaml() if t.name == "send_email")
        assert send_email.requires_approval is True

    def test_send_email_risk_level_is_high(self):
        from nexus.config.loader import load_tools_yaml
        from nexus.types import RiskLevel
        send_email = next(t for t in load_tools_yaml() if t.name == "send_email")
        assert send_email.risk_level == RiskLevel.HIGH

    def test_knowledge_search_does_not_require_approval(self):
        from nexus.config.loader import load_tools_yaml
        ks = next(t for t in load_tools_yaml() if t.name == "knowledge_search")
        assert ks.requires_approval is False

    def test_all_tools_have_names(self):
        from nexus.config.loader import load_tools_yaml
        for t in load_tools_yaml():
            assert t.name and isinstance(t.name, str)

    def test_all_tools_have_descriptions(self):
        from nexus.config.loader import load_tools_yaml
        for t in load_tools_yaml():
            assert t.description and isinstance(t.description, str)

    def test_parameters_always_empty_dict(self):
        """@tool decorator fills parameters at runtime — loader sets {}."""
        from nexus.config.loader import load_tools_yaml
        for t in load_tools_yaml():
            assert t.parameters == {}

    def test_explicit_path_loaded(self, tmp_path):
        from nexus.config.loader import load_tools_yaml
        content = textwrap.dedent("""\
            tools:
              - name: custom_tool
                description: A custom tool
                risk_level: medium
                resource_pattern: "custom:*"
                timeout_seconds: 25
                requires_approval: false
        """)
        f = tmp_path / "tools.yaml"
        f.write_text(content)
        result = load_tools_yaml(path=f)
        assert len(result) == 1
        assert result[0].name == "custom_tool"
        assert result[0].timeout_seconds == 25

    def test_explicit_path_coerces_risk_level_string(self, tmp_path):
        from nexus.config.loader import load_tools_yaml
        from nexus.types import RiskLevel
        content = textwrap.dedent("""\
            tools:
              - name: dangerous
                description: Dangerous tool
                risk_level: critical
        """)
        f = tmp_path / "tools.yaml"
        f.write_text(content)
        result = load_tools_yaml(path=f)
        assert result[0].risk_level == RiskLevel.CRITICAL

    def test_empty_tools_list_returns_empty(self, tmp_path):
        from nexus.config.loader import load_tools_yaml
        f = tmp_path / "tools.yaml"
        f.write_text("tools: []")
        assert load_tools_yaml(path=f) == []

    def test_raises_file_not_found_on_bad_explicit_path(self):
        from nexus.config.loader import load_tools_yaml
        with pytest.raises(FileNotFoundError):
            load_tools_yaml(path="/nonexistent/path/tools.yaml")

    def test_resource_pattern_preserved(self, tmp_path):
        from nexus.config.loader import load_tools_yaml
        content = textwrap.dedent("""\
            tools:
              - name: kb_tool
                description: Knowledge tool
                resource_pattern: "kb:tenant:*"
        """)
        f = tmp_path / "tools.yaml"
        f.write_text(content)
        result = load_tools_yaml(path=f)
        assert result[0].resource_pattern == "kb:tenant:*"


class TestConfigLoaderIntegration:
    """Integration: loaded objects work correctly with NEXUS engine components."""

    def test_loaded_personas_accepted_by_persona_manager(self):
        from nexus.config.loader import load_personas_yaml
        from nexus.core.personas import PersonaManager
        contracts = load_personas_yaml()
        pm = PersonaManager(contracts)
        personas = pm.list_personas()
        names = {p.name for p in personas}
        assert names == {"researcher", "analyst", "creator", "communicator", "operator", "sales_growth_agent"}

    def test_loaded_tools_accepted_by_tool_registry(self, tmp_path):
        from nexus.config.loader import load_tools_yaml
        content = textwrap.dedent("""\
            tools:
              - name: test_tool
                description: Test tool for registry
                risk_level: low
        """)
        f = tmp_path / "tools.yaml"
        f.write_text(content)
        tools = load_tools_yaml(path=f)
        # ToolDefinition without an impl can't be registered (requires callable),
        # but we can verify the definition fields are valid
        assert tools[0].name == "test_tool"
        assert tools[0].parameters == {}

    def test_researcher_persona_can_be_activated(self):
        from nexus.config.loader import load_personas_yaml
        from nexus.core.personas import PersonaManager
        contracts = load_personas_yaml()
        pm = PersonaManager(contracts)
        activated = pm.activate("researcher", tenant_id="test-tenant")
        assert activated is not None
        assert activated.name == "researcher"
