"""Tests for nexus/cache/fingerprints.py and nexus/cache/redis_client.py."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from nexus.cache.fingerprints import FingerprintCache

TENANT = "tenant-cache-001"
PERSONA = "researcher"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_redis(lrange_return: list[bytes] = None):
    """Build a mock RedisClient with list-operation support."""
    mock = MagicMock()
    mock._key = lambda tenant_id, key: f"nexus:{tenant_id}:{key}"
    mock.client = MagicMock()
    mock.client.rpush = AsyncMock(return_value=1)
    mock.client.ltrim = AsyncMock(return_value=True)
    mock.client.lrange = AsyncMock(return_value=lrange_return or [])
    return mock


def _expected_key(tenant_id: str, persona_id: str) -> str:
    return f"nexus:{tenant_id}:fingerprints:{persona_id}"


# ── FingerprintCache.store() ──────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFingerprintCacheStore:

    async def test_rpush_called_with_correct_key_and_value(self):
        redis = _make_redis()
        cache = FingerprintCache(redis_client=redis)
        await cache.store(TENANT, PERSONA, "fp-abc123")
        redis.client.rpush.assert_awaited_once_with(
            _expected_key(TENANT, PERSONA), "fp-abc123"
        )

    async def test_ltrim_called_after_rpush(self):
        redis = _make_redis()
        cache = FingerprintCache(redis_client=redis)
        await cache.store(TENANT, PERSONA, "fp-abc123")
        redis.client.ltrim.assert_awaited_once_with(
            _expected_key(TENANT, PERSONA), -1000, -1
        )

    async def test_rpush_before_ltrim(self):
        """Order matters: push first, then trim."""
        redis = _make_redis()
        manager = MagicMock()
        manager.attach_mock(redis.client.rpush, "rpush")
        manager.attach_mock(redis.client.ltrim, "ltrim")
        cache = FingerprintCache(redis_client=redis)
        await cache.store(TENANT, PERSONA, "fp-abc123")
        # Verify call order
        assert manager.mock_calls[0][0] == "rpush"
        assert manager.mock_calls[1][0] == "ltrim"

    async def test_store_different_personas_use_distinct_keys(self):
        redis = _make_redis()
        cache = FingerprintCache(redis_client=redis)
        await cache.store(TENANT, "researcher", "fp-r")
        await cache.store(TENANT, "analyst", "fp-a")
        calls = redis.client.rpush.await_args_list
        keys = [c[0][0] for c in calls]
        assert _expected_key(TENANT, "researcher") in keys
        assert _expected_key(TENANT, "analyst") in keys
        assert keys[0] != keys[1]

    async def test_store_different_tenants_use_distinct_keys(self):
        redis = _make_redis()
        cache = FingerprintCache(redis_client=redis)
        await cache.store("tenant-A", PERSONA, "fp-1")
        await cache.store("tenant-B", PERSONA, "fp-1")
        calls = redis.client.rpush.await_args_list
        keys = [c[0][0] for c in calls]
        assert keys[0] != keys[1]

    async def test_store_returns_none(self):
        redis = _make_redis()
        cache = FingerprintCache(redis_client=redis)
        result = await cache.store(TENANT, PERSONA, "fp-abc")
        assert result is None


# ── FingerprintCache.get_baseline() ──────────────────────────────────────────

@pytest.mark.asyncio
class TestFingerprintCacheGetBaseline:

    async def test_lrange_called_with_correct_key(self):
        redis = _make_redis()
        cache = FingerprintCache(redis_client=redis)
        await cache.get_baseline(TENANT, PERSONA)
        redis.client.lrange.assert_awaited_once_with(
            _expected_key(TENANT, PERSONA), 0, -1
        )

    async def test_returns_required_keys(self):
        redis = _make_redis()
        cache = FingerprintCache(redis_client=redis)
        result = await cache.get_baseline(TENANT, PERSONA)
        assert "fingerprints" in result
        assert "sample_count" in result
        assert "frequency_map" in result

    async def test_empty_store_returns_zero_sample_count(self):
        redis = _make_redis(lrange_return=[])
        cache = FingerprintCache(redis_client=redis)
        result = await cache.get_baseline(TENANT, PERSONA)
        assert result["sample_count"] == 0
        assert result["fingerprints"] == []
        assert result["frequency_map"] == {}

    async def test_fingerprints_decoded_from_bytes(self):
        redis = _make_redis(lrange_return=[b"fp-aaa", b"fp-bbb"])
        cache = FingerprintCache(redis_client=redis)
        result = await cache.get_baseline(TENANT, PERSONA)
        assert result["fingerprints"] == ["fp-aaa", "fp-bbb"]
        assert all(isinstance(fp, str) for fp in result["fingerprints"])

    async def test_sample_count_matches_list_length(self):
        fps = [b"fp-aaa", b"fp-bbb", b"fp-aaa", b"fp-ccc"]
        redis = _make_redis(lrange_return=fps)
        cache = FingerprintCache(redis_client=redis)
        result = await cache.get_baseline(TENANT, PERSONA)
        assert result["sample_count"] == 4

    async def test_frequency_map_counts_correctly(self):
        fps = [b"fp-aaa", b"fp-bbb", b"fp-aaa", b"fp-aaa", b"fp-bbb"]
        redis = _make_redis(lrange_return=fps)
        cache = FingerprintCache(redis_client=redis)
        result = await cache.get_baseline(TENANT, PERSONA)
        assert result["frequency_map"]["fp-aaa"] == 3
        assert result["frequency_map"]["fp-bbb"] == 2

    async def test_all_unique_fingerprints_have_count_one(self):
        fps = [b"fp-x", b"fp-y", b"fp-z"]
        redis = _make_redis(lrange_return=fps)
        cache = FingerprintCache(redis_client=redis)
        result = await cache.get_baseline(TENANT, PERSONA)
        assert all(v == 1 for v in result["frequency_map"].values())

    async def test_frequency_map_keys_match_fingerprints(self):
        fps = [b"fp-aaa", b"fp-bbb", b"fp-aaa"]
        redis = _make_redis(lrange_return=fps)
        cache = FingerprintCache(redis_client=redis)
        result = await cache.get_baseline(TENANT, PERSONA)
        assert set(result["frequency_map"].keys()) == {"fp-aaa", "fp-bbb"}

    async def test_single_fingerprint(self):
        redis = _make_redis(lrange_return=[b"fp-only"])
        cache = FingerprintCache(redis_client=redis)
        result = await cache.get_baseline(TENANT, PERSONA)
        assert result["sample_count"] == 1
        assert result["fingerprints"] == ["fp-only"]
        assert result["frequency_map"] == {"fp-only": 1}

    async def test_string_items_not_double_decoded(self):
        """Handle case where Redis returns str instead of bytes."""
        redis = _make_redis(lrange_return=["fp-str", "fp-str"])
        cache = FingerprintCache(redis_client=redis)
        result = await cache.get_baseline(TENANT, PERSONA)
        assert result["fingerprints"] == ["fp-str", "fp-str"]
        assert result["frequency_map"]["fp-str"] == 2
