"""Tests for Redis-backed RateLimiter — Gap 6.

Coverage:
  First request (count=1) → allowed + TTL set
  Subsequent request (count>1) → allowed + TTL NOT reset
  Exceeded limit → NexusError raised
  Chain action uses 3600s TTL window
  API action uses 60s TTL window
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.auth.rate_limiter import RateLimiter
from nexus.exceptions import NexusError


def _make_limiter(incr_return: int) -> tuple[RateLimiter, MagicMock]:
    """Build a RateLimiter backed by a mock Redis client."""
    mock_redis = MagicMock()
    mock_redis._key = MagicMock(return_value="nexus:tenant-1:rate:api")
    mock_redis.client = MagicMock()
    mock_redis.client.incr = AsyncMock(return_value=incr_return)
    mock_redis.client.expire = AsyncMock()
    return RateLimiter(mock_redis), mock_redis


class TestRateLimiterApiAction:

    async def test_first_request_allowed_and_sets_ttl(self):
        """count==1 → allowed, TTL set once."""
        with patch("nexus.auth.rate_limiter.config") as mock_cfg:
            mock_cfg.rate_limit_requests_per_minute = 60
            limiter, mock_redis = _make_limiter(incr_return=1)
            result = await limiter.check("tenant-1", "api")

        assert result is True
        mock_redis.client.expire.assert_awaited_once()

    async def test_subsequent_request_no_ttl_reset(self):
        """count>1 → allowed, TTL not touched again."""
        with patch("nexus.auth.rate_limiter.config") as mock_cfg:
            mock_cfg.rate_limit_requests_per_minute = 60
            limiter, mock_redis = _make_limiter(incr_return=5)
            result = await limiter.check("tenant-1", "api")

        assert result is True
        mock_redis.client.expire.assert_not_awaited()

    async def test_exceeded_limit_raises_nexus_error(self):
        """count > limit → NexusError with 'Rate limit exceeded'."""
        with patch("nexus.auth.rate_limiter.config") as mock_cfg:
            mock_cfg.rate_limit_requests_per_minute = 60
            # incr returns 61 which is > 60
            limiter, _ = _make_limiter(incr_return=61)
            with pytest.raises(NexusError, match="Rate limit exceeded"):
                await limiter.check("tenant-1", "api")

    async def test_exactly_at_limit_is_allowed(self):
        """count == limit → still allowed (only count > limit raises)."""
        with patch("nexus.auth.rate_limiter.config") as mock_cfg:
            mock_cfg.rate_limit_requests_per_minute = 60
            limiter, _ = _make_limiter(incr_return=60)  # exactly at limit
            result = await limiter.check("tenant-1", "api")

        assert result is True

    async def test_api_action_uses_60s_ttl(self):
        """API rate window is 60 seconds (per minute)."""
        with patch("nexus.auth.rate_limiter.config") as mock_cfg:
            mock_cfg.rate_limit_requests_per_minute = 60
            limiter, mock_redis = _make_limiter(incr_return=1)
            await limiter.check("tenant-1", "api")

        expire_args = mock_redis.client.expire.call_args
        # Second positional arg (or 'time' kwarg) is the TTL seconds
        ttl = expire_args.args[1] if expire_args.args else expire_args.kwargs.get("time")
        assert ttl == 60


class TestRateLimiterChainAction:

    async def test_chain_action_uses_separate_key(self):
        """Chain action uses 'rate:chain' sub-key, not 'rate:api'."""
        with patch("nexus.auth.rate_limiter.config") as mock_cfg:
            mock_cfg.rate_limit_chains_per_hour = 100
            mock_redis = MagicMock()
            mock_redis._key = MagicMock(return_value="nexus:tenant-1:rate:chain")
            mock_redis.client = MagicMock()
            mock_redis.client.incr = AsyncMock(return_value=1)
            mock_redis.client.expire = AsyncMock()
            limiter = RateLimiter(mock_redis)
            await limiter.check("tenant-1", "chain")

        # Verify _key was called with the chain sub-key
        mock_redis._key.assert_called_once_with("tenant-1", "rate:chain")

    async def test_chain_action_uses_3600s_ttl(self):
        """Chain rate window is 3600 seconds (per hour)."""
        with patch("nexus.auth.rate_limiter.config") as mock_cfg:
            mock_cfg.rate_limit_chains_per_hour = 100
            mock_redis = MagicMock()
            mock_redis._key = MagicMock(return_value="nexus:t:rate:chain")
            mock_redis.client = MagicMock()
            mock_redis.client.incr = AsyncMock(return_value=1)
            mock_redis.client.expire = AsyncMock()
            limiter = RateLimiter(mock_redis)
            await limiter.check("tenant-1", "chain")

        expire_args = mock_redis.client.expire.call_args
        ttl = expire_args.args[1] if expire_args.args else expire_args.kwargs.get("time")
        assert ttl == 3600

    async def test_chain_exceeded_raises(self):
        with patch("nexus.auth.rate_limiter.config") as mock_cfg:
            mock_cfg.rate_limit_chains_per_hour = 100
            mock_redis = MagicMock()
            mock_redis._key = MagicMock(return_value="nexus:t:rate:chain")
            mock_redis.client = MagicMock()
            mock_redis.client.incr = AsyncMock(return_value=101)
            mock_redis.client.expire = AsyncMock()
            limiter = RateLimiter(mock_redis)
            with pytest.raises(NexusError, match="Rate limit exceeded"):
                await limiter.check("tenant-1", "chain")
