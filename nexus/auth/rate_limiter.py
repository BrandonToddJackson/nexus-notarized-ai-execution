"""Redis-backed rate limiting."""

from nexus.cache.redis_client import RedisClient
from nexus.config import config
from nexus.exceptions import NexusError


class RateLimiter:
    """Redis-backed rate limiting."""

    def __init__(self, redis_client: RedisClient):
        """
        Args:
            redis_client: Redis client for counters
        """
        self.redis = redis_client

    async def check(self, tenant_id: str, action: str = "api") -> bool:
        """Check rate limit.

        Args:
            tenant_id: Tenant to check
            action: "api" (per minute) or "chain" (per hour)

        Returns:
            True if allowed

        Raises:
            NexusError: If rate limit exceeded
        """
        # TODO: Implement — increment Redis counter with TTL,
        # check against config limits
        raise NotImplementedError("Phase 8: Implement rate limiting")
