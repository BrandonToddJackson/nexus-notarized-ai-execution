"""Tenant-namespaced Redis operations."""

from typing import Optional

import redis.asyncio as redis
from nexus.config import config


class RedisClient:
    """Tenant-namespaced Redis operations."""

    def __init__(self):
        self.pool = redis.ConnectionPool.from_url(config.redis_url)
        self.client = redis.Redis(connection_pool=self.pool)

    def _key(self, tenant_id: str, key: str) -> str:
        """Generate namespaced key."""
        return f"nexus:{tenant_id}:{key}"

    async def get(self, tenant_id: str, key: str) -> Optional[str]:
        """Get a value."""
        result = await self.client.get(self._key(tenant_id, key))
        return result.decode() if result else None

    async def set(self, tenant_id: str, key: str, value: str, ttl: int = None) -> None:
        """Set a value with optional TTL."""
        k = self._key(tenant_id, key)
        if ttl:
            await self.client.setex(k, ttl, value)
        else:
            await self.client.set(k, value)

    async def delete(self, tenant_id: str, key: str) -> None:
        """Delete a key."""
        await self.client.delete(self._key(tenant_id, key))

    async def incr(self, tenant_id: str, key: str) -> int:
        """Increment a counter."""
        return await self.client.incr(self._key(tenant_id, key))

    async def health(self) -> bool:
        """Check Redis connectivity."""
        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    async def close(self):
        """Close connections."""
        await self.client.close()
