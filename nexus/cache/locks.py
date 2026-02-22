"""Redis-based distributed locks for ledger write ordering.

These locks guard against concurrent ledger writes when running multiple
uvicorn worker processes (e.g., ``uvicorn --workers 4``). They are NOT
needed for single-process deployments — Python's asyncio event loop and
PostgreSQL's ACID guarantees handle concurrency within a single process.

Typical usage in a multi-worker route handler:

    from nexus.cache.locks import chain_lock

    async with chain_lock(redis_client, tenant_id, chain.id):
        await ledger.append(seal)
"""

import asyncio
from contextlib import asynccontextmanager

from nexus.cache.redis_client import RedisClient


class DistributedLock:
    """Redis-based distributed lock for a single resource."""

    def __init__(
        self,
        redis_client: RedisClient,
        tenant_id: str,
        resource: str,
        ttl_seconds: int = 30,
        retry_interval: float = 0.05,
        max_retries: int = 100,
    ):
        self.redis = redis_client
        self.tenant_id = tenant_id
        self.resource = resource
        self.ttl_seconds = ttl_seconds
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self._key = redis_client._key(tenant_id, f"lock:{resource}")

    async def acquire(self) -> bool:
        """Try to acquire the lock. Returns True if acquired."""
        result = await self.redis.client.set(
            self._key,
            "1",
            nx=True,
            ex=self.ttl_seconds,
        )
        return bool(result)

    async def release(self) -> None:
        """Release the lock."""
        await self.redis.client.delete(self._key)

    async def acquire_with_retry(self) -> bool:
        """Spin-wait until lock is acquired or max_retries exceeded."""
        for _ in range(self.max_retries):
            if await self.acquire():
                return True
            await asyncio.sleep(self.retry_interval)
        return False


@asynccontextmanager
async def chain_lock(
    redis_client: RedisClient,
    tenant_id: str,
    chain_id: str,
    ttl_seconds: int = 30,
):
    """Context manager that holds a distributed lock for a chain's ledger writes.

    Usage:
        async with chain_lock(redis, tenant_id, chain_id):
            await ledger.append(seal)
    """
    lock = DistributedLock(
        redis_client=redis_client,
        tenant_id=tenant_id,
        resource=f"chain:{chain_id}",
        ttl_seconds=ttl_seconds,
    )
    acquired = await lock.acquire_with_retry()
    if not acquired:
        raise TimeoutError(f"Could not acquire lock for chain {chain_id}")
    try:
        yield lock
    finally:
        await lock.release()
