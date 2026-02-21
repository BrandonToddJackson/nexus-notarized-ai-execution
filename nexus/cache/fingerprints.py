"""Stores behavioral fingerprints for Gate 4 drift detection."""

import json
import re
from typing import Optional

from nexus.cache.redis_client import RedisClient

_SAFE_ID = re.compile(r'[^a-zA-Z0-9_-]')


class FingerprintCache:
    """Stores behavioral fingerprints for Gate 4 drift detection."""

    def __init__(self, redis_client: RedisClient):
        """
        Args:
            redis_client: Redis client for persistence
        """
        self.redis = redis_client

    async def store(self, tenant_id: str, persona_id: str, fingerprint: str) -> None:
        """Append fingerprint to persona's history (max 1000).

        Args:
            tenant_id: Tenant scope
            persona_id: Persona this fingerprint belongs to
            fingerprint: The behavioral fingerprint hash
        """
        safe_pid = _SAFE_ID.sub("", persona_id)[:50]
        key = self.redis._key(tenant_id, f"fingerprints:{safe_pid}")
        await self.redis.client.rpush(key, fingerprint)
        await self.redis.client.ltrim(key, -1000, -1)

    async def get_baseline(self, tenant_id: str, persona_id: str) -> dict:
        """Return baseline statistics for drift comparison.

        Returns:
            {
                "fingerprints": list[str],
                "sample_count": int,
                "frequency_map": dict[str, int]  # fingerprint -> count
            }
        """
        safe_pid = _SAFE_ID.sub("", persona_id)[:50]
        key = self.redis._key(tenant_id, f"fingerprints:{safe_pid}")
        raw = await self.redis.client.lrange(key, 0, -1)
        fingerprints = [item.decode() if isinstance(item, bytes) else item for item in raw]

        frequency_map: dict[str, int] = {}
        for fp in fingerprints:
            frequency_map[fp] = frequency_map.get(fp, 0) + 1

        return {
            "fingerprints": fingerprints,
            "sample_count": len(fingerprints),
            "frequency_map": frequency_map,
        }
