"""Immutable audit log. Append-only. All seals go here.

In-memory store is always maintained. If a repository (DB) is available,
seals are also persisted.
"""

from typing import Optional

from nexus.types import Seal
from nexus.core.notary import Notary


class Ledger:
    """Immutable audit log. Append-only. All seals go here."""

    def __init__(self, repository=None):
        """
        Args:
            repository: Injected DB repository for persistence (Phase 5).
                        Can be None for in-memory only mode.
        """
        self._memory_store: list[Seal] = []
        self._repository = repository

    async def append(self, seal: Seal) -> None:
        """Append seal. If repository available, persist. Always keep in memory.

        Args:
            seal: The finalized seal to append
        """
        self._memory_store.append(seal)
        if self._repository is not None:
            await self._repository.create_seal(seal)

    async def get_chain(self, chain_id: str) -> list[Seal]:
        """Get all seals for a chain, ordered by step_index.

        Args:
            chain_id: The chain to retrieve seals for

        Returns:
            Ordered list of seals
        """
        if self._repository is not None:
            db_seals = await self._repository.get_chain_seals(chain_id)
            return sorted(
                [Seal(**s.__dict__) if not isinstance(s, Seal) else s for s in db_seals],
                key=lambda s: s.step_index,
            )
        return sorted(
            [s for s in self._memory_store if s.chain_id == chain_id],
            key=lambda s: s.step_index,
        )

    async def get_by_tenant(self, tenant_id: str, limit: int = 100, offset: int = 0) -> list[Seal]:
        """Paginated seal history for a tenant.

        Args:
            tenant_id: Tenant to filter by
            limit: Max results
            offset: Pagination offset

        Returns:
            List of seals for this tenant
        """
        if self._repository is not None:
            db_seals = await self._repository.list_seals(tenant_id, limit=limit, offset=offset)
            return [Seal(**s.__dict__) if not isinstance(s, Seal) else s for s in db_seals]
        tenant_seals = [s for s in self._memory_store if s.tenant_id == tenant_id]
        return tenant_seals[offset: offset + limit]

    async def verify_integrity(self, chain_id: str) -> bool:
        """Verify Merkle chain integrity for all seals in a chain.

        Args:
            chain_id: The chain to verify

        Returns:
            True if chain is intact
        """
        seals = await self.get_chain(chain_id)
        notary = Notary()
        return notary.verify_chain(seals)
