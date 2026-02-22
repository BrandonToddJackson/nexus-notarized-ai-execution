"""NEXUS Python SDK — thin async client for the NEXUS API.

Usage:
    import asyncio
    from nexus_client import NexusClient

    async def main():
        client = NexusClient(base_url="http://localhost:8000", api_key="nxs_demo_key_12345")
        async with client:
            result = await client.execute("What is NEXUS?")
            print(result["status"])

    asyncio.run(main())

Streaming:
    async for event in client.stream("Research AI safety"):
        print(event)
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx


class NexusClientError(Exception):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class NexusClient:
    """Async HTTP client for the NEXUS API.

    Args:
        base_url: API base URL, e.g. "http://localhost:8000"
        api_key:  NEXUS API key (prefix "nxs_")
        timeout:  Request timeout in seconds (default 60)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._http: Optional[httpx.AsyncClient] = None
        self._token: Optional[str] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def __aenter__(self) -> "NexusClient":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def connect(self) -> None:
        """Open HTTP session and authenticate."""
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
        )
        if self._api_key:
            await self._authenticate()

    async def close(self) -> None:
        """Close HTTP session."""
        if self._http:
            await self._http.aclose()

    # ── Auth ───────────────────────────────────────────────────────────────

    async def _authenticate(self) -> None:
        """Exchange API key for JWT token."""
        resp = await self._http.post(
            "/v1/auth/token",
            json={"api_key": self._api_key},
        )
        self._raise_for_status(resp)
        self._token = resp.json()["access_token"]

    def _headers(self) -> dict[str, str]:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    # ── Execute ────────────────────────────────────────────────────────────

    async def execute(
        self,
        task: str,
        persona: Optional[str] = None,
        tenant_id: str = "demo",
    ) -> dict[str, Any]:
        """Execute a task synchronously. Returns the completed chain result.

        Args:
            task:      Natural language task description
            persona:   Optional persona name to pin (e.g., "researcher")
            tenant_id: Tenant context (default "demo")

        Returns:
            dict with keys: chain_id, status, seals, cost_usd, ...
        """
        payload: dict[str, Any] = {"task": task, "tenant_id": tenant_id}
        if persona:
            payload["persona"] = persona

        resp = await self._http.post(
            "/v1/execute",
            json=payload,
            headers=self._headers(),
        )
        self._raise_for_status(resp)
        return resp.json()

    async def stream(
        self,
        task: str,
        persona: Optional[str] = None,
        tenant_id: str = "demo",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a task execution via SSE. Yields parsed event dicts.

        Each event has:
          - event: "chain_started" | "step_started" | "gate_result" |
                   "seal_created" | "step_completed" | "chain_completed"
          - data:  event-specific payload dict

        Usage:
            async for event in client.stream("my task"):
                print(event["event"], event.get("data", {}))
        """
        payload: dict[str, Any] = {"task": task, "tenant_id": tenant_id}
        if persona:
            payload["persona"] = persona

        async with self._http.stream(
            "POST",
            "/v1/execute/stream",
            json=payload,
            headers={**self._headers(), "Accept": "text/event-stream"},
        ) as resp:
            self._raise_for_status(resp)
            event_type = None
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    raw = line[len("data:"):].strip()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = {"raw": raw}
                    yield {"event": event_type, "data": data}
                    event_type = None

    # ── Ledger ─────────────────────────────────────────────────────────────

    async def ledger(
        self,
        tenant_id: str = "demo",
        chain_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Fetch seals from the immutable ledger.

        Args:
            tenant_id:  Tenant scope
            chain_id:   Filter by chain
            persona_id: Filter by persona
            limit:      Max records (default 50)
            offset:     Pagination offset

        Returns:
            List of seal dicts
        """
        params: dict[str, Any] = {
            "tenant_id": tenant_id,
            "limit": limit,
            "offset": offset,
        }
        if chain_id:
            params["chain_id"] = chain_id
        if persona_id:
            params["persona_id"] = persona_id

        resp = await self._http.get(
            "/v1/ledger",
            params=params,
            headers=self._headers(),
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Personas ───────────────────────────────────────────────────────────

    async def list_personas(self, tenant_id: str = "demo") -> list[dict[str, Any]]:
        """List available personas for a tenant."""
        resp = await self._http.get(
            "/v1/personas",
            params={"tenant_id": tenant_id},
            headers=self._headers(),
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Knowledge ──────────────────────────────────────────────────────────

    async def upload_document(
        self,
        content: str,
        namespace: str = "default",
        source: str = "sdk_upload",
        tenant_id: str = "demo",
        access_level: str = "internal",
    ) -> dict[str, Any]:
        """Upload a document to the knowledge base.

        Args:
            content:      Document text
            namespace:    Logical grouping (e.g., "product_docs")
            source:       Source label (filename or URL)
            tenant_id:    Tenant scope
            access_level: "public" | "internal" | "restricted" | "confidential"

        Returns:
            Created document metadata
        """
        resp = await self._http.post(
            "/v1/knowledge",
            json={
                "content": content,
                "namespace": namespace,
                "source": source,
                "tenant_id": tenant_id,
                "access_level": access_level,
            },
            headers=self._headers(),
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Health ─────────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Check API health."""
        resp = await self._http.get("/v1/health")
        self._raise_for_status(resp)
        return resp.json()

    # ── Internal ───────────────────────────────────────────────────────────

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise NexusClientError(resp.status_code, detail)
