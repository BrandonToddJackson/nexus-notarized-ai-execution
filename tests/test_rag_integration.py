"""Tests for RAG-Anything integration — adapter, tools, and API endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─── Adapter unit tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_adapter_ingest_text():
    """Adapter ingests plain text content and returns a doc_id string."""
    mock_rag = AsyncMock()
    mock_rag.process_document_complete = AsyncMock(return_value=None)

    mock_config = MagicMock()
    mock_config.rag_anything_dir = "/tmp/nexus_rag_test"
    mock_llm = MagicMock()
    mock_embed = MagicMock()

    with patch("nexus.knowledge.rag_adapter.RAGANYTHING_AVAILABLE", True), \
         patch("nexus.knowledge.rag_adapter.RAGAnything", return_value=mock_rag, create=True):
        from nexus.knowledge.rag_adapter import RAGAnythingAdapter
        adapter = RAGAnythingAdapter(mock_config, mock_llm, mock_embed)
        doc_id = await adapter.ingest(
            tenant_id="t1", namespace="test", content="Target audience: SaaS founders"
        )

    assert isinstance(doc_id, str)
    assert len(doc_id) > 0
    mock_rag.process_document_complete.assert_called_once()


@pytest.mark.asyncio
async def test_rag_adapter_ingest_url():
    """Adapter fetches URL via httpx and passes temp file to RAGAnything."""
    mock_rag = AsyncMock()
    mock_rag.process_document_complete = AsyncMock(return_value=None)

    mock_config = MagicMock()
    mock_config.rag_anything_dir = "/tmp/nexus_rag_test"

    mock_response = MagicMock()
    mock_response.content = b"<html>Test page</html>"
    mock_response.raise_for_status = MagicMock()

    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=None)

    with patch("nexus.knowledge.rag_adapter.RAGANYTHING_AVAILABLE", True), \
         patch("nexus.knowledge.rag_adapter.RAGAnything", return_value=mock_rag, create=True), \
         patch("httpx.AsyncClient", return_value=mock_http_client):
        from importlib import reload
        import nexus.knowledge.rag_adapter as mod
        adapter = mod.RAGAnythingAdapter(mock_config, MagicMock(), MagicMock())
        doc_id = await adapter.ingest(
            tenant_id="t1", namespace="test", url="http://example.com/brief"
        )

    assert isinstance(doc_id, str)
    mock_http_client.get.assert_called_once_with("http://example.com/brief")


@pytest.mark.asyncio
async def test_rag_adapter_query():
    """Adapter queries RAGAnything and returns formatted string."""
    mock_rag = AsyncMock()
    mock_rag.aquery = AsyncMock(return_value="SaaS founders in 20-200 employee range")

    mock_config = MagicMock()
    mock_config.rag_anything_dir = "/tmp/nexus_rag_test"

    with patch("nexus.knowledge.rag_adapter.RAGANYTHING_AVAILABLE", True), \
         patch("nexus.knowledge.rag_adapter.RAGAnything", return_value=mock_rag, create=True):
        from nexus.knowledge.rag_adapter import RAGAnythingAdapter
        adapter = RAGAnythingAdapter(mock_config, MagicMock(), MagicMock())
        result = await adapter.query(tenant_id="t1", namespace="test", query="find target audience")

    assert "SaaS founders" in result
    mock_rag.aquery.assert_called_once_with("find target audience", mode="hybrid")


# ─── Tool tests ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_ingest_tool_returns_error_when_no_adapter():
    """rag_ingest returns helpful error when adapter is not configured."""
    # Reset adapter to None
    import nexus.tools.builtin.rag_tools as rag_mod
    original = rag_mod._rag_adapter
    rag_mod._rag_adapter = None
    try:
        result = await rag_mod.rag_ingest(content="test content", namespace="default")
        assert "not configured" in result.lower() or "NEXUS_RAG_ANYTHING_ENABLED" in result
    finally:
        rag_mod._rag_adapter = original


@pytest.mark.asyncio
async def test_rag_query_tool_returns_error_when_no_adapter():
    """rag_query returns helpful message when adapter is not configured."""
    import nexus.tools.builtin.rag_tools as rag_mod
    original = rag_mod._rag_adapter
    rag_mod._rag_adapter = None
    try:
        result = await rag_mod.rag_query(query="find leads", namespace="default")
        assert "not configured" in result.lower() or "find leads" in result
    finally:
        rag_mod._rag_adapter = original


# ─── API endpoint tests ────────────────────────────────────────────────────────

def _make_knowledge_test_app(rag_adapter=None):
    """Build minimal FastAPI app with knowledge router for testing."""
    from fastapi import FastAPI
    from nexus.api.routes.knowledge import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.rag_adapter = rag_adapter

    # Inject demo tenant for all requests
    from starlette.middleware.base import BaseHTTPMiddleware
    class TenantMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = "demo"
            return await call_next(request)

    app.add_middleware(TenantMiddleware)
    return app


def test_multimodal_ingest_endpoint_returns_doc_id():
    """POST /knowledge/multimodal with text content returns 200 + document_id."""
    from fastapi.testclient import TestClient

    mock_adapter = AsyncMock()
    mock_adapter.ingest = AsyncMock(return_value="abc123doc")

    app = _make_knowledge_test_app(rag_adapter=mock_adapter)
    client = TestClient(app)

    response = client.post(
        "/v1/knowledge/multimodal",
        data={"content": "Target: SaaS founders", "namespace": "cold_campaign"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "abc123doc"
    assert body["status"] == "ingested"
    assert body["namespace"] == "cold_campaign"


def test_rag_disabled_graceful():
    """POST /knowledge/multimodal returns 503 when rag_adapter is None."""
    from fastapi.testclient import TestClient

    app = _make_knowledge_test_app(rag_adapter=None)
    client = TestClient(app)

    response = client.post(
        "/v1/knowledge/multimodal",
        data={"content": "some content"},
    )
    assert response.status_code == 503
    assert "RAG-Anything not enabled" in response.json()["detail"]
