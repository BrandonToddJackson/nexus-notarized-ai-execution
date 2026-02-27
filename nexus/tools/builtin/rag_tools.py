"""Built-in RAG tools: ingest and query via RAGAnything."""

from nexus.tools.plugin import tool
from nexus.types import RiskLevel

# Module-level adapter reference — set at startup via set_rag_adapter()
_rag_adapter = None


def set_rag_adapter(adapter) -> None:
    """Wire the RAGAnythingAdapter at startup."""
    global _rag_adapter
    _rag_adapter = adapter


@tool(
    name="rag_ingest",
    description="Ingest content into RAG knowledge base",
    risk_level=RiskLevel.MEDIUM,
    resource_pattern="rag:*",
)
async def rag_ingest(content: str = "", url: str = "", namespace: str = "default") -> str:
    """Ingest text, URL, or file content into the RAG knowledge base.

    Args:
        content: Text content to ingest
        url: URL to fetch and ingest
        namespace: Knowledge namespace (default: "default")

    Returns:
        document_id of the ingested document
    """
    if _rag_adapter is None:
        return "RAG adapter not configured. Enable with NEXUS_RAG_ANYTHING_ENABLED=true."
    # tenant_id injected via execution context — use "default" as fallback
    doc_id = await _rag_adapter.ingest(
        tenant_id="default",
        namespace=namespace,
        content=content or None,
        url=url or None,
    )
    return f"Ingested document: {doc_id} (namespace: {namespace})"


@tool(
    name="rag_query",
    description="Query RAG knowledge base for campaign context",
    risk_level=RiskLevel.LOW,
    resource_pattern="rag:*",
)
async def rag_query(query: str, namespace: str = "default", mode: str = "hybrid") -> str:
    """Query the RAG knowledge base.

    Args:
        query: Search query
        namespace: Knowledge namespace to search
        mode: Query mode (hybrid, local, global, naive)

    Returns:
        Relevant knowledge as text
    """
    if _rag_adapter is None:
        return f"RAG adapter not configured. Query '{query}' would search namespace '{namespace}'."
    return await _rag_adapter.query(
        tenant_id="default",
        namespace=namespace,
        query=query,
        mode=mode,
    )
