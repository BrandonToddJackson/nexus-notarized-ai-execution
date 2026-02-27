"""Knowledge base CRUD: ingest docs, query, manage namespaces."""

from fastapi import APIRouter, Request, Form, File, UploadFile
from nexus.api.schemas import IngestRequest
from nexus.types import KnowledgeDocument
import tempfile
import os

router = APIRouter(tags=["knowledge"])


@router.post("/knowledge/ingest")
async def ingest_document(request: Request, body: IngestRequest):
    """Upload and ingest a document into the knowledge base."""
    tenant_id = getattr(request.state, "tenant_id", "demo")
    knowledge_store = request.app.state.knowledge_store

    doc = KnowledgeDocument(
        tenant_id=tenant_id,
        namespace=body.namespace,
        source=body.source,
        content=body.content,
        access_level=body.access_level,
        metadata=body.metadata,
    )
    doc_id = await knowledge_store.ingest(doc)
    return {"document_id": doc_id, "status": "ingested", "namespace": body.namespace}


@router.get("/knowledge/query")
async def query_knowledge(
    request: Request,
    query: str,
    namespace: str = "default",
    n_results: int = 5,
):
    """Query the knowledge base."""
    tenant_id = getattr(request.state, "tenant_id", "demo")
    knowledge_store = request.app.state.knowledge_store

    context = await knowledge_store.query(
        tenant_id=tenant_id,
        namespace=namespace,
        query=query,
        n_results=n_results,
    )
    return {
        "results": context.documents,
        "confidence": context.confidence,
        "sources": context.sources,
    }


@router.get("/knowledge/namespaces")
async def list_namespaces(request: Request):
    """List all knowledge namespaces for the tenant."""
    tenant_id = getattr(request.state, "tenant_id", "demo")
    knowledge_store = request.app.state.knowledge_store
    namespaces = knowledge_store.list_namespaces(tenant_id)
    return {"namespaces": namespaces}


@router.post("/knowledge/multimodal")
async def ingest_multimodal(
    request: Request,
    content: str = Form(""),
    url: str = Form(""),
    namespace: str = Form("default"),
    file: UploadFile = File(None),
):
    """Ingest multimodal content (text, URL, or file) via RAGAnything."""
    from fastapi import HTTPException
    tenant_id = getattr(request.state, "tenant_id", "demo")
    rag_adapter = getattr(request.app.state, "rag_adapter", None)
    if rag_adapter is None:
        raise HTTPException(
            status_code=503,
            detail="RAG-Anything not enabled. Set NEXUS_RAG_ANYTHING_ENABLED=true.",
        )

    file_path = None
    try:
        if file is not None:
            suffix = os.path.splitext(file.filename or "upload")[1] or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                file_path = tmp.name

        doc_id = await rag_adapter.ingest(
            tenant_id=tenant_id,
            namespace=namespace,
            content=content or None,
            file_path=file_path,
            url=url or None,
        )
    finally:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)

    return {"document_id": doc_id, "status": "ingested", "namespace": namespace}
