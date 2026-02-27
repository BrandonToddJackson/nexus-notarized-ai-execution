"""RAGAnything adapter — bridges NEXUS LLM/embedding clients to RAGAnything."""

import logging
import os
import tempfile

logger = logging.getLogger(__name__)

try:
    from raganything import RAGAnything
    RAGANYTHING_AVAILABLE = True
except ImportError:
    RAGANYTHING_AVAILABLE = False

from nexus.exceptions import ToolError


class RAGAnythingAdapter:
    """Wraps RAGAnything with NEXUS llm_client and embedding_service bridges."""

    def __init__(self, config, llm_client, embedding_service):
        self._config = config
        self._llm_client = llm_client
        self._embedding_service = embedding_service

    def _get_rag(self, tenant_id: str, namespace: str):
        if not RAGANYTHING_AVAILABLE:
            raise ToolError("Install raganything: pip install raganything[all]")

        working_dir = os.path.join(self._config.rag_anything_dir, tenant_id, namespace)
        os.makedirs(working_dir, exist_ok=True)

        async def _llm_func(prompt, system_prompt=None, history_messages=None, **kw):
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt})
            result = await self._llm_client.complete(msgs)
            return result["content"]

        async def _embed_func(texts):
            return self._embedding_service.embed(texts)

        return RAGAnything(
            working_dir=working_dir,
            llm_model_func=_llm_func,
            embedding_func=_embed_func,
        )

    async def ingest(
        self,
        tenant_id: str,
        namespace: str,
        content: str | None = None,
        file_path: str | None = None,
        url: str | None = None,
    ) -> str:
        """Ingest content into RAG. Returns document_id."""
        if not RAGANYTHING_AVAILABLE:
            raise ToolError("Install raganything: pip install raganything[all]")

        rag = self._get_rag(tenant_id, namespace)
        tmp_path = None

        try:
            if url:
                import httpx
                async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    suffix = ".html"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(resp.content)
                        tmp_path = tmp.name
                target_path = tmp_path
            elif file_path:
                target_path = file_path
            elif content:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                target_path = tmp_path
            else:
                raise ToolError("Provide content, file_path, or url")

            import hashlib
            doc_id = hashlib.md5(f"{tenant_id}:{namespace}:{target_path}".encode()).hexdigest()

            await rag.process_document_complete(
                file_path=target_path,
                document_id=doc_id,
            )
            return doc_id

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def query(
        self,
        tenant_id: str,
        namespace: str,
        query: str,
        mode: str = "hybrid",
    ) -> str:
        """Query RAG. Returns formatted string of results."""
        if not RAGANYTHING_AVAILABLE:
            raise ToolError("Install raganything: pip install raganything[all]")

        rag = self._get_rag(tenant_id, namespace)
        result = await rag.aquery(query, mode=mode)
        if isinstance(result, str):
            return result
        return str(result)
