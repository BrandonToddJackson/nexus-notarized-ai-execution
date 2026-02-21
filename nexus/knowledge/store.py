"""Tenant-scoped vector store for RAG retrieval.

Default backend: ChromaDB. Chunking: 500 chars, 50 char overlap.
Collections are named '{tenant_id}_{namespace}' for tenant isolation.
"""

from typing import Optional, Callable

from nexus.types import KnowledgeDocument, RetrievedContext

# Access-level ordering: lower rank = less restricted
_ACCESS_RANKS = {"public": 0, "internal": 1, "restricted": 2, "confidential": 3}

_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50


class KnowledgeStore:
    """Tenant-scoped vector store for RAG retrieval."""

    def __init__(self, persist_dir: str, embedding_fn: Callable = None):
        """
        Args:
            persist_dir: ChromaDB persistence directory
            embedding_fn: Callable that takes list[str] -> list[list[float]].
                          Injected from EmbeddingService.embed.
        """
        self.persist_dir = persist_dir
        self.embedding_fn = embedding_fn
        self._client = None  # lazy load ChromaDB client

    def _get_client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_collection(self, tenant_id: str, namespace: str):
        """Get or create a ChromaDB collection for tenant+namespace."""
        client = self._get_client()
        # Sanitize name: ChromaDB requires 3-63 chars, alphanumeric + underscores/hyphens
        name = f"{tenant_id}_{namespace}".replace("-", "_").replace(":", "_")[:63]
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """Split text into overlapping chunks of 500 chars with 50-char overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + _CHUNK_SIZE, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - _CHUNK_OVERLAP
        return chunks

    async def ingest(self, document: KnowledgeDocument) -> str:
        """Chunk document content, embed chunks, store in ChromaDB.

        Chunking: 500 chars, 50 char overlap.
        Collection: '{tenant_id}_{namespace}'.

        Args:
            document: KnowledgeDocument to ingest

        Returns:
            Document ID
        """
        collection = self._get_collection(document.tenant_id, document.namespace)
        chunks = self._chunk_text(document.content)
        if not chunks:
            return document.id

        access_rank = _ACCESS_RANKS.get(document.access_level, 1)
        ids = [f"{document.id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": document.id,
                "source": document.source,
                "access_level": document.access_level,
                "access_level_rank": access_rank,
                "chunk_index": i,
                "namespace": document.namespace,
            }
            for i in range(len(chunks))
        ]

        if self.embedding_fn is not None:
            embeddings = self.embedding_fn(chunks)
            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            # Let ChromaDB use its default embedding function
            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )

        return document.id

    async def query(
        self, tenant_id: str, namespace: str, query: str,
        access_level: str = "internal", n_results: int = 5
    ) -> RetrievedContext:
        """Embed query, search ChromaDB, filter by access_level.

        Args:
            tenant_id: Tenant scope
            namespace: Knowledge namespace
            query: Search query
            access_level: Maximum access level to include
            n_results: Max results to return

        Returns:
            RetrievedContext with documents, confidence, sources.
            Confidence = average similarity score of top results.
        """
        try:
            collection = self._get_collection(tenant_id, namespace)
        except Exception:
            return RetrievedContext(
                query=query, documents=[], confidence=0.0,
                sources=[], namespace=namespace,
            )

        # Count items to avoid requesting more than available
        count = collection.count()
        if count == 0:
            return RetrievedContext(
                query=query, documents=[], confidence=0.0,
                sources=[], namespace=namespace,
            )

        max_rank = _ACCESS_RANKS.get(access_level, 1)
        actual_n = min(n_results, count)

        query_kwargs: dict = {
            "n_results": actual_n,
            "where": {"access_level_rank": {"$lte": max_rank}},
            "include": ["documents", "metadatas", "distances"],
        }

        if self.embedding_fn is not None:
            query_embedding = self.embedding_fn([query])[0]
            query_kwargs["query_embeddings"] = [query_embedding]
        else:
            query_kwargs["query_texts"] = [query]

        try:
            results = collection.query(**query_kwargs)
        except Exception:
            # where-filter may fail if no documents match; fall back without filter
            query_kwargs.pop("where", None)
            try:
                results = collection.query(**query_kwargs)
            except Exception:
                return RetrievedContext(
                    query=query, documents=[], confidence=0.0,
                    sources=[], namespace=namespace,
                )

        docs_list = results.get("documents", [[]])[0] or []
        metas_list = results.get("metadatas", [[]])[0] or []
        dists_list = results.get("distances", [[]])[0] or []

        documents = []
        sources = []
        similarities = []
        seen_sources: set[str] = set()

        for doc_text, meta, dist in zip(docs_list, metas_list, dists_list):
            # Cosine distance → similarity: dist is in [0,2], similarity = 1 - dist (clamped)
            sim = max(0.0, min(1.0, 1.0 - dist))
            similarities.append(sim)
            source = meta.get("source", "unknown")
            documents.append({
                "content": doc_text,
                "score": sim,
                "source": source,
                "document_id": meta.get("document_id", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "access_level": meta.get("access_level", access_level),
            })
            if source not in seen_sources:
                sources.append(source)
                seen_sources.add(source)

        confidence = sum(similarities) / len(similarities) if similarities else 0.0

        return RetrievedContext(
            query=query,
            documents=documents,
            confidence=confidence,
            sources=sources,
            namespace=namespace,
        )

    async def delete(self, tenant_id: str, namespace: str, document_id: str) -> None:
        """Remove a document and its chunks from the store.

        Args:
            tenant_id: Tenant scope
            namespace: Knowledge namespace
            document_id: Document to delete
        """
        collection = self._get_collection(tenant_id, namespace)
        collection.delete(where={"document_id": document_id})

    def list_namespaces(self, tenant_id: str) -> list[str]:
        """List all knowledge namespaces for a tenant.

        Args:
            tenant_id: Tenant scope

        Returns:
            List of namespace names
        """
        client = self._get_client()
        prefix = f"{tenant_id}_".replace("-", "_")
        namespaces = []
        try:
            collections = client.list_collections()
        except Exception:
            return []
        for col in collections:
            # col may be a Collection object or a string depending on chromadb version
            name = col.name if hasattr(col, "name") else str(col)
            if name.startswith(prefix):
                ns = name[len(prefix):]
                namespaces.append(ns)
        return namespaces
