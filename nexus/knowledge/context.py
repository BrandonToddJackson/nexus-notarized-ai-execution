"""Assembles retrieval context for each action.

This is what makes the agent useful — it combines RAG results, session history,
and persona constraints into a coherent context for tool selection.
"""

from typing import Optional

from nexus.types import PersonaContract, RetrievedContext
from nexus.knowledge.store import KnowledgeStore


class ContextBuilder:
    """Assembles retrieval context for each action."""

    def __init__(self, knowledge_store: KnowledgeStore):
        """
        Args:
            knowledge_store: The tenant-scoped vector store
        """
        self.knowledge_store = knowledge_store

    async def build(
        self,
        tenant_id: str,
        task: str,
        persona: PersonaContract,
        session_history: list[dict] = None,
    ) -> RetrievedContext:
        """Build context for an action.

        Steps:
        1. Determine which namespaces this persona can access (from resource_scopes)
        2. Query knowledge store across allowed namespaces
        3. Merge results, deduplicate, rank by relevance
        4. Include session_history (previous chain step results) for continuity
        5. Return assembled context with confidence score

        Args:
            tenant_id: Tenant scope
            task: The task or step description to retrieve context for
            persona: Active persona (used to filter accessible namespaces)
            session_history: Previous chain step results for continuity

        Returns:
            RetrievedContext with merged, ranked results
        """
        # Step 1: Extract KB namespaces from persona's resource_scopes.
        # Scopes like "kb:docs" -> namespace "docs"
        # Scopes like "kb:*"   -> query all namespaces for this tenant
        kb_scopes = [s for s in persona.resource_scopes if s.startswith("kb:")]
        namespaces: list[str] = []

        has_wildcard = any(s == "kb:*" for s in kb_scopes)
        if has_wildcard:
            try:
                namespaces = self.knowledge_store.list_namespaces(tenant_id)
            except Exception:
                namespaces = []
        else:
            for scope in kb_scopes:
                ns = scope[3:]  # strip "kb:"
                if ns and ns != "*" and ns not in namespaces:
                    namespaces.append(ns)

        # Fallback: if no specific namespaces resolved, try "default"
        if not namespaces:
            namespaces = ["default"]

        # Step 2: Query each namespace and collect results
        all_documents: list[dict] = []
        all_sources: list[str] = []
        seen_sources: set[str] = set()

        for ns in namespaces:
            try:
                ctx = await self.knowledge_store.query(
                    tenant_id=tenant_id,
                    namespace=ns,
                    query=task,
                    n_results=5,
                )
            except Exception:
                continue
            all_documents.extend(ctx.documents)
            for src in ctx.sources:
                if src not in seen_sources:
                    all_sources.append(src)
                    seen_sources.add(src)

        # Step 3: Deduplicate by source+chunk_index, rank by score descending
        seen_chunks: set[str] = set()
        unique_docs: list[dict] = []
        for doc in all_documents:
            key = f"{doc.get('document_id', '')}:{doc.get('chunk_index', 0)}"
            if key not in seen_chunks:
                seen_chunks.add(key)
                unique_docs.append(doc)

        unique_docs.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        top_docs = unique_docs[:10]  # keep top 10 across all namespaces

        # Step 4: Prepend session history as synthetic context entries
        if session_history:
            for entry in session_history[-3:]:  # last 3 steps for recency
                result_str = str(entry.get("result", ""))[:300]
                if result_str:
                    top_docs.insert(0, {
                        "content": f"[Previous step result] {result_str}",
                        "score": 0.5,
                        "source": "session_history",
                        "document_id": "",
                        "chunk_index": 0,
                        "access_level": "internal",
                    })

        # Step 5: Compute aggregate confidence
        kb_scores = [d["score"] for d in top_docs if d.get("source") != "session_history"]
        confidence = sum(kb_scores) / len(kb_scores) if kb_scores else 0.0

        primary_ns = namespaces[0] if namespaces else "default"
        return RetrievedContext(
            query=task,
            documents=top_docs,
            confidence=confidence,
            sources=all_sources,
            namespace=primary_ns,
        )
