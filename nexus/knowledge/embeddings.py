"""Unified embedding service. Used by RAG AND Gate 2 intent matching.

Uses sentence-transformers for local embeddings. No API key needed.
"""

import numpy as np


class EmbeddingService:
    """Unified embedding service. Used by RAG AND Gate 2 intent matching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load sentence-transformers model.

        Args:
            model_name: HuggingFace model name for sentence-transformers.
                        Default "all-MiniLM-L6-v2" is small, fast, 384 dimensions.
        """
        self.model_name = model_name
        self._model = None  # lazy load

    def _load_model(self):
        """Lazy-load the model on first use."""
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def _ensure_model(self):
        if self._model is None:
            self._load_model()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of vectors.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is list[float])
        """
        self._ensure_model()
        assert self._model is not None
        return self._model.encode(texts).tolist()

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts. Returns 0.0-1.0.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Cosine similarity score
        """
        vecs = self.embed([text_a, text_b])
        a = np.array(vecs[0])
        b = np.array(vecs[1])
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def similarities(self, query: str, candidates: list[str]) -> list[float]:
        """Cosine similarity of query against each candidate.

        Args:
            query: Query text
            candidates: List of candidate texts

        Returns:
            List of similarity scores, one per candidate
        """
        if not candidates:
            return []
        all_vecs = self.embed([query] + candidates)
        q = np.array(all_vecs[0])
        q_norm = np.linalg.norm(q)
        results = []
        for vec in all_vecs[1:]:
            c = np.array(vec)
            denom = q_norm * np.linalg.norm(c)
            if denom == 0:
                results.append(0.0)
            else:
                results.append(float(np.dot(q, c) / denom))
        return results
