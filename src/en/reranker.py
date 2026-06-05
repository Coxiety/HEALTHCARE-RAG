"""Reranker using a configurable cross-encoder checkpoint.

Usage in pipeline:
    reranker = Reranker()
    top_chunks = reranker.rerank(query, chunks, top_k=3)
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.database.vector_store import RetrievedChunk


class Reranker:
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name_or_path: str | None = None):
        self.model_name_or_path = model_name_or_path or self.MODEL_NAME
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name_or_path)
        return self._model

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 3
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        model = self._get_model()
        pairs = [(query, c.text) for c in chunks]
        scores = model.predict(pairs)

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(text=c.text, source=c.source, score=float(s))
            for s, c in ranked[:top_k]
        ]
