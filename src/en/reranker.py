"""Reranker using cross-encoder/ms-marco-MiniLM-L-6-v2 (pretrained, no fine-tune).

Usage in pipeline:
    reranker = Reranker()
    top_chunks = reranker.rerank(query, chunks, top_k=3)
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.database.vector_store import RetrievedChunk


class Reranker:
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.MODEL_NAME)
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
