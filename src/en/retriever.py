"""Retrieval baselines for the EN pipeline.

Classes:
  TFIDFRetriever   — sparse baseline (sklearn TF-IDF)
  BM25Retriever    — sparse variant (rank_bm25)
  DenseRetriever   — dense baseline via ChromaDB (vanilla or fine-tuned)
  HybridRetriever  — BM25 + Dense with Reciprocal Rank Fusion (k=60)

Contract: retrieve(query, top_k) -> list[RetrievedChunk]
"""

from __future__ import annotations

import json

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.database.vector_store import RetrievedChunk, VectorStore


# ---------------------------------------------------------------------------
# TF-IDF (sparse baseline — MVP)
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self._index = None
        self._chunks: list[dict] | None = None
        self.vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))

    def _build_index(self) -> None:
        self._chunks = self.vs.get_all_chunks()
        texts = [c["text"] for c in self._chunks]
        self._index = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._index is None:
            self._build_index()
        qvec = self.vectorizer.transform([query])
        scores = cosine_similarity(qvec, self._index).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                text=self._chunks[i]["text"],
                source=self._chunks[i].get("source", ""),
                score=float(scores[i]),
            )
            for i in top_idx
            if scores[i] > 0
        ]


# ---------------------------------------------------------------------------
# BM25 (sparse variant)
# Loads corpus from data/en/corpus.jsonl — TV2 replace file, schema unchanged.
# ---------------------------------------------------------------------------

class BM25Retriever:
    CORPUS_PATH = "data/en/corpus.jsonl"

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _build_index(self) -> None:
        self._corpus = []
        with open(self.CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                self._corpus.append(json.loads(line))
        tokenized = [self._tokenize(c["text"]) for c in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._bm25 is None:
            self._build_index()
        scores = self._bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        max_score = float(scores[top_idx[0]]) if len(top_idx) else 1.0
        return [
            RetrievedChunk(
                text=self._corpus[i]["text"],
                source=self._corpus[i].get("source", ""),
                score=float(scores[i]) / max(max_score, 1e-9),  # normalize to [0,1]
            )
            for i in top_idx
            if scores[i] > 0
        ]


# ---------------------------------------------------------------------------
# Dense (vanilla all-MiniLM or fine-tuned model via ChromaDB)
# ---------------------------------------------------------------------------

class DenseRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.vs.query(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Hybrid RRF — BM25 + Dense, Reciprocal Rank Fusion k=60
# ---------------------------------------------------------------------------

class HybridRetriever:
    RRF_K = 60

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever):
        self.bm25  = bm25
        self.dense = dense

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # Fetch more candidates before fusing
        fetch_k = max(top_k * 4, 20)
        bm25_hits  = self.bm25.retrieve(query,  top_k=fetch_k)
        dense_hits = self.dense.retrieve(query, top_k=fetch_k)

        # RRF score: sum of 1/(k + rank) across both lists
        rrf: dict[str, float] = {}
        text_to_chunk: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(bm25_hits):
            rrf[chunk.text] = rrf.get(chunk.text, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            text_to_chunk[chunk.text] = chunk

        for rank, chunk in enumerate(dense_hits):
            rrf[chunk.text] = rrf.get(chunk.text, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            text_to_chunk[chunk.text] = chunk

        top_texts = sorted(rrf, key=lambda t: rrf[t], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                text=t,
                source=text_to_chunk[t].source,
                score=rrf[t],
            )
            for t in top_texts
        ]
