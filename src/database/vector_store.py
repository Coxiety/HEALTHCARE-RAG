from __future__ import annotations

import os
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
import chromadb


@dataclass
class RetrievedChunk:
    text:   str
    source: str
    score:  float  # similarity [0.0, 1.0]


class VectorStore:
    # Dùng lazy init để tránh load model khi chỉ import module.
    # Model SentenceTransformer (~400MB) chỉ load khi thực sự gọi embed/query.

    def __init__(
        self,
        persist_dir:     str = "data/chroma_db",
        collection_name: str = "health_articles",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.persist_dir     = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self._client:     chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None       = None
        self._embedder:   SentenceTransformer | None       = None

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self._get_client().get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # báo ChromaDB dùng cosine distance
            )
        return self._collection

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize=True bắt buộc khi dùng cosine — vector phải có độ dài = 1
        return self._get_embedder().encode(texts, normalize_embeddings=True).tolist()

    def add(self, chunks: list[dict]) -> None:
        # chunks: [{"id": str, "text": str, "source": str}, ...]
        if not chunks:
            return

        collection = self._get_collection()
        texts      = [c["text"]   for c in chunks]

        collection.add(
            ids        =[c["id"]     for c in chunks],
            embeddings =self.embed(texts),
            documents  =texts,
            metadatas  =[{"source": c["source"]} for c in chunks],
        )

    def query(self, text: str, top_k: int = 5) -> list[RetrievedChunk]:
        collection = self._get_collection()

        if collection.count() == 0:
            return []

        results = collection.query(
            query_embeddings=[self.embed([text])[0]],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB trả về cosine distance [0, 2], đổi về similarity [0, 1]
            score = max(0.0, min(1.0, 1 - dist / 2))
            chunks.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                score=score,
            ))

        return chunks

    def get_all_chunks(self) -> list[dict]:
        # Dùng để build BM25 index trong retriever.py
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        results = collection.get(include=["documents", "metadatas"])
        return [
            {"id": id_, "text": doc, "source": meta.get("source", "unknown")}
            for id_, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]

    def count(self) -> int:
        return self._get_collection().count()

    def clear(self) -> None:
        client = self._get_client()
        client.delete_collection(self.collection_name)
        self._collection = None  # reset cache để get_or_create tạo lại


if __name__ == "__main__":
    vs = VectorStore()

    vs.add([
        {"id": "t1", "text": "Người bị tiểu đường nên hạn chế tinh bột và đường.", "source": "test"},
        {"id": "t2", "text": "Cá hồi giàu omega-3, tốt cho tim mạch và não bộ.", "source": "test"},
        {"id": "t3", "text": "Rau xanh cung cấp chất xơ, vitamin và khoáng chất.", "source": "test"},
    ])

    print(f"Tổng chunks: {vs.count()}")
    for r in vs.query("tiểu đường nên ăn gì", top_k=2):
        print(f"  [{r.score:.3f}] {r.text[:60]}...")
