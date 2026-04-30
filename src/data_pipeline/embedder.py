from __future__ import annotations

import json

from src.database.vector_store import VectorStore
from src.data_pipeline.chunker import Chunker


class Embedder:
    # Hai luồng tùy nguồn dữ liệu:
    #   embed_jsonl()     — medical_knowledge.jsonl, mỗi dòng đã ngắn sẵn
    #   embed_directory() — bài crawl thật, cần chunk trước

    def __init__(
        self,
        vector_store: VectorStore,
        chunk_size:   int = 500,
        chunk_overlap: int = 100,
    ):
        self.vector_store  = vector_store
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def embed_jsonl(self, filepath: str) -> int:
        import uuid

        chunks = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text   = obj.get("text", "").strip()
                source = obj.get("source", filepath)
                if text:
                    chunks.append({
                        "id":     str(uuid.uuid4()),
                        "text":   text,
                        "source": source,
                    })

        self.vector_store.add(chunks)
        return len(chunks)

    def embed_directory(self, dir_path: str) -> int:
        chunker = Chunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = chunker.chunk_directory(dir_path)
        self.vector_store.add(chunks)
        return len(chunks)


if __name__ == "__main__":
    vs = VectorStore()

    if vs.count() > 0:
        print(f"ChromaDB already has {vs.count()} chunks. Skipping.")
    else:
        embedder = Embedder(vs)
        n = embedder.embed_jsonl("data/medical_knowledge.jsonl")
        print(f"Embedded {n} chunks.")

    print("\nTest query: 'tiểu đường nên ăn gì'")
    results = vs.query("tiểu đường nên ăn gì", top_k=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.text[:70]}...")
