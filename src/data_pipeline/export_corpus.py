"""Export ChromaDB chunks to data/en/corpus.jsonl.

Dùng data đã load từ load_lavita_quick.py (6,178 chunks).
TV2 replace bằng load_lavita.py với dedup MinHash LSH — giữ nguyên schema.

Schema: {"id": "str", "text": "str", "source": "str"}

Usage:
    python -m src.data_pipeline.export_corpus
"""

import json
import os

import yaml

from src.database.vector_store import VectorStore

cfg = yaml.safe_load(open("configs/config.yaml"))
vs = VectorStore(cfg["chroma_persist_dir"], cfg["chroma_collection"], cfg["embedding_model"])

os.makedirs("data/en", exist_ok=True)
out_path = "data/en/corpus.jsonl"

print(f"Reading {vs.count()} chunks from ChromaDB...")
chunks = vs.get_all_chunks()

with open(out_path, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps({"id": chunk["id"], "text": chunk["text"], "source": chunk["source"]}) + "\n")

print(f"Wrote {len(chunks)} records → {out_path}")
