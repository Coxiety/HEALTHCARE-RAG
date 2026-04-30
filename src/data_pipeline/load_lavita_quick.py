"""Load lavita/medical-qa-datasets into ChromaDB — run once.

Usage:
    python -m src.data_pipeline.load_lavita_quick

TV2 sẽ replace bằng pipeline đầy đủ hơn khi có data/en/corpus.jsonl.
"""

import yaml
from datasets import load_dataset

from src.data_pipeline.chunker import Chunker
from src.database.vector_store import VectorStore

cfg = yaml.safe_load(open("configs/config.yaml"))
vs = VectorStore(cfg["chroma_persist_dir"], cfg["chroma_collection"], cfg["embedding_model"])
chunker = Chunker(cfg["chunk_size"], cfg["chunk_overlap"])

print("Loading lavita/medical-qa-datasets (first 2000 samples)...")
ds = load_dataset("lavita/medical-qa-datasets", "all-processed", split="train[:2000]")

chunks = []
for row in ds:
    text = f"Q: {row['input']}\nA: {row['output']}"
    chunks.extend(chunker.chunk_text(text, source="lavita"))

print(f"Embedding {len(chunks)} chunks into ChromaDB...")
batch_size = 2000
for i in range(0, len(chunks), batch_size):
    vs.add(chunks[i : i + batch_size])
    print(f"  Added {min(i + batch_size, len(chunks))}/{len(chunks)}")
print(f"Done. Total in store: {vs.count()}")
