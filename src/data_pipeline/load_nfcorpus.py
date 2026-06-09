"""Download NFCorpus (BEIR) → ChromaDB collection "nfcorpus" + data/en/corpus.jsonl.

Downloads ~50 MB zip, extracts to data/nfcorpus/, indexes 9,964 documents.
Run once from project root:
    python -m src.data_pipeline.load_nfcorpus
"""

from __future__ import annotations

import io
import json
import os
import zipfile

import requests
import yaml

from src.database.vector_store import RetrievedChunk, VectorStore

NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
RAW_DIR      = "data/nfcorpus"
CORPUS_OUT   = "data/en/corpus.jsonl"
BATCH_SIZE   = 5000   # ChromaDB hard limit is 5,461
CONFIG_PATH  = "configs/config.yaml"


def download_and_extract() -> None:
    if os.path.exists(os.path.join(RAW_DIR, "corpus.jsonl")):
        print(f"NFCorpus already extracted at {RAW_DIR}/ — skipping download.")
        return

    print(f"Downloading NFCorpus (~50 MB)...")
    os.makedirs("data", exist_ok=True)
    resp = requests.get(NFCORPUS_URL, timeout=180, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
        buf.write(chunk)
        downloaded += len(chunk)
        if total:
            print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()

    print("Extracting zip...")
    with zipfile.ZipFile(buf) as z:
        z.extractall("data")
    print(f"Extracted → {RAW_DIR}/")


def load_and_index(vs: VectorStore) -> list[dict]:
    """Read BEIR corpus.jsonl, normalize to internal schema, index into ChromaDB."""
    print("Reading corpus.jsonl...")
    docs: list[dict] = []
    with open(os.path.join(RAW_DIR, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            # Combine title + abstract — standard for PubMed abstracts
            text = f"{raw['title']}\n\n{raw['text']}".strip()
            docs.append({"id": raw["_id"], "text": text, "source": "nfcorpus"})
    print(f"Loaded {len(docs)} documents")

    if vs.count() > 0:
        print(f"Clearing existing collection '{vs.collection_name}' ({vs.count()} docs)...")
        vs.clear()

    print(f"Indexing into ChromaDB '{vs.collection_name}' (batch_size={BATCH_SIZE})...")
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        vs.add(batch)
        print(f"  Indexed {min(i + BATCH_SIZE, len(docs))} / {len(docs)}")

    print(f"ChromaDB total: {vs.count()}")
    return docs


def export_corpus(docs: list[dict]) -> None:
    os.makedirs("data/en", exist_ok=True)
    with open(CORPUS_OUT, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    print(f"Exported {len(docs)} lines → {CORPUS_OUT}")


def main() -> None:
    cfg = yaml.safe_load(open(CONFIG_PATH))
    vs  = VectorStore(cfg["chroma_persist_dir"], cfg["chroma_collection"], cfg["embedding_model"])

    download_and_extract()
    docs = load_and_index(vs)
    export_corpus(docs)

    # Quick sanity check
    print("\n=== Sanity check ===")
    print(f"corpus.jsonl lines : {sum(1 for _ in open(CORPUS_OUT, encoding='utf-8'))}")
    with open(CORPUS_OUT, encoding="utf-8") as f:
        sample = json.loads(f.readline())
    print(f"Sample id          : {sample['id']}")
    print(f"Sample text[:80]   : {sample['text'][:80]}")
    print(f"Sample source      : {sample['source']}")
    print("Done.")


if __name__ == "__main__":
    main()
