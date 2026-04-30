"""Mine BM25 hard negatives từ NFCorpus → data/en/triplets.jsonl.

Strategy:
  - Query + positive lấy từ NFCorpus train qrels (score >= 1)
  - Hard negative: BM25 top-k chunk không phải positive

Output schema: {"query": str, "positive": str, "negative": str}

Usage:
    conda run -n nutrition-rag python -m src.data_pipeline.triplet_miner
"""

from __future__ import annotations

import csv
import json
import os
import random

from rank_bm25 import BM25Okapi

CORPUS_PATH   = "data/en/corpus.jsonl"
QUERIES_PATH  = "data/nfcorpus/queries.jsonl"
QRELS_TRAIN   = "data/nfcorpus/qrels/train.tsv"
OUT_PATH      = "data/en/triplets.jsonl"
TOP_K_NEG     = 10
SEED          = 42

random.seed(SEED)
os.makedirs("data/en", exist_ok=True)


def tokenize(text: str) -> list[str]:
    return text.lower().split()


# --- Load corpus ---
print("Loading corpus...")
corpus: list[dict] = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        corpus.append(json.loads(line))

corpus_texts  = [c["text"] for c in corpus]
id_to_text    = {c["id"]: c["text"] for c in corpus}
tokenized     = [tokenize(t) for t in corpus_texts]

print(f"Building BM25 index over {len(corpus)} docs...")
bm25 = BM25Okapi(tokenized)

# --- Load queries ---
print("Loading queries...")
queries: dict[str, str] = {}
with open(QUERIES_PATH, encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        queries[q["_id"]] = q["text"]

# --- Load train qrels (score >= 1 = relevant) ---
print("Loading train qrels...")
pairs: list[tuple[str, str]] = []   # (query_text, positive_text)
with open(QRELS_TRAIN, encoding="utf-8", newline="") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)  # skip header: query-id  corpus-id  score
    for row in reader:
        if len(row) < 3:
            continue
        qid, did, score = row[0], row[1], int(row[2])
        if score >= 1 and qid in queries and did in id_to_text:
            pairs.append((queries[qid], id_to_text[did]))

print(f"Found {len(pairs)} (query, positive) pairs from train qrels")
random.shuffle(pairs)

# --- Mine hard negatives ---
written = 0
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for query, positive in pairs:
        scores   = bm25.get_scores(tokenize(query))
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K_NEG]

        pos_toks   = set(tokenize(positive))
        candidates = [
            i for i in top_idxs
            if len(set(tokenize(corpus_texts[i])) & pos_toks) / max(len(pos_toks), 1) < 0.5
        ]
        if not candidates:
            candidates = top_idxs

        neg_idx  = random.choice(candidates)
        negative = corpus_texts[neg_idx]

        f.write(json.dumps({"query": query, "positive": positive, "negative": negative}) + "\n")
        written += 1

print(f"Wrote {written} triplets → {OUT_PATH}")
