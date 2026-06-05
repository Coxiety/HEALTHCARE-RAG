"""Build data/en/eval_200.jsonl với relevant_docs đúng.

Cách hoạt động:
  1. Load corpus.jsonl vào memory
  2. Load lavita[0:2000] — cùng nguồn với corpus
  3. Với mỗi QA pair, tìm chunk(s) có text là substring của full_text gốc
     (vì chunker cắt thẳng từ full_text → chunk luôn là substring)
  4. Lấy 200 QA có ít nhất 1 relevant chunk

Output schema: {"question": str, "answer": str, "relevant_docs": [chunk_id, ...]}

Usage:
    conda run -n nutrition-rag python -m src.data_pipeline.build_eval_set
"""

import json
import os
import random

from datasets import load_dataset

CORPUS_PATH = "data/en/corpus.jsonl"
OUT_PATH    = "data/en/eval_200.jsonl"
N_EVAL      = 200
SEED        = 42

random.seed(SEED)
os.makedirs("data/en", exist_ok=True)

# --- Load corpus ---
print("Loading corpus...")
corpus: list[dict] = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        corpus.append(json.loads(line))

# --- Load lavita (same split as corpus) ---
print("Loading lavita[0:2000]...")
ds = load_dataset("lavita/medical-qa-datasets", "all-processed", split="train[:2000]")

# --- Match QA pairs to their corpus chunks ---
records = []
for row in ds:
    question = row.get("input",  "").strip()
    answer   = row.get("output", "").strip()
    if not question or not answer:
        continue

    full_text   = f"Q: {question}\nA: {answer}"
    fingerprint = full_text[:80]  # unique prefix per QA pair

    # Chỉ dùng fingerprint để match chunk đầu tiên của QA pair này.
    # Tránh false positive từ condition "c['text'] in full_text".
    relevant_ids = [c["id"] for c in corpus if fingerprint in c["text"]]

    if relevant_ids:
        records.append({
            "question":     question,
            "answer":       answer,
            "relevant_docs": relevant_ids,
        })

print(f"Found {len(records)} QA pairs with at least 1 relevant chunk")

# Sample 200
sampled = random.sample(records, min(N_EVAL, len(records)))

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for rec in sampled:
        f.write(json.dumps(rec) + "\n")

print(f"Wrote {len(sampled)} eval pairs -> {OUT_PATH}")

# Quick stats
avg_rel = sum(len(r["relevant_docs"]) for r in sampled) / len(sampled)
print(f"Avg relevant chunks per question: {avg_rel:.1f}")
