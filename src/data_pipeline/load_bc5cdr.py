"""Convert BC5CDR (tner/bc5cdr) → data/en/bc5cdr_bio.jsonl.

Load thẳng JSON files từ HuggingFace, bypass loading script.
TNER integer label mapping: 0→O, 1→B-NUTRIENT, 2→I-NUTRIENT, 3→B-DISEASE, 4→I-DISEASE

Output schema: {"tokens": [str, ...], "labels": [str, ...]}
TV2 có thể rewrite với MinHash LSH dedup — giữ nguyên schema và file path.

Usage:
    conda run -n nutrition-rag python -m src.data_pipeline.load_bc5cdr
"""

import json
import os

from datasets import load_dataset

BASE = "https://huggingface.co/datasets/tner/bc5cdr/raw/main/dataset"
INT_TO_LABEL = {0: "O", 1: "B-NUTRIENT", 2: "I-NUTRIENT", 3: "B-DISEASE", 4: "I-DISEASE"}

os.makedirs("data/en", exist_ok=True)
OUT_PATH = "data/en/bc5cdr_bio.jsonl"

print("Loading tner/bc5cdr JSON files from HuggingFace...")
ds = load_dataset("json", data_files={
    "train":      f"{BASE}/train.json",
    "validation": f"{BASE}/valid.json",
    "test":       f"{BASE}/test.json",
})

count = 0
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for split in ("train", "validation", "test"):
        for row in ds[split]:
            tokens = row["tokens"]
            labels = [INT_TO_LABEL.get(t, "O") for t in row["tags"]]
            f.write(json.dumps({"tokens": tokens, "labels": labels}) + "\n")
            count += 1

print(f"Wrote {count} sentences → {OUT_PATH}")
