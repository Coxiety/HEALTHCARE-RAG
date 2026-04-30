"""Generate synthetic intent dataset via Ollama llama3.1:8b.

Produces 600 questions/class (buffer for dedup) → data/en/synthetic_intent.csv.
Run eval_synthetic_intent.ipynb after to dedup + produce intent_data.csv for training.

Usage:
    python src/data_pipeline/synthesize_intent.py
    python src/data_pipeline/synthesize_intent.py --target 600 --batch 20
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests

_OLLAMA_URL = "http://localhost:11434/api/generate"
_MODEL = "llama3.1:8b"
_CHECKPOINT = "data/en/.synthesize_progress.json"
_DEFAULT_OUTPUT = "data/en/synthetic_intent.csv"
_TARGET_PER_CLASS = 600
_BATCH_SIZE = 20

_PROMPTS: dict[str, str] = {
    "NUTRITION_LOOKUP": (
        "Generate {n} diverse English questions that look up specific nutritional data.\n"
        "Questions ask for exact values (calories, protein, fat, carbs, vitamins, minerals, sodium, fiber, sugar, etc.) "
        "about specific foods or portion sizes.\n"
        "Vary food types (meats, vegetables, fruits, grains, dairy, nuts, legumes) and question structures.\n"
        "Return ONLY the questions, one per line, no numbering, no extra text."
    ),
    "HEALTH_ADVICE": (
        "Generate {n} diverse English questions asking for dietary health advice.\n"
        "Questions ask about foods to eat/avoid for health conditions, dietary recommendations for diseases, "
        "or nutrition strategies for health goals.\n"
        "Vary health conditions (diabetes, hypertension, high cholesterol, gout, anemia, obesity, celiac, IBS, etc.).\n"
        "Return ONLY the questions, one per line, no numbering, no extra text."
    ),
    "BOTH": (
        "Generate {n} diverse English questions that need BOTH nutritional data AND health advice.\n"
        "Questions mention a specific food or nutrient AND a health condition together, "
        "e.g. 'Is salmon good for cholesterol? How much omega-3 does it contain?'\n"
        "Vary the foods, nutrients, and health conditions.\n"
        "Return ONLY the questions, one per line, no numbering, no extra text."
    ),
}


def _call_ollama(prompt: str) -> str:
    payload = {
        "model": _MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.9, "top_p": 0.95},
    }
    resp = requests.post(_OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


def _parse_questions(raw: str) -> list[str]:
    questions = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or len(line) < 10:
            continue
        # Strip common numbering prefixes: "1." "1)" "1:"
        if len(line) > 2 and line[0].isdigit() and line[1] in ".):-":
            line = line[2:].strip()
        elif len(line) > 3 and line[:2].isdigit() and line[2] in ".):-":
            line = line[3:].strip()
        if line:
            questions.append(line)
    return questions


def _load_checkpoint() -> dict[str, list[str]]:
    if os.path.exists(_CHECKPOINT):
        with open(_CHECKPOINT, encoding="utf-8") as f:
            return json.load(f)
    return {"NUTRITION_LOOKUP": [], "HEALTH_ADVICE": [], "BOTH": []}


def _save_checkpoint(progress: dict[str, list[str]]) -> None:
    Path(_CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
    with open(_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def _generate_class(label: str, target: int, batch: int, progress: dict[str, list[str]]) -> None:
    seen: set[str] = set(progress[label])
    questions: list[str] = list(seen)
    print(f"\n[{label}] Resuming from {len(questions)}/{target}")

    while len(questions) < target:
        need = min(batch, target - len(questions) + batch)
        prompt = _PROMPTS[label].format(n=need)
        try:
            raw = _call_ollama(prompt)
            added = 0
            for q in _parse_questions(raw):
                if q not in seen:
                    questions.append(q)
                    seen.add(q)
                    added += 1
            progress[label] = questions
            _save_checkpoint(progress)
            print(f"  [{label}] {len(questions)}/{target} (+{added})")
        except requests.RequestException as exc:
            print(f"  [{label}] Ollama error: {exc} — retrying in 5s")
            time.sleep(5)

    print(f"[{label}] Done — {len(questions)} questions")


def main(target: int, batch: int, output: str) -> None:
    progress = _load_checkpoint()

    for label in ["NUTRITION_LOOKUP", "HEALTH_ADVICE", "BOTH"]:
        if len(progress.get(label, [])) < target:
            _generate_class(label, target, batch, progress)
        else:
            print(f"[{label}] Already have {len(progress[label])} — skipping")

    rows: list[dict] = []
    for label in ["NUTRITION_LOOKUP", "HEALTH_ADVICE", "BOTH"]:
        for q in progress[label][:target]:
            rows.append({"question": q, "intent_label": label})

    random.shuffle(rows)
    df = pd.DataFrame(rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nSaved {len(df)} rows → {output}")
    print(df["intent_label"].value_counts().to_string())

    if os.path.exists(_CHECKPOINT):
        os.remove(_CHECKPOINT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic intent dataset via Ollama")
    parser.add_argument("--target", type=int, default=_TARGET_PER_CLASS,
                        help="Questions to generate per class (default: 600)")
    parser.add_argument("--batch", type=int, default=_BATCH_SIZE,
                        help="Questions per LLM call (default: 20)")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT,
                        help="Output CSV path")
    args = parser.parse_args()
    main(args.target, args.batch, args.output)
