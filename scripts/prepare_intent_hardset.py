from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


NUTRITION_LOOKUP = "NUTRITION_LOOKUP"
HEALTH_ADVICE = "HEALTH_ADVICE"
BOTH = "BOTH"


NUTRITION_LOOKUP_PATTERNS = [
    r"\bhow much\b",
    r"\bhow many\b",
    r"\bcalorie(s)?\s+(in|of|per)\b",
    r"\b(content|amount|level|value)\s+(of|for)\b",
    r"\b(protein|carb(s|ohydrate(s)?)?|fat|fiber|sugar|sodium|cholesterol|vitamin|calcium|iron|potassium|magnesium|zinc|phosphorus|folate|omega-3|dha|epa)\s+(content|amount|level|value)\b",
    r"\bcontain(s)?\s+.*\b(protein|carb(s|ohydrate(s)?)?|fat|fiber|sugar|sodium|cholesterol|vitamin|calcium|iron|potassium|magnesium|zinc|phosphorus|folate|omega-3|dha|epa)\b",
    r"\bper\s*100\s*g\b",
    r"\bserving\b",
    r"\bmg\b",
    r"\bgrams?\b",
]


HEALTH_PATTERNS = [
    r"\bhelp(s|ful)?\b",
    r"\baffect(s)?\b",
    r"\beffect(s)?\b",
    r"\brisk(s)?\b",
    r"\bsafe(ty)?\b",
    r"\btoxicity\b",
    r"\bdanger(s)?\b",
    r"\bprevent(s|ion)?\b",
    r"\btreat(s|ment)?\b",
    r"\blower(s|ing)?\b",
    r"\bincrease(s|d)?\b",
    r"\breduce(s|d|ing)?\b",
    r"\bimprove(s|d)?\b",
    r"\brecommended\b",
    r"\bdiabetes\b",
    r"\bmetabolic\b",
    r"\bcardiovascular\b",
    r"\bcancer\b",
    r"\bpregnancy\b",
    r"\bdisease\b",
    r"\bsyndrome\b",
    r"\bpatients?\b",
    r"\bblood\b",
    r"\bcholesterol\b",
    r"\bnausea\b",
    r"\bvomiting\b",
    r"\binteractions?\b",
]


def _has_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def suggest_intent(question: str) -> tuple[str, str]:
    """Suggest the routing intent under the project schema.

    The key rule is that BOTH is not a general "food + disease" topic label.
    BOTH means the query needs exact USDA nutrition lookup and health retrieval.
    """
    q = question.lower()
    has_nutrition_lookup = _has_any(NUTRITION_LOOKUP_PATTERNS, q)
    has_health = _has_any(HEALTH_PATTERNS, q)

    if has_nutrition_lookup and has_health:
        return BOTH, "explicit nutrition lookup cue + health/clinical cue"
    if has_nutrition_lookup:
        return NUTRITION_LOOKUP, "explicit nutrition lookup cue only"
    return HEALTH_ADVICE, "health/research question; no explicit USDA nutrition lookup cue"


def load_cases_predictions(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["question"]: row.get("predicted_intent", "") for row in reader}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_report(rows: list[dict], output_path: Path) -> None:
    old_counts = Counter(row["old_expected_intent"] for row in rows)
    suggested_counts = Counter(row["suggested_intent"] for row in rows)
    predicted_counts = Counter(row["current_predicted_intent"] for row in rows if row["current_predicted_intent"])
    changed = sum(1 for row in rows if row["old_expected_intent"] != row["suggested_intent"])
    review = sum(1 for row in rows if row["needs_manual_review"] == "yes")

    lines = [
        "# Intent Hard-Set Relabel Audit",
        "",
        "This file audits the multi-hop nutrition dataset against the project routing intent schema.",
        "",
        "Schema:",
        "- NUTRITION_LOOKUP: exact nutrition facts from USDA are needed.",
        "- HEALTH_ADVICE: health, disease, safety, or clinical advice/research question.",
        "- BOTH: both exact USDA nutrition facts and health retrieval are needed.",
        "",
        f"Total rows: {len(rows)}",
        f"Rows whose suggested label differs from old label: {changed}",
        f"Rows flagged for manual review: {review}",
        "",
        "## Old Label Distribution",
        "",
    ]
    lines.extend(f"- {label}: {count}" for label, count in sorted(old_counts.items()))
    lines.extend(["", "## Suggested Label Distribution", ""])
    lines.extend(f"- {label}: {count}" for label, count in sorted(suggested_counts.items()))
    if predicted_counts:
        lines.extend(["", "## Current Classifier Prediction Distribution", ""])
        lines.extend(f"- {label}: {count}" for label, count in sorted(predicted_counts.items()))

    lines.extend(
        [
            "",
            "## Review Guidance",
            "",
            "Open `data/en/intent_hard_review.csv` and fill `final_label`.",
            "Keep BOTH only when the question truly needs exact USDA nutrition data and health retrieval.",
            "If the question is about evidence, safety, disease risk, treatment, or clinical effect only, use HEALTH_ADVICE.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/en/eval_hotpot_nutrition.jsonl")
    parser.add_argument("--cases", default="reports/en/rag_eval_hotpot/cases.csv")
    parser.add_argument("--output", default="data/en/intent_hard_review.csv")
    parser.add_argument("--report", default="reports/en/intent_hardset_audit.md")
    args = parser.parse_args()

    input_path = Path(args.input)
    cases_path = Path(args.cases)
    output_path = Path(args.output)
    report_path = Path(args.report)

    predictions = load_cases_predictions(cases_path)
    source_rows = read_jsonl(input_path)
    review_rows = []

    for idx, row in enumerate(source_rows):
        question = row["question"]
        old_label = row.get("expected_intent", "")
        suggested, reason = suggest_intent(question)
        predicted = predictions.get(question, "")
        needs_review = old_label != suggested or (predicted and predicted != suggested)
        review_rows.append(
            {
                "id": idx,
                "question": question,
                "old_expected_intent": old_label,
                "current_predicted_intent": predicted,
                "suggested_intent": suggested,
                "final_label": suggested,
                "needs_manual_review": "yes" if needs_review else "no",
                "reason": reason,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)

    write_report(review_rows, report_path)
    print(f"Saved review CSV: {output_path} ({len(review_rows)} rows)")
    print(f"Saved audit report: {report_path}")


if __name__ == "__main__":
    main()
