from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ["NUTRITION_LOOKUP", "HEALTH_ADVICE", "BOTH"]


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_metrics(rows: list[dict]) -> dict:
    labels = sorted(set(LABELS) | {row["final_label"] for row in rows} | {row["current_predicted_intent"] for row in rows})
    labels = [label for label in labels if label]

    total = len(rows)
    correct = sum(1 for row in rows if row["final_label"] == row["current_predicted_intent"])
    confusion = Counter((row["final_label"], row["current_predicted_intent"]) for row in rows)

    per_label = {}
    for label in labels:
        tp = confusion[(label, label)]
        fp = sum(confusion[(gold, label)] for gold in labels if gold != label)
        fn = sum(confusion[(label, pred)] for pred in labels if pred != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        support = sum(confusion[(label, pred)] for pred in labels)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_f1 = safe_div(sum(item["f1"] for item in per_label.values()), len(per_label))
    return {
        "labels": labels,
        "total": total,
        "accuracy": safe_div(correct, total),
        "macro_f1": macro_f1,
        "confusion": confusion,
        "per_label": per_label,
        "gold_counts": Counter(row["final_label"] for row in rows),
        "pred_counts": Counter(row["current_predicted_intent"] for row in rows),
    }


def write_report(metrics: dict, output_path: Path) -> None:
    labels = metrics["labels"]
    lines = [
        "# Intent Hard-Set Evaluation",
        "",
        "Evaluation uses `final_label` from `data/en/intent_hard_review.csv`.",
        "`current_predicted_intent` is taken from the latest multi-hop evaluation cases file.",
        "",
        f"Total rows: {metrics['total']}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro-F1: {metrics['macro_f1']:.4f}",
        "",
        "## Label Distribution",
        "",
        "| Label | Gold | Predicted |",
        "|---|---:|---:|",
    ]
    for label in labels:
        lines.append(f"| {label} | {metrics['gold_counts'].get(label, 0)} | {metrics['pred_counts'].get(label, 0)} |")

    lines.extend(["", "## Per-Label Metrics", "", "| Label | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"])
    for label in labels:
        item = metrics["per_label"][label]
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |"
        )

    lines.extend(["", "## Confusion Matrix", "", "| Gold \\ Pred | " + " | ".join(labels) + " |", "|" + "---|" * (len(labels) + 1)])
    for gold in labels:
        values = [str(metrics["confusion"].get((gold, pred), 0)) for pred in labels]
        lines.append(f"| {gold} | " + " | ".join(values) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/en/intent_hard_review.csv")
    parser.add_argument("--output", default="reports/en/intent_hardset_eval.md")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    missing = [row for row in rows if not row.get("final_label") or not row.get("current_predicted_intent")]
    if missing:
        raise SystemExit(f"Missing final_label/current_predicted_intent in {len(missing)} rows")

    metrics = compute_metrics(rows)
    write_report(metrics, Path(args.output))
    print(f"Saved evaluation report: {args.output}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
