from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ["NUTRITION_LOOKUP", "HEALTH_ADVICE", "BOTH"]


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"text", "label"}
    if set(df.columns) != expected:
        raise ValueError(f"{path} must have columns {expected}, got {list(df.columns)}")
    return df.dropna(subset=["text", "label"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/classifier_bert")
    parser.add_argument("--csv", default="data/en/intent_v2/intent_hard_test.csv")
    parser.add_argument("--output", default="reports/en/intent_v2/current_on_hard_test.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    df = load_csv(args.csv)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", None) or {idx: label for idx, label in enumerate(LABELS)}

    y_pred = []
    for start in range(0, len(df), args.batch_size):
        batch = df["text"].iloc[start : start + args.batch_size].tolist()
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        y_pred.extend([id2label[int(idx)] for idx in pred_ids])

    y_true = df["label"].tolist()

    result = {
        "model_path": args.model_path,
        "csv": args.csv,
        "rows": len(df),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["rows", "accuracy", "macro_f1", "confusion_matrix"]}, indent=2))


if __name__ == "__main__":
    main()
