from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import inspect


LABELS = ["NUTRITION_LOOKUP", "HEALTH_ADVICE", "BOTH"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[str], tokenizer: AutoTokenizer, max_length: int = 128):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = [LABEL2ID[label] for label in labels]

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"text", "label"}
    if set(df.columns) != expected:
        raise ValueError(f"{path} must have columns {expected}, got {list(df.columns)}")
    bad = sorted(set(df["label"]) - set(LABELS))
    if bad:
        raise ValueError(f"{path} has unknown labels: {bad}")
    return df.dropna(subset=["text", "label"]).reset_index(drop=True)


def metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def evaluate_to_files(trainer: Trainer, dataset: IntentDataset, output_dir: Path, prefix: str) -> dict:
    pred = trainer.predict(dataset)
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=-1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": LABELS,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    rows = []
    for gold, pred_id in zip(y_true, y_pred):
        rows.append({"gold": ID2LABEL[int(gold)], "predicted": ID2LABEL[int(pred_id)]})
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/en/intent_v2/intent_train_v2.csv")
    parser.add_argument("--hard_val_csv", default="data/en/intent_v2/intent_hard_val.csv")
    parser.add_argument("--hard_test_csv", default="data/en/intent_v2/intent_hard_test.csv")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--output_dir", default="models/classifier_bert_v2")
    parser.add_argument("--report_dir", default="reports/en/intent_v2")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    train_all = load_csv(args.train_csv)
    hard_val = load_csv(args.hard_val_csv)
    hard_test = load_csv(args.hard_test_csv)

    train_df, synthetic_val_df = train_test_split(
        train_all,
        test_size=0.1,
        random_state=42,
        stratify=train_all["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = IntentDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, args.max_length)
    synthetic_val_dataset = IntentDataset(
        synthetic_val_df["text"].tolist(), synthetic_val_df["label"].tolist(), tokenizer, args.max_length
    )
    hard_val_dataset = IntentDataset(hard_val["text"].tolist(), hard_val["label"].tolist(), tokenizer, args.max_length)
    hard_test_dataset = IntentDataset(hard_test["text"].tolist(), hard_test["label"].tolist(), tokenizer, args.max_length)

    training_arg_params = inspect.signature(TrainingArguments.__init__).parameters
    training_arg_kwargs = {
        "output_dir": args.output_dir,
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 20,
        "report_to": "none",
        "seed": 42,
    }
    if "eval_strategy" in training_arg_params:
        training_arg_kwargs["eval_strategy"] = "epoch"
    else:
        training_arg_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_arg_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": synthetic_val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer),
        "compute_metrics": metrics_fn,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    report_dir = Path(args.report_dir)
    synthetic_val_result = evaluate_to_files(trainer, synthetic_val_dataset, report_dir, "synthetic_val")
    hard_val_result = evaluate_to_files(trainer, hard_val_dataset, report_dir, "hard_val")
    hard_test_result = evaluate_to_files(trainer, hard_test_dataset, report_dir, "hard_test")

    summary = {
        "train_rows": len(train_df),
        "synthetic_val_rows": len(synthetic_val_df),
        "hard_val_rows": len(hard_val),
        "hard_test_rows": len(hard_test),
        "synthetic_val": {
            "accuracy": synthetic_val_result["accuracy"],
            "macro_f1": synthetic_val_result["macro_f1"],
        },
        "hard_val": {
            "accuracy": hard_val_result["accuracy"],
            "macro_f1": hard_val_result["macro_f1"],
        },
        "hard_test": {
            "accuracy": hard_test_result["accuracy"],
            "macro_f1": hard_test_result["macro_f1"],
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
