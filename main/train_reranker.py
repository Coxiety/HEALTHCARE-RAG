"""Fine-tune a cross-encoder reranker for NFCorpus on Colab or Kaggle.

Inputs:
  - data/en/corpus.jsonl
  - data/nfcorpus/queries.jsonl
  - data/nfcorpus/qrels/train.tsv
  - data/nfcorpus/qrels/dev.tsv (optional; falls back to a train split if missing)

Output:
  - output_dir/final/ contains the best reranker checkpoint

Example:
  python main/train_reranker.py --repo-root . --output-dir models/reranker_domain
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from tqdm import tqdm


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = ENGLISH_STOP_WORDS


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [token for token in tokens if token not in STOP_WORDS]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for row in read_jsonl(path):
        qid = row.get("_id") or row.get("id") or row.get("qid")
        text = row.get("text") or row.get("query")
        if qid and text:
            queries[str(qid)] = str(text).strip()
    return queries


def load_corpus(path: Path) -> list[dict[str, str]]:
    corpus: list[dict[str, str]] = []
    for row in read_jsonl(path):
        doc_id = row.get("id") or row.get("_id")
        text = row.get("text") or row.get("document")
        source = row.get("source", "nfcorpus")
        if doc_id and text:
            corpus.append({"id": str(doc_id), "text": str(text), "source": str(source)})
    return corpus


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            qid, doc_id, score = row[0], row[1], int(row[2])
            qrels.setdefault(qid, {})[doc_id] = score
    return qrels


def build_bm25(corpus: list[dict[str, str]]) -> BM25Okapi:
    tokenized = [tokenize(item["text"]) for item in corpus]
    return BM25Okapi(tokenized)


def get_positive_docs(
    qrels: dict[str, dict[str, int]],
    qid: str,
    corpus_by_id: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    positives: list[dict[str, str]] = []
    for doc_id, score in qrels.get(qid, {}).items():
        if score >= 1 and doc_id in corpus_by_id:
            positives.append(corpus_by_id[doc_id])
    return positives


def mine_negatives(
    query: str,
    positive_ids: set[str],
    corpus: list[dict[str, str]],
    bm25: BM25Okapi,
    rng: random.Random,
    hard_negative_top_k: int,
    hard_negatives_per_query: int,
    random_negatives_per_query: int,
) -> list[dict[str, str]]:
    scores = bm25.get_scores(tokenize(query))
    ranked_idx = np.argsort(scores)[::-1]

    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()

    for idx in ranked_idx[:hard_negative_top_k]:
        item = corpus[int(idx)]
        doc_id = item["id"]
        if doc_id in positive_ids or doc_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(doc_id)
        if len(selected) >= hard_negatives_per_query:
            break

    if len(selected) < hard_negatives_per_query:
        for idx in ranked_idx[hard_negative_top_k:]:
            item = corpus[int(idx)]
            doc_id = item["id"]
            if doc_id in positive_ids or doc_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(doc_id)
            if len(selected) >= hard_negatives_per_query:
                break

    random_pool = [item for item in corpus if item["id"] not in positive_ids and item["id"] not in selected_ids]
    if random_negatives_per_query > 0 and random_pool:
        sample_size = min(random_negatives_per_query, len(random_pool))
        selected.extend(rng.sample(random_pool, sample_size))

    return selected


def build_training_dataset(
    qrels: dict[str, dict[str, int]],
    queries: dict[str, str],
    corpus: list[dict[str, str]],
    corpus_by_id: dict[str, dict[str, str]],
    bm25: BM25Okapi,
    rng: random.Random,
    hard_negative_top_k: int,
    hard_negatives_per_query: int,
    random_negatives_per_query: int,
) -> Dataset:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, float]] = set()

    for qid in tqdm(sorted(qrels), desc="Building train pairs"):
        query = queries.get(qid)
        if not query:
            continue

        positives = get_positive_docs(qrels, qid, corpus_by_id)
        if not positives:
            continue

        positive_ids = {item["id"] for item in positives}
        negatives = mine_negatives(
            query=query,
            positive_ids=positive_ids,
            corpus=corpus,
            bm25=bm25,
            rng=rng,
            hard_negative_top_k=hard_negative_top_k,
            hard_negatives_per_query=hard_negatives_per_query,
            random_negatives_per_query=random_negatives_per_query,
        )

        for item in positives:
            key = (query, item["text"], 1.0)
            if key not in seen:
                rows.append({"query": query, "document": item["text"], "label": 1.0})
                seen.add(key)

        for item in negatives:
            key = (query, item["text"], 0.0)
            if key not in seen:
                rows.append({"query": query, "document": item["text"], "label": 0.0})
                seen.add(key)

    if not rows:
        raise RuntimeError("No training pairs were built. Check qrels and corpus paths.")

    dataset = Dataset.from_list(rows)
    dataset = dataset.select_columns(["query", "document", "label"])
    dataset = dataset.shuffle(seed=rng.randint(0, 2**31 - 1))
    return dataset


def build_reranking_samples(
    qrels: dict[str, dict[str, int]],
    queries: dict[str, str],
    corpus: list[dict[str, str]],
    corpus_by_id: dict[str, dict[str, str]],
    bm25: BM25Okapi,
    rng: random.Random,
    hard_negative_top_k: int,
    hard_negatives_per_query: int,
) -> list[dict[str, list[str] | str]]:
    samples: list[dict[str, list[str] | str]] = []

    for qid in tqdm(sorted(qrels), desc="Building dev samples"):
        query = queries.get(qid)
        if not query:
            continue

        positives = get_positive_docs(qrels, qid, corpus_by_id)
        if not positives:
            continue

        positive_ids = {item["id"] for item in positives}
        negatives = mine_negatives(
            query=query,
            positive_ids=positive_ids,
            corpus=corpus,
            bm25=bm25,
            rng=rng,
            hard_negative_top_k=hard_negative_top_k,
            hard_negatives_per_query=hard_negatives_per_query,
            random_negatives_per_query=0,
        )

        sample = {
            "query": query,
            "positive": [item["text"] for item in positives],
            "negative": [item["text"] for item in negatives],
        }
        samples.append(sample)

    if not samples:
        raise RuntimeError("No dev samples were built. Check dev qrels and corpus paths.")

    return samples


def split_train_dev_qids(
    train_qrels: dict[str, dict[str, int]],
    dev_qrels_path: Path,
    seed: int,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    if dev_qrels_path.exists():
        return train_qrels, load_qrels(dev_qrels_path)

    qids = sorted(train_qrels)
    rng = random.Random(seed)
    rng.shuffle(qids)
    cut = max(1, int(len(qids) * 0.9))
    train_ids = set(qids[:cut])
    dev_ids = set(qids[cut:])

    split_train = {qid: train_qrels[qid] for qid in train_ids}
    split_dev = {qid: train_qrels[qid] for qid in dev_ids}
    return split_train, split_dev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the NFCorpus cross-encoder reranker for retrieval ranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/reranker_domain"),
        help="Directory where checkpoints and the final model are saved.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Base cross-encoder checkpoint to fine-tune.",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Maximum token length for each pair.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Per-device training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Per-device evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for the scheduler.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--logging-steps", type=int, default=100, help="Logging frequency in steps.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--eval-at-k", type=int, default=10, help="Cutoff for reranking metrics.")
    parser.add_argument(
        "--train-hard-negative-top-k",
        type=int,
        default=100,
        help="How many BM25 candidates to inspect per query while mining train negatives.",
    )
    parser.add_argument(
        "--train-hard-negatives-per-query",
        type=int,
        default=2,
        help="Number of BM25 hard negatives to keep per training query.",
    )
    parser.add_argument(
        "--train-random-negatives-per-query",
        type=int,
        default=2,
        help="Number of random negatives to add per training query.",
    )
    parser.add_argument(
        "--eval-hard-negative-top-k",
        type=int,
        default=100,
        help="How many BM25 candidates to inspect per query while building dev samples.",
    )
    parser.add_argument(
        "--eval-hard-negatives-per-query",
        type=int,
        default=20,
        help="Number of negative documents per dev query for the reranking evaluator.",
    )
    parser.add_argument(
        "--train-qrels",
        type=Path,
        default=Path("data/nfcorpus/qrels/train.tsv"),
        help="Training qrels file.",
    )
    parser.add_argument(
        "--dev-qrels",
        type=Path,
        default=Path("data/nfcorpus/qrels/dev.tsv"),
        help="Development qrels file.",
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=Path("data/nfcorpus/queries.jsonl"),
        help="NFCorpus queries JSONL file.",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/en/corpus.jsonl"),
        help="Corpus JSONL file used for mining negatives.",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        help="Trainer logging backend. Use 'none' to disable W&B / TensorBoard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()

    train_qrels_path = (repo_root / args.train_qrels).resolve() if not args.train_qrels.is_absolute() else args.train_qrels.resolve()
    dev_qrels_path = (repo_root / args.dev_qrels).resolve() if not args.dev_qrels.is_absolute() else args.dev_qrels.resolve()
    queries_path = (repo_root / args.queries_path).resolve() if not args.queries_path.is_absolute() else args.queries_path.resolve()
    corpus_path = (repo_root / args.corpus_path).resolve() if not args.corpus_path.is_absolute() else args.corpus_path.resolve()

    if not train_qrels_path.exists():
        raise FileNotFoundError(f"Train qrels not found: {train_qrels_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and torch.cuda.is_bf16_supported()
    fp16 = bool(cuda_available and not bf16_supported)
    bf16 = bool(cuda_available and bf16_supported)

    logging.info("Loading corpus and queries")
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    corpus_by_id = {item["id"]: item for item in corpus}

    logging.info("Loading qrels")
    train_qrels_full = load_qrels(train_qrels_path)
    train_qrels, dev_qrels = split_train_dev_qids(train_qrels_full, dev_qrels_path, args.seed)

    logging.info("Building BM25 index over %d docs", len(corpus))
    bm25 = build_bm25(corpus)
    rng = random.Random(args.seed)

    train_dataset = build_training_dataset(
        qrels=train_qrels,
        queries=queries,
        corpus=corpus,
        corpus_by_id=corpus_by_id,
        bm25=bm25,
        rng=rng,
        hard_negative_top_k=args.train_hard_negative_top_k,
        hard_negatives_per_query=args.train_hard_negatives_per_query,
        random_negatives_per_query=args.train_random_negatives_per_query,
    )
    dev_dataset = build_training_dataset(
        qrels=dev_qrels,
        queries=queries,
        corpus=corpus,
        corpus_by_id=corpus_by_id,
        bm25=bm25,
        rng=rng,
        hard_negative_top_k=args.eval_hard_negative_top_k,
        hard_negatives_per_query=1,
        random_negatives_per_query=1,
    )
    dev_samples = build_reranking_samples(
        qrels=dev_qrels,
        queries=queries,
        corpus=corpus,
        corpus_by_id=corpus_by_id,
        bm25=bm25,
        rng=rng,
        hard_negative_top_k=args.eval_hard_negative_top_k,
        hard_negatives_per_query=args.eval_hard_negatives_per_query,
    )

    logging.info("Train pairs: %d", len(train_dataset))
    logging.info("Dev pairs: %d", len(dev_dataset))
    logging.info("Dev queries: %d", len(dev_samples))

    model = CrossEncoder(
        args.base_model,
        num_labels=1,
        max_length=args.max_length,
        model_kwargs={"torch_dtype": "float32"},
    )

    evaluator = CrossEncoderRerankingEvaluator(
        samples=dev_samples,
        at_k=args.eval_at_k,
        name="nfcorpus-dev",
        batch_size=args.eval_batch_size,
        show_progress_bar=True,
        write_csv=False,
    )

    baseline = evaluator(model)
    logging.info("Baseline metric (%s): %.4f", evaluator.primary_metric, baseline[evaluator.primary_metric])

    total_train_steps = max(1, math.ceil(len(train_dataset) / args.train_batch_size * args.epochs))
    warmup_steps = max(1, int(total_train_steps * args.warmup_ratio))

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        fp16=fp16,
        bf16=bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model=f"nfcorpus-dev_mrr@{args.eval_at_k}",
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        loss=BinaryCrossEntropyLoss(model),
        evaluator=evaluator,
    )

    trainer.train()

    final_metrics = evaluator(trainer.model)
    logging.info("Final metric (%s): %.4f", evaluator.primary_metric, final_metrics[evaluator.primary_metric])
    logging.info("Final MRR@%d: %.4f", args.eval_at_k, final_metrics[f"nfcorpus-dev_mrr@{args.eval_at_k}"])

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(
        str(final_dir),
        create_model_card=True,
        train_datasets=["NFCorpus train qrels"],
        safe_serialization=True,
    )

    summary = {
        "base_model": args.base_model,
        "output_dir": str(output_dir),
        "final_dir": str(final_dir),
        "train_pairs": len(train_dataset),
        "dev_queries": len(dev_samples),
        "baseline": baseline,
        "final_metrics": final_metrics,
    }
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("Saved final model to %s", final_dir)
    logging.info("Training summary written to %s", output_dir / "train_summary.json")


if __name__ == "__main__":
    main()