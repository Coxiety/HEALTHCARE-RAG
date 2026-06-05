from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

# Add repository root to sys.path to allow running the file directly
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_FOOD_PREP_RE = re.compile(r"\b(?:in|of|for)\s+([a-z][a-z\s]{1,30}?)\s*$", re.IGNORECASE)


@dataclass
class EvalCase:
    question: str
    reference_answer: str = ""
    relevant_docs: list[str] | None = None
    expected_intent: str = ""
    expected_answer_keywords: list[str] | None = None
    keyword_match: str = "any"
    expected_source_keywords: list[str] | None = None
    source_match: str = "any"


@dataclass
class EvalResult:
    question: str
    reference_answer: str
    predicted_answer: str
    expected_intent: str
    predicted_intent: str
    relevant_doc_ids: str
    retrieved_doc_ids: str
    predicted_sources: str
    latency_sec: float
    intent_correct: bool | None
    retrieval_hit_at_k: bool | None
    retrieval_recall_at_k: float | None
    retrieval_precision_at_k: float | None
    retrieval_mrr_at_k: float | None
    answer_exact_match: bool | None
    answer_token_f1: float | None
    answer_keyword_hit: bool | None
    source_keyword_hit: bool | None
    notes: str = ""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return value != value
    except Exception:
        return False


def _first_text(*values: Any) -> str:
    for value in values:
        if _is_missing(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _coerce_list(value: Any) -> list[str]:
    if _is_missing(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith(("[", "(")):
        parsed: Any = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None

        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]

    if any(sep in text for sep in ("|", ";", ",")):
        parts = re.split(r"\s*[|;,]\s*", text)
        if len(parts) > 1:
            return [part.strip().strip("'\"") for part in parts if part.strip()]

    return [text]


def _normalize_answer(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower())).strip()


def _token_f1(prediction: str, reference: str) -> float | None:
    pred_tokens = _WORD_RE.findall(prediction.lower())
    ref_tokens = _WORD_RE.findall(reference.lower())
    if not pred_tokens or not ref_tokens:
        return None

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = sum(min(pred_counts[token], ref_counts.get(token, 0)) for token in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _extract_food_from_query(query: str) -> str | None:
    query = query.rstrip("?.! ").strip()
    match = _FOOD_PREP_RE.search(query)
    return match.group(1).strip() if match else None


def _load_records(input_path: Path) -> list[dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path).to_dict(orient="records")

    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with open(input_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if suffix == ".json":
        with open(input_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported JSON structure in {input_path}: expected a list of records")

    raise ValueError(f"Unsupported input format: {input_path.suffix} (use .jsonl, .json, or .csv)")


def _normalize_case(record: dict[str, Any]) -> EvalCase:
    question = _first_text(record.get("question"), record.get("query"), record.get("text"), record.get("input"))
    if not question:
        raise ValueError("Missing question text in evaluation record")

    reference_answer = _first_text(
        record.get("reference_answer"),
        record.get("gold_answer"),
        record.get("answer"),
        record.get("output"),
    )

    relevant_docs = _coerce_list(record.get("relevant_docs") or record.get("relevant_doc_ids") or record.get("qrels"))
    expected_intent = _first_text(record.get("expected_intent"), record.get("intent"), record.get("label"))
    expected_answer_keywords = _coerce_list(
        record.get("expected_answer_keywords")
        or record.get("answer_keywords")
        or record.get("keywords")
    )
    expected_source_keywords = _coerce_list(
        record.get("expected_source_keywords")
        or record.get("source_keywords")
        or record.get("expected_sources")
    )

    keyword_match = _first_text(record.get("keyword_match"), record.get("answer_match"), "any").lower()
    source_match = _first_text(record.get("source_match"), "any").lower()

    return EvalCase(
        question=question,
        reference_answer=reference_answer,
        relevant_docs=relevant_docs or None,
        expected_intent=expected_intent,
        expected_answer_keywords=expected_answer_keywords or None,
        keyword_match=keyword_match if keyword_match in {"any", "all"} else "any",
        expected_source_keywords=expected_source_keywords or None,
        source_match=source_match if source_match in {"any", "all"} else "any",
    )


def _load_text_to_id_map(corpus_path: Path) -> dict[str, str]:
    if not corpus_path.exists():
        return {}

    mapping: dict[str, str] = {}
    with open(corpus_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "")
            doc_id = record.get("id", "")
            if text and doc_id:
                mapping[text] = doc_id
    return mapping


def _mean_or_none(values: list[float]) -> float | None:
    return round(fmean(values), 4) if values else None


def _load_pipeline(config_path: str) -> Any:
    from src.en.pipeline import ENPipeline

    return ENPipeline(config_path)


class PipelineRetriever:
    def __init__(self, config_path: str):
        import yaml
        from src.database.vector_store import VectorStore
        from src.en.retriever import BM25Retriever, DenseRetriever, HybridRetriever
        from src.en.reranker import Reranker

        cfg = yaml.safe_load(open(config_path))
        self.vs = VectorStore(
            cfg["chroma_persist_dir"],
            cfg["chroma_collection"],
            cfg["embedding_model"],
        )
        bm25 = BM25Retriever()
        dense = DenseRetriever(self.vs)
        self.retriever = HybridRetriever(bm25, dense)
        self.reranker = Reranker(cfg.get("reranker_model"))

    def retrieve(self, query: str, top_k: int = 5) -> list[Any]:
        candidates = self.retriever.retrieve(query, top_k=max(top_k * 4, 20))
        return self.reranker.rerank(query, candidates, top_k=top_k)


def _load_retriever(config_path: str) -> Any:
    return PipelineRetriever(config_path)


def _evaluate_retrieval_case(
    retriever: Any,
    case: EvalCase,
    text_to_id: dict[str, str],
    top_k: int,
) -> tuple[EvalResult, list[str]]:
    start = time.perf_counter()
    final_chunks = retriever.retrieve(case.question, top_k=top_k)
    latency = time.perf_counter() - start

    retrieved_ids = [text_to_id.get(chunk.text, "") for chunk in final_chunks]
    relevant_ids = case.relevant_docs or []
    relevant_set = set(relevant_ids)

    hit_count = sum(1 for doc_id in retrieved_ids if doc_id and doc_id in relevant_set)
    first_hit_rank = next((rank for rank, doc_id in enumerate(retrieved_ids, start=1) if doc_id in relevant_set), None)

    retrieval_hit_at_k = bool(hit_count) if relevant_ids else None
    retrieval_recall_at_k = round(hit_count / len(relevant_ids), 4) if relevant_ids else None
    retrieval_precision_at_k = round(hit_count / len([doc_id for doc_id in retrieved_ids if doc_id]), 4) if retrieved_ids else None
    retrieval_mrr_at_k = round(1.0 / first_hit_rank, 4) if first_hit_rank else None

    result = EvalResult(
        question=case.question,
        reference_answer=case.reference_answer,
        predicted_answer="",
        expected_intent=case.expected_intent,
        predicted_intent="",
        relevant_doc_ids="|".join(relevant_ids),
        retrieved_doc_ids="|".join(doc_id for doc_id in retrieved_ids if doc_id),
        predicted_sources="",
        latency_sec=round(latency, 4),
        intent_correct=None,
        retrieval_hit_at_k=retrieval_hit_at_k,
        retrieval_recall_at_k=retrieval_recall_at_k,
        retrieval_precision_at_k=retrieval_precision_at_k,
        retrieval_mrr_at_k=retrieval_mrr_at_k,
        answer_exact_match=None,
        answer_token_f1=None,
        answer_keyword_hit=None,
        source_keyword_hit=None,
        notes="retrieval-only",
    )
    return result, retrieved_ids


def _evaluate_full_case(
    pipeline: Any,
    case: EvalCase,
    text_to_id: dict[str, str],
    top_k: int,
) -> tuple[EvalResult, list[str]]:
    start = time.perf_counter()

    predicted_intent = pipeline.clf.classify(case.question)
    entities = pipeline.ner.predict(case.question)
    nutrition_data = None
    chunks: list[Any] = []

    if predicted_intent in ("NUTRITION_LOOKUP", "BOTH"):
        foods = entities.get("FOOD", [])
        if not foods:
            extracted = _extract_food_from_query(case.question)
            if extracted:
                foods = [extracted]
        if foods:
            nutrition_data = pipeline.db.lookup_en(foods[0])

    if predicted_intent in ("HEALTH_ADVICE", "BOTH"):
        candidates = pipeline.retriever.retrieve(case.question, top_k=max(top_k * 4, 20))
        chunks = pipeline.reranker.rerank(case.question, candidates, top_k=top_k)

    generation = pipeline.generator.generate(case.question, nutrition_data, chunks, predicted_intent)
    predicted_answer = generation.get("answer", "")
    predicted_sources = generation.get("sources", []) or []
    latency = time.perf_counter() - start

    retrieved_ids = [text_to_id.get(chunk.text, "") for chunk in chunks]
    relevant_ids = case.relevant_docs or []
    relevant_set = set(relevant_ids)

    hit_count = sum(1 for doc_id in retrieved_ids if doc_id and doc_id in relevant_set)
    first_hit_rank = next((rank for rank, doc_id in enumerate(retrieved_ids, start=1) if doc_id in relevant_set), None)

    retrieval_hit_at_k = bool(hit_count) if relevant_ids else None
    retrieval_recall_at_k = round(hit_count / len(relevant_ids), 4) if relevant_ids else None
    retrieval_precision_at_k = round(hit_count / len([doc_id for doc_id in retrieved_ids if doc_id]), 4) if retrieved_ids else None
    retrieval_mrr_at_k = round(1.0 / first_hit_rank, 4) if first_hit_rank else None

    expected_answer_keywords = case.expected_answer_keywords or []
    expected_source_keywords = case.expected_source_keywords or []
    answer_keyword_hit: bool | None = None
    source_keyword_hit: bool | None = None

    if expected_answer_keywords:
        answer_blob = predicted_answer.lower()
        if case.keyword_match == "all":
            answer_keyword_hit = all(keyword.lower() in answer_blob for keyword in expected_answer_keywords)
        else:
            answer_keyword_hit = any(keyword.lower() in answer_blob for keyword in expected_answer_keywords)

    if expected_source_keywords:
        source_blob = " ".join(predicted_sources).lower()
        if case.source_match == "all":
            source_keyword_hit = all(keyword.lower() in source_blob for keyword in expected_source_keywords)
        else:
            source_keyword_hit = any(keyword.lower() in source_blob for keyword in expected_source_keywords)

    answer_exact_match = None
    answer_token_f1 = None
    if case.reference_answer:
        answer_exact_match = _normalize_answer(predicted_answer) == _normalize_answer(case.reference_answer)
        answer_token_f1 = _token_f1(predicted_answer, case.reference_answer)

    intent_correct = predicted_intent.upper() == case.expected_intent.upper() if case.expected_intent else None

    result = EvalResult(
        question=case.question,
        reference_answer=case.reference_answer,
        predicted_answer=predicted_answer,
        expected_intent=case.expected_intent,
        predicted_intent=predicted_intent,
        relevant_doc_ids="|".join(relevant_ids),
        retrieved_doc_ids="|".join(doc_id for doc_id in retrieved_ids if doc_id),
        predicted_sources="|".join(sorted(set(str(source) for source in predicted_sources if str(source).strip()))),
        latency_sec=round(latency, 4),
        intent_correct=intent_correct,
        retrieval_hit_at_k=retrieval_hit_at_k,
        retrieval_recall_at_k=retrieval_recall_at_k,
        retrieval_precision_at_k=retrieval_precision_at_k,
        retrieval_mrr_at_k=retrieval_mrr_at_k,
        answer_exact_match=answer_exact_match,
        answer_token_f1=round(answer_token_f1, 4) if answer_token_f1 is not None else None,
        answer_keyword_hit=answer_keyword_hit,
        source_keyword_hit=source_keyword_hit,
        notes="full",
    )
    return result, retrieved_ids


def _summarize(results: list[EvalResult], mode: str, top_k: int, total_input_rows: int) -> dict[str, Any]:
    def bool_values(attr: str) -> list[float]:
        values: list[float] = []
        for row in results:
            value = getattr(row, attr)
            if value is None:
                continue
            values.append(1.0 if value else 0.0)
        return values

    def float_values(attr: str) -> list[float]:
        values: list[float] = []
        for row in results:
            value = getattr(row, attr)
            if value is not None:
                values.append(float(value))
        return values

    metrics = {
        "avg_latency_sec": _mean_or_none(float_values("latency_sec")),
        "intent_accuracy": _mean_or_none(bool_values("intent_correct")),
        "retrieval_hit_rate": _mean_or_none(bool_values("retrieval_hit_at_k")),
        "retrieval_recall_at_k": _mean_or_none(float_values("retrieval_recall_at_k")),
        "retrieval_precision_at_k": _mean_or_none(float_values("retrieval_precision_at_k")),
        "retrieval_mrr_at_k": _mean_or_none(float_values("retrieval_mrr_at_k")),
        "answer_exact_match_rate": _mean_or_none(bool_values("answer_exact_match")),
        "answer_token_f1": _mean_or_none(float_values("answer_token_f1")),
        "answer_keyword_hit_rate": _mean_or_none(bool_values("answer_keyword_hit")),
        "source_keyword_hit_rate": _mean_or_none(bool_values("source_keyword_hit")),
    }

    metrics = {key: value for key, value in metrics.items() if value is not None}

    return {
        "mode": mode,
        "top_k": top_k,
        "total_input_rows": total_input_rows,
        "evaluated_rows": len(results),
        "counts": {
            "with_reference_answer": sum(1 for row in results if row.reference_answer),
            "with_relevant_docs": sum(1 for row in results if row.relevant_doc_ids),
            "with_expected_intent": sum(1 for row in results if row.expected_intent),
            "with_expected_answer_keywords": sum(1 for row in results if row.answer_keyword_hit is not None),
            "with_expected_source_keywords": sum(1 for row in results if row.source_keyword_hit is not None),
        },
        "metrics": metrics,
    }


def run_evaluation(
    input_path: str,
    output_dir: str,
    config_path: str = "configs/config.yaml",
    corpus_path: str = "data/en/corpus.jsonl",
    top_k: int = 5,
    limit: int | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_records = _load_records(input_file)
    cases = [_normalize_case(record) for record in raw_records]
    if limit is not None:
        cases = cases[:limit]

    text_to_id = _load_text_to_id_map(Path(corpus_path))
    pipeline = _load_pipeline(config_path) if mode == "full" else None
    retriever = _load_retriever(config_path) if mode == "retrieval" else None

    results: list[EvalResult] = []
    print(f"Running {mode} evaluation on {len(cases)} record(s)...")
    for index, case in enumerate(cases, start=1):
        safe_question = case.question[:80].encode("ascii", "replace").decode()
        print(f"  [{index}/{len(cases)}] {safe_question}")
        if mode == "retrieval":
            assert retriever is not None
            result, _ = _evaluate_retrieval_case(retriever, case, text_to_id, top_k)
        else:
            assert pipeline is not None
            result, _ = _evaluate_full_case(pipeline, case, text_to_id, top_k)
        results.append(result)

    summary = _summarize(results, mode, top_k, len(raw_records))
    summary["input_path"] = str(input_file)
    summary["output_dir"] = str(output_path)
    summary["config_path"] = str(config_path)
    summary["corpus_path"] = str(corpus_path)

    summary_path = output_path / "summary.json"
    cases_path = output_path / "cases.csv"
    summary["summary_path"] = str(summary_path)
    summary["cases_path"] = str(cases_path)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    pd.DataFrame([asdict(row) for row in results]).to_csv(cases_path, index=False)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("Evaluation complete")
    print(f"Mode        : {summary['mode']}")
    print(f"Top-k       : {summary['top_k']}")
    print(f"Rows        : {summary['evaluated_rows']}/{summary['total_input_rows']}")
    print(f"Output dir  : {summary['output_dir']}")
    print(f"Summary     : {summary['summary_path']}")
    print(f"Cases       : {summary['cases_path']}")
    print()

    metrics = summary.get("metrics", {})
    if not metrics:
        print("No metrics were available for the selected input.")
        return

    for key in sorted(metrics):
        print(f"{key:<28} {metrics[key]}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatic end-to-end evaluation for the EN RAG pipeline")
    parser.add_argument("--input", default="data/en/eval_200.jsonl", help="Input QA dataset (.jsonl, .json, or .csv)")
    parser.add_argument("--output-dir", default="reports/en/rag_eval", help="Directory where the report will be written")
    parser.add_argument("--config", default="configs/config.yaml", help="Pipeline config file")
    parser.add_argument("--corpus-path", default="data/en/corpus.jsonl", help="Corpus file used to map retrieved texts back to doc IDs")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final chunks kept after reranking")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows")
    parser.add_argument("--mode", choices=("full", "retrieval"), default="full", help="full = route + retrieval + generation, retrieval = retrieval only")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    summary = run_evaluation(
        input_path=args.input,
        output_dir=args.output_dir,
        config_path=args.config,
        corpus_path=args.corpus_path,
        top_k=args.top_k,
        limit=args.limit,
        mode=args.mode,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())