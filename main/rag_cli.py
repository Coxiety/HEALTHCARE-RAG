from __future__ import annotations

import json
import os
import sys

# Spring Boot chạy script từ thư mục main/ → cần add parent vào sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# silence noisy warnings
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Java dùng redirectErrorStream(true) → stderr bị gộp vào stdout
# → phải redirect cả stdout lẫn stderr sang /dev/null khi load model
_real_stdout = sys.stdout
_devnull = open(os.devnull, "w", encoding="utf-8")


def _silence():
    sys.stdout = _devnull
    sys.stderr = _devnull


def _restore():
    sys.stdout = _real_stdout
    # stderr giữ nguyên devnull — không cần hiển thị warning khi chạy qua Java


_silence()
from src.pipeline import NutritionPipeline
_restore()

_pipeline: NutritionPipeline | None = None


def get_pipeline() -> NutritionPipeline:
    global _pipeline
    if _pipeline is None:
        _silence()
        _pipeline = NutritionPipeline()
        _restore()
    return _pipeline


def main() -> None:
    # force UTF-8 on Windows stdout
    try:
        if hasattr(_real_stdout, "reconfigure"):
            _real_stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing query"}), file=_real_stdout)
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    try:
        pipeline = get_pipeline()
        result = pipeline.answer(query)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "answer": "Hệ thống gặp lỗi, vui lòng thử lại."}),
              file=_real_stdout)
        sys.exit(1)

    ner_entities = result.get("entities", {})
    entities_out = {
        "foods":     ner_entities.get("FOOD", []),
        "diseases":  ner_entities.get("DISEASE", []),
        "nutrients": ner_entities.get("NUTRIENT", []),
        "symptoms":  ner_entities.get("SYMPTOM", []),
    }

    energy = None
    nutrition_data = result.get("nutrition_data")
    if nutrition_data and isinstance(nutrition_data, dict):
        energy = {
            "amountPer100g": nutrition_data.get("amount_per_100g"),
            "unitName":      nutrition_data.get("unit"),
        }

    output = {
        "answer":   result.get("answer", ""),
        "intent":   result.get("query_type", ""),
        "entities": entities_out,
        "sources":  sorted(set(result.get("sources", []))),
        "energy":   energy,
    }

    print(json.dumps(output, ensure_ascii=False), file=_real_stdout)


if __name__ == "__main__":
    main()
