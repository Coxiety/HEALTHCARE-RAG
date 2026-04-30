"""Intent classifier — BERT fine-tuned on synthetic 3-class data.

CONTRACT: QueryClassifier().classify(text) -> "NUTRITION_LOOKUP" | "HEALTH_ADVICE" | "BOTH"
Falls back to keyword rules when BERT model is absent (pre-training).
"""
from __future__ import annotations

import os

import yaml

_CONFIG_PATH = "configs/config.yaml"
_MODEL_PATH_FALLBACK = "models/classifier_bert"

LABELS = ("NUTRITION_LOOKUP", "HEALTH_ADVICE", "BOTH")


def _classifier_model_path() -> str:
    try:
        cfg = yaml.safe_load(open(_CONFIG_PATH))
        return cfg.get("classifier_model_path", _MODEL_PATH_FALLBACK)
    except Exception:
        return _MODEL_PATH_FALLBACK


def _model_ready(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


class QueryClassifier:
    """3-class intent classifier.

    Uses fine-tuned BERT when model weights are present;
    falls back to keyword rules otherwise.
    """

    def __init__(self):
        model_path = _classifier_model_path()
        if _model_ready(model_path):
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                "text-classification",
                model=model_path,
                device=-1,
            )
            self._fallback: _RuleBasedFallback | None = None
        else:
            self._pipe = None
            self._fallback = _RuleBasedFallback()

    def classify(self, text: str) -> str:
        if self._fallback is not None:
            return self._fallback.classify(text)
        result = self._pipe(text, truncation=True, max_length=128)
        return result[0]["label"]


# ---------------------------------------------------------------------------
# Fallback — keyword rules, active until BERT model is present
# ---------------------------------------------------------------------------

_NUTRITION_KW = {
    "calorie", "protein", "carb", "fat", "vitamin", "mineral", "nutrient",
    "sodium", "fiber", "sugar", "iron", "calcium", "how much", "nutrition",
    "contain", "per 100g", "serving", "macro", "micronutrient", "cholesterol",
    "potassium", "magnesium", "zinc", "omega", "saturated",
}

_HEALTH_KW = {
    "should i eat", "good for", "avoid", "diet for", "recommend", "diabetes",
    "hypertension", "heart", "obese", "symptom", "disease", "condition",
    "treatment", "lose weight", "gain weight", "what to eat", "food for",
    "healthy", "benefit", "risk", "prevent", "manage", "blood pressure",
    "blood sugar", "inflammation", "digestion", "immunity",
}


class _RuleBasedFallback:
    def classify(self, text: str) -> str:
        t = text.lower()
        has_n = any(k in t for k in _NUTRITION_KW)
        has_h = any(k in t for k in _HEALTH_KW)
        if has_n and has_h:
            return "BOTH"
        if has_n:
            return "NUTRITION_LOOKUP"
        return "HEALTH_ADVICE"
