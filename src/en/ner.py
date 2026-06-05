"""NER model — BioBERT fine-tuned on BC5CDR for NUTRIENT and DISEASE detection.

CONTRACT: NERModel().predict(text) -> {"FOOD": list, "DISEASE": list, "NUTRIENT": list, "SYMPTOM": list}
Falls back to spaCy keyword lookup when BioBERT model is absent (pre-training).
"""

from __future__ import annotations

import os

import yaml

_CONFIG_PATH = "configs/config.yaml"
_MODEL_PATH_FALLBACK = "models/en/ner_bert"

# entity_group values from aggregation_strategy="simple" → project entity type
_LABEL_TO_TYPE: dict[str, str] = {
    "NUTRIENT": "NUTRIENT",
    "DISEASE":  "DISEASE",
    "FOOD":     "FOOD",
    "SYMPTOM":  "SYMPTOM",
}


def _ner_model_path() -> str:
    try:
        cfg = yaml.safe_load(open(_CONFIG_PATH))
        return cfg.get("ner_model_path", _MODEL_PATH_FALLBACK)
    except Exception:
        return _MODEL_PATH_FALLBACK


def _model_ready(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


class NERModel:
    """Token-classification NER.

    Uses BioBERT (fine-tuned on BC5CDR) when model weights are present;
    falls back to spaCy keyword matching otherwise.
    """

    def __init__(self):
        model_path = _ner_model_path()
        if _model_ready(model_path):
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                "token-classification",
                model=model_path,
                aggregation_strategy="first",
                device=-1,
            )
            self._fallback: _SpacyFallback | None = None
        else:
            self._pipe = None
            self._fallback = _SpacyFallback()

    def predict(self, text: str) -> dict[str, list[str]]:
        if self._fallback is not None:
            return self._fallback.predict(text)
        
        # Truncate text to a safe length (e.g. 150 words) to avoid exceeding BERT's 512 token limit
        words = text.split()
        if len(words) > 150:
            text = " ".join(words[:150])
            
        try:
            return _run_bert(self._pipe, text)
        except Exception:
            # Graceful fallback to spaCy keyword matcher if BERT model fails
            return _SpacyFallback().predict(text)


def _run_bert(pipe, text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"FOOD": [], "DISEASE": [], "NUTRIENT": [], "SYMPTOM": []}
    for ent in pipe(text):
        etype = _LABEL_TO_TYPE.get(ent["entity_group"])
        word  = ent["word"].strip()
        if etype and word and word not in result[etype]:
            result[etype].append(word)
    return result


# ---------------------------------------------------------------------------
# Fallback — spaCy keyword MVP, active until BioBERT training completes
# ---------------------------------------------------------------------------

_FOOD_TOKENS: set[str] = {
    "chicken", "rice", "beef", "pork", "fish", "salmon", "tuna", "shrimp",
    "broccoli", "spinach", "carrot", "tomato", "potato", "onion", "garlic",
    "egg", "milk", "cheese", "yogurt", "butter", "bread", "oat", "oatmeal",
    "apple", "banana", "orange", "grape", "strawberry", "blueberry",
    "almond", "walnut", "peanut", "tofu", "lentil", "bean", "quinoa",
}
_DISEASE_TOKENS: set[str] = {
    "diabetes", "hypertension", "obesity", "gout", "anemia", "arthritis",
    "cancer", "cholesterol", "osteoporosis", "asthma", "depression",
    "anxiety", "insomnia", "constipation", "diarrhea", "gastritis",
    "celiac", "ibd", "ibs", "ckd", "nafld",
}
_NUTRIENT_TOKENS: set[str] = {
    "protein", "carbohydrate", "fat", "fiber", "calorie", "vitamin",
    "mineral", "calcium", "iron", "sodium", "potassium", "magnesium",
    "zinc", "phosphorus", "folate", "omega-3", "omega-6", "antioxidant",
    "cholesterol", "glucose", "fructose", "lactose",
}
_SYMPTOM_TOKENS: set[str] = {
    "fatigue", "nausea", "headache", "dizziness", "bloating", "cramp",
    "insomnia", "weakness", "swelling", "inflammation", "pain", "fever",
}


class _SpacyFallback:
    def predict(self, text: str) -> dict[str, list[str]]:
        words = set(text.lower().split())
        return {
            "FOOD":     list(words & _FOOD_TOKENS),
            "DISEASE":  list(words & _DISEASE_TOKENS),
            "NUTRIENT": list(words & _NUTRIENT_TOKENS),
            "SYMPTOM":  list(words & _SYMPTOM_TOKENS),
        }
