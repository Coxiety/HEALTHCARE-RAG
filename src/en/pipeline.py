from __future__ import annotations

import re
import yaml

from src.database.sqlite_manager import SqliteManager
from src.database.vector_store import VectorStore
from src.en.classifier import QueryClassifier
from src.en.ner import NERModel
from src.en.preprocessor import Preprocessor
from src.en.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from src.en.reranker import Reranker
from src.generation.generator import Generator

# NER misses food names (trained on BC5CDR Chemical/Disease only) — regex fallback
_FOOD_PREP_RE = re.compile(
    r'\b(?:in|of|for)\s+([a-z][a-z\s]{1,30}?)\s*$',
    re.IGNORECASE,
)


def _extract_food_from_query(query: str) -> str | None:
    """Extract food name from query when NER returns no FOOD entities."""
    q = query.rstrip("?.! ").strip()
    m = _FOOD_PREP_RE.search(q)
    return m.group(1).strip() if m else None


class ENPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg = yaml.safe_load(open(config_path))

        self.prep = Preprocessor()
        self.clf = QueryClassifier()
        self.ner = NERModel()

        self.vs = VectorStore(
            cfg["chroma_persist_dir"],
            cfg["chroma_collection"],
            cfg["embedding_model"],
        )
        bm25 = BM25Retriever()
        dense = DenseRetriever(self.vs)
        self.retriever = HybridRetriever(bm25, dense)
        self.reranker = Reranker()
        self.db = SqliteManager(cfg["sqlite_path"])
        self.generator = Generator(model=cfg["llm_model"])
        self.top_k = cfg.get("top_k", 5)

    def answer(self, query: str) -> dict:
        intent = self.clf.classify(query)
        entities = self.ner.predict(query)
        nutrition = None
        chunks = []

        if intent in ("NUTRITION_LOOKUP", "BOTH"):
            foods = entities.get("FOOD", [])
            if not foods:
                extracted = _extract_food_from_query(query)
                if extracted:
                    foods = [extracted]
            if foods:
                nutrition = self.db.lookup_en(foods[0])

        if intent in ("HEALTH_ADVICE", "BOTH"):
            candidates = self.retriever.retrieve(query, top_k=20)
            chunks = self.reranker.rerank(query, candidates, top_k=self.top_k)

        result = self.generator.generate(query, nutrition, chunks, intent)
        result.update({"intent": intent, "entities": entities})
        return result
