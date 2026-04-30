from __future__ import annotations

import os

import yaml

from src.nlp.preprocessor import VietnamesePreprocessor
from src.nlp.classifier import QueryClassifier, QueryType
from src.nlp.retriever import HybridRetriever
from src.nlp.ner_model import load_ner_model
from src.database.sqlite_manager import SqliteManager
from src.database.vector_store import VectorStore
from src.data_pipeline.embedder import Embedder
from src.generation.generator import Generator


# Mapping từ khóa tiếng Việt → tên nutrient trong USDA
NUTRIENT_MAP = {
    "protein":    "Protein",
    "đạm":        "Protein",
    "calo":       "Energy",
    "calorie":    "Energy",
    "kcal":       "Energy",
    "chất béo":   "Total lipid (fat)",
    "tinh bột":   "Carbohydrate, by difference",
    "carb":       "Carbohydrate, by difference",
    "vitamin c":  "Vitamin C, total ascorbic acid",
    "vitamin d":  "Vitamin D (D2 + D3)",
    "canxi":      "Calcium, Ca",
    "sắt":        "Iron, Fe",
    "kali":       "Potassium, K",
    "natri":      "Sodium, Na",
    "kẽm":        "Zinc, Zn",
    "chất xơ":    "Fiber, total dietary",
    "magiê":      "Magnesium, Mg",
    "omega-3":    "Fatty acids, total polyunsaturated",
    "omega 3":    "Fatty acids, total polyunsaturated",
}


class NutritionPipeline:

    def __init__(self, config_path: str = None):
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if config_path is None:
            config_path = os.path.join(ROOT, "configs", "config.yaml")
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        def p(key):
            return os.path.join(ROOT, cfg[key])

        self.preprocessor = VietnamesePreprocessor()
        self.classifier   = QueryClassifier()
        self.sqlite       = SqliteManager(p("sqlite_path"), p("vi_mapping_path"))
        self.generator    = Generator(model=cfg["llm_model"])

        self.vs = VectorStore(
            persist_dir=p("chroma_persist_dir"),
            collection_name=cfg["chroma_collection"],
            embedding_model=cfg["embedding_model"],
        )
        self.retriever = HybridRetriever(
            self.vs,
            top_k=cfg["top_k"],
            score_threshold=cfg.get("similarity_threshold", 0.3),
        )
        self.ner = load_ner_model(cfg, ROOT)

        self._raw_articles_path = p("raw_articles_path") if "raw_articles_path" in cfg else None
        self._init_vector_store(p("medical_docs_path"))

    def _init_vector_store(self, docs_path: str) -> None:
        # Chỉ embed lần đầu, những lần sau đọc từ ChromaDB persist trên disk
        if self.vs.count() == 0:
            embedder = Embedder(
                self.vs,
                chunk_size=500,
                chunk_overlap=100,
            )
            # Ưu tiên embed 183 bài raw/articles (đầy đủ hơn)
            # Fallback về medical_knowledge.jsonl nếu thư mục không tồn tại
            import os
            if self._raw_articles_path and os.path.isdir(self._raw_articles_path):
                n = embedder.embed_directory(self._raw_articles_path)
                print(f"[Pipeline] embedded {n} chunks from {self._raw_articles_path}")
            else:
                n = embedder.embed_jsonl(docs_path)
                print(f"[Pipeline] embedded {n} chunks from {docs_path}")

        self.retriever.sync_bm25_from_store()

    def _extract_food(self, query: str) -> str | None:
        # Fallback keyword matching khi NER không có
        q = query.lower()
        for vi_name in self.sqlite.list_mapped_foods():
            if vi_name in q:
                return vi_name
        return None

    def _extract_nutrient(self, query: str) -> str | None:
        # Fallback keyword matching khi NER không có
        q = query.lower()
        for keyword, usda_name in NUTRIENT_MAP.items():
            if keyword in q:
                return usda_name
        return None

    def _lookup_nutrition_ner(self, entities: dict) -> dict | None:
        # Chuẩn hóa: "ức_gà" → "ức gà"
        food_tokens = [f.replace("_", " ") for f in entities.get("FOOD", [])]

        # Underthesea đôi khi không nối compound word → thử cả cụm ghép
        # vd: ["ức", "gà"] → thử thêm "ức gà"
        food_candidates = list(food_tokens)
        if len(food_tokens) > 1:
            food_candidates.append(" ".join(food_tokens))

        for food in food_candidates:
            for nutrient_token in entities.get("NUTRIENT", []):
                normalized = nutrient_token.replace("_", " ").replace("-", " ").lower()
                usda_name = NUTRIENT_MAP.get(normalized)
                if usda_name is None:
                    for kw, name in NUTRIENT_MAP.items():
                        if kw in normalized or normalized in kw:
                            usda_name = name
                            break
                if usda_name:
                    data = self.sqlite.lookup(food, usda_name)
                    if data:
                        return data
        return None

    _GREETING_PATTERNS = [
        "chào", "xin chào", "hello", "hi", "hey",
        "alo", "good morning", "good afternoon", "good evening",
        "helo", "hê lô",
    ]
    _GREETING_REPLY = (
        "Xin chào! Tôi là trợ lý dinh dưỡng. "
        "Bạn có thể hỏi tôi về thành phần dinh dưỡng của thực phẩm "
        "hoặc lời khuyên sức khỏe. Tôi có thể giúp gì cho bạn?"
    )

    def _is_greeting(self, query: str) -> bool:
        q = query.strip().lower()
        return any(q == p or q.startswith(p + " ") or q.startswith(p + "!") for p in self._GREETING_PATTERNS)

    def answer(self, query: str) -> dict:
        if self._is_greeting(query):
            return {
                "answer":         self._GREETING_REPLY,
                "sources":        [],
                "used_llm":       False,
                "query_type":     "GREETING",
                "entities":       {},
                "nutrition_data": None,
            }

        processed  = self.preprocessor.preprocess(query)
        clf_result = self.classifier.classify(processed)
        query_type = clf_result.query_type

        nutrition_data = None
        health_chunks  = []

        # Chạy NER một lần, dùng cho cả hai nhánh
        entities = self.ner.predict(processed) if self.ner else {}

        if query_type in (QueryType.NUTRITION_LOOKUP, QueryType.BOTH):
            if entities:
                nutrition_data = self._lookup_nutrition_ner(entities)
            else:
                # Fallback keyword
                food_name     = self._extract_food(processed)
                nutrient_name = self._extract_nutrient(processed)
                if food_name and nutrient_name:
                    nutrition_data = self.sqlite.lookup(food_name, nutrient_name)

        if query_type in (QueryType.HEALTH_ADVICE, QueryType.BOTH):
            health_chunks = self.retriever.retrieve(query)  # dùng query gốc — sạch hơn cho semantic search

        result = self.generator.generate(query, nutrition_data, health_chunks, query_type=query_type.value)
        result["query_type"]     = query_type.value
        result["entities"]       = entities
        result["nutrition_data"] = nutrition_data
        return result


if __name__ == "__main__":
    pipeline = NutritionPipeline()

    tests = [
        "100g ức gà có bao nhiêu protein?",
        "bị tiểu đường nên ăn gì?",
        "người bị gout ăn hải sản được không?",
    ]

    for q in tests:
        print(f"\nQ: {q}")
        r = pipeline.answer(q)
        print(f"   type    : {r['query_type']}")
        print(f"   entities: {r['entities']}")
        print(f"   answer  : {r['answer'][:120]}...")
        print(f"   sources : {r['sources']}")
