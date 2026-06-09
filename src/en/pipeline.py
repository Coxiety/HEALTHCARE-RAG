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

import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import sys
    print("[RAG Server] Downloading spaCy model 'en_core_web_sm'...", file=sys.stderr)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

_EXCLUDED_NOUNS = {
    # Nutrients & Generic Medical
    "protein", "proteins", "calorie", "calories", "fat", "fats", "lipid", "lipids", "carbs", "carb", "carbohydrate", 
    "carbohydrates", "fiber", "fibers", "energy", "cholesterol", "sugar", "sugars", "sodium", "vitamin", "vitamins",
    "calcium", "iron", "potassium", "magnesium", "zinc", "phosphorus", "folate", "antioxidant", "glucose", "fructose", "lactose",
    "disease", "condition", "blood", "pressure", "inflammation", "symptom", "treatment",
    
    # Generic measurements/words
    "amount", "amounts", "grams", "gram", "ounces", "ounce", "serving", "servings", "piece", "pieces", "portion", "portions",
    "people", "person", "body", "health", "diet", "meal", "food", "foods", "drink", "drinks", "water",
    "type", "types", "kind", "kinds", "sort", "sorts", "value", "values", "nutrition", "nutrient", "nutrients",
    "info", "information", "fact", "facts", "data", "source", "sources", "content", "contents", "benefit", "risk"
}

def _extract_foods_from_query(query: str) -> list[str]:
    """Extract food names from query using spaCy Noun Chunks."""
    doc = nlp(query)
    foods = []
    for chunk in doc.noun_chunks:
        root_lemma = chunk.root.lemma_.lower()
        if root_lemma in _EXCLUDED_NOUNS or chunk.root.pos_ == "PRON":
            continue
        
        text = chunk.text.lower()
        # Remove common determiners
        if text.startswith("a "): text = text[2:]
        elif text.startswith("an "): text = text[3:]
        elif text.startswith("the "): text = text[4:]
        
        if text and text not in foods:
            foods.append(text)
    return foods


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
        self.reranker = Reranker(cfg.get("reranker_model"))
        self.db = SqliteManager(cfg["sqlite_path"])

        self.rewriter_model = cfg.get("rewriter_model", "llama3.1:8b")

        self.generator = Generator(
            model=cfg["llm_model"],
            host=cfg.get("ollama_host", "http://localhost:11434")
        )
        self.top_k = cfg.get("top_k", 5)

    def _condense_query(self, query: str, history: list[dict]) -> str:
        """Sử dụng Ollama để viết lại câu hỏi dựa trên lịch sử hội thoại."""
        if not history:
            return query

        history_text = ""
        for msg in history[-5:]:  # lấy tối đa 5 tin nhắn gần nhất
            history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

        prompt = (
            "Given the following conversation history and a follow-up question, "
            "rephrase the follow-up question to be a standalone question (in English) "
            "that captures all necessary context.\n"
            "Rules:\n"
            "1. If the follow-up question contains pronouns (e.g., 'it', 'its', 'they', 'this', 'that') or is a contextual query (e.g., 'what about calories?', 'how about fat?'), you MUST rewrite it to explicitly include the food/context from the history (for example, replace 'its' with 'salmon').\n"
            "2. If the follow-up question is already a standalone question that is fully explicit and contains no pronouns, output the follow-up question EXACTLY as it is without adding any unnecessary context.\n"
            "3. If the follow-up question introduces a completely new topic or food (e.g. 'Is garlic help lower blood pressure?'), do NOT carry over context from the previous questions. Treat it as a new question and output it EXACTLY as it is.\n"
            "Do NOT answer the question. Only output the rephrased standalone question.\n\n"
            f"Chat History:\n{history_text}"
            f"Follow-up Question: {query}\n\n"
            "Standalone Question:"
        )
        condensed = self.generator._call_ollama_generate(prompt)
        return condensed.strip() if condensed else query


    def answer(self, query: str, history: list[dict] = None) -> dict:
        search_query = self._condense_query(query, history)
        print(f"[DEBUG] Original Query: '{query}' -> Condensed/Rewritten Query: '{search_query}'")

        intent = self.clf.classify(search_query)
        entities = self.ner.predict(search_query)
        nutrition = None
        chunks = []

        if intent in ("NUTRITION_LOOKUP", "BOTH"):
            foods = entities.get("FOOD", [])
            if not foods:
                foods = _extract_foods_from_query(search_query)
            if foods:
                nutrition = []
                for food in foods[:3]:
                    nut_data = self.db.lookup_en(food)
                    if nut_data:
                        nutrition.append(nut_data)
                if not nutrition:
                    nutrition = None

        if intent in ("HEALTH_ADVICE", "BOTH"):
            candidates = self.retriever.retrieve(search_query, top_k=20)
            chunks = self.reranker.rerank(search_query, candidates, top_k=self.top_k)

        result = self.generator.generate(search_query, nutrition, chunks, intent, history=history)
        result.update({"intent": intent, "entities": entities})
        return result
