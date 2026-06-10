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

# NER misses food names (trained on BC5CDR Chemical/Disease only) — regex/keyword fallback
def clean_food_name(text: str) -> str:
    # Remove prep/number prefix like "100g of", "3 oz of", "3-ounce serving of"
    t = text.lower()
    t = re.sub(r'\b\d+(?:\s*(?:g|gram|grams|oz|ounce|ounces|lbs|kg|serving|servings|piece|pieces))?\b', '', t)
    # Remove question words and nutrition keywords
    t = re.sub(r'\b(?:of|for|in|about|what|how|compare|with|many|much|protein|proteins|carbs|carbohydrate|carbohydrates|fat|fats|lipid|lipids|calories|energy|sugar|sugars|vitamin|vitamins|mineral|minerals|nutrition|nutritional|value|is|are|does|do|has|have|the|a|an)\b', '', t)
    t = re.sub(r'[^\w\s]', ' ', t)
    words = [w.strip() for w in t.split() if w.strip()]
    return " ".join(words)


def _extract_foods_from_query(query: str) -> list[str]:
    """Extract food names from query when NER returns no FOOD entities."""
    parts = re.split(r'\b(?:vs|compared\s+to|and)\b|[,/]', query, flags=re.IGNORECASE)
    foods = []
    for part in parts:
        cleaned = clean_food_name(part)
        if cleaned:
            foods.append(cleaned)
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

        self.rewriter_backend = cfg.get("rewriter_backend", "ollama")
        self.rewriter_model = cfg.get("rewriter_model", "gemini-2.5-flash")

        # Initialize Gemini Client if needed
        self.genai_client = None
        if self.rewriter_backend == "gemini" or cfg.get("llm_backend") == "gemini":
            try:
                import os
                from google import genai
                api_key = os.environ.get("GEMINI_API_KEY") or cfg.get("gemini_api_key")
                if api_key:
                    self.genai_client = genai.Client(api_key=api_key)
                else:
                    self.genai_client = genai.Client()
                print(f"[ENPipeline] Gemini client initialized successfully.")
            except Exception as e:
                print(f"[ENPipeline] Warning: Could not initialize Gemini client ({e}). Falling back to Ollama.")
                self.rewriter_backend = "ollama"

        self.generator = Generator(
            model=cfg["llm_model"],
            backend=cfg.get("llm_backend", "ollama"),
            genai_client=self.genai_client
        )
        self.top_k = cfg.get("top_k", 5)

    def _condense_query(self, query: str, history: list[dict]) -> str:
        """Sử dụng LLM (Gemini hoặc Ollama) để viết lại câu hỏi dựa trên lịch sử hội thoại."""
        if not history:
            return query

        if self.rewriter_backend == "gemini" and self.genai_client:
            try:
                from google.genai import types

                rewriter_system_prompt = (
                    "You are a back-end query rewriter layer for a nutrition chatbot.\n"
                    "Your only job is to look at the recent conversation history and the user's latest message.\n"
                    "If the user's message contains pronouns (it, its, they, that, those) or is a contextual follow-up "
                    "(e.g., 'How about X?', 'Its nutrition values', 'Is it safe?'), rewrite it into a single, fully "
                    "independent, explicit question (in English).\n"
                    "Do NOT answer the question. Do NOT include introductory text. Only output the rewritten question string."
                )

                formatted_history = ""
                for turn in history[-4:]:  # Lấy tối đa 4 lượt hội thoại gần nhất
                    role = "User" if turn["role"] == "user" else "Bot"
                    formatted_history += f"{role}: {turn['content']}\n"

                prompt = f"""CONVERSATION HISTORY:
{formatted_history}

LATEST USER MESSAGE:
{query}

REWRITTEN QUESTION:"""

                response = self.genai_client.models.generate_content(
                    model=self.rewriter_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=rewriter_system_prompt,
                        temperature=0.0
                    )
                )
                condensed = response.text.strip() if response.text else ""
                if condensed:
                    if (condensed.startswith('"') and condensed.endswith('"')) or (condensed.startswith("'") and condensed.endswith("'")):
                        condensed = condensed[1:-1].strip()
                    return condensed
            except Exception as e:
                print(f"[ENPipeline] Gemini rewriter failed: {e}. Falling back to Ollama.")

        # Fallback to Ollama
        history_text = ""
        for msg in history[-5:]:  # lấy tối đa 5 tin nhắn gần nhất
            history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

        prompt = (
            "Given the following conversation history and a follow-up question, "
            "rephrase the follow-up question to be a standalone question (in English) "
            "that captures all necessary context. Do NOT answer the question, just output the rephrased question.\n\n"
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
                elif len(nutrition) == 1:
                    # Nếu chỉ có 1 món ăn, bung list ra thành dict để khớp với điều kiện fast-path trong generator.py
                    nutrition = nutrition[0]

        if intent in ("HEALTH_ADVICE", "BOTH"):
            candidates = self.retriever.retrieve(search_query, top_k=20)
            chunks = self.reranker.rerank(search_query, candidates, top_k=self.top_k)

        result = self.generator.generate(search_query, nutrition, chunks, intent, history=history)
        result.update({"intent": intent, "entities": entities})
        return result
