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
    
    # Define keywords/stopwords to discard (question words, helper verbs, articles, general RAG words, nutrients, etc.)
    stopwords = {
        # Question / helper / search words
        "what", "how", "why", "when", "where", "who", "which", "whom", "whose",
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "has", "have", "had", "can", "could", "will", "would", "should", "shall", "may", "might", "must",
        "give", "me", "show", "tell", "find", "lookup", "search", "get", "check", "estimate", "estimated", "calculate", "calculated", "provide", "provided", "list", "listed",
        # Articles & prepositions & conjunctions
        "a", "an", "the", "in", "of", "for", "with", "between", "and", "or", "vs", "versus", "compared", "to", "at", "by", "from", "on", "about", "into", "through", "during", "under", "over", "like", "as",
        # Pronouns
        "i", "you", "he", "she", "it", "we", "they", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
        # Nutrients
        "protein", "proteins", "calorie", "calories", "fat", "fats", "lipid", "lipids", "carbs", "carb", "carbohydrate", 
        "carbohydrates", "fiber", "fibers", "energy", "cholesterol", "sugar", "sugars", "sodium", "vitamin", "vitamins",
        "kcal", "g", "mg", "microgram", "micrograms", "gram", "grams", "ounce", "ounces", "serving", "servings", "piece", "pieces", "portion", "portions",
        # Quantity / amount words
        "many", "much", "some", "any", "few", "little", "more", "most", "all", "both", "each", "every", "other", "another",
        # Rephrasing / metadata / filler noise words
        "approximate", "approximately", "amount", "amounts", "typical", "typically", "average", "source", "sources", "content", "contents", "approx", 
        "estimated", "estimate", "type", "types", "kind", "kinds", "sort", "sorts", "one", "two", "three", "value", "values", "nutrition", "nutrient", "nutrients", "nutritional", "info", "information",
        "fact", "facts", "data", "found", "contain", "contains", "contained", "containing", "recommend", "recommends", "recommended", "suggest", "suggests", "suggested",
        "good", "bad", "healthy", "unhealthy", "best", "worst", "better", "health", "body", "human", "people", "person",
        "specifically", "particular", "particularly", "exact", "exactly", "general", "generally", "usually", "common", "commonly", "normal", "normally",
        "high", "low", "rich", "poor", "eat", "eating", "consume", "consuming", "diet", "meal", "food", "foods", "drink", "drinks", "make", "makes", "made",
        "cooked", "raw", "braised", "roasted", "fried", "boiled", "baked", "steamed", "grilled", "fresh", "frozen", "dried", "canned", "skinless", "boneless", "meat", "only"
    }
    
    t = re.sub(r'[^\w\s]', ' ', t)
    words = [w.strip() for w in t.split() if w.strip()]
    cleaned_words = [w for w in words if w not in stopwords]
    return " ".join(cleaned_words)


def _extract_foods_from_query(query: str) -> list[str]:
    """Extract food names from query when NER returns no FOOD entities."""
    parts = re.split(r'\b(?:vs|compared\s+to|and|or)\b|[,/]', query, flags=re.IGNORECASE)
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
