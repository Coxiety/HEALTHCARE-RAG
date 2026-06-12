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

_EXCLUDED_FOODS = {
    "a", "an", "the", "it", "its", "they", "them", "this", "that", "which", "what", "who", "whom", "whose", "whosever",
    "one", "ones", "someone", "something", "anything", "nothing", "everything", "some", "any", "all", "both", "each", "every",
    "other", "another", "either", "neither", "none"
}

def _clean_food_entity(food: str) -> str | None:
    """Clean food entities by stripping determiners, lemmatizing to singular form, and excluding non-food terms."""
    food = food.strip().lower()
    for prefix in ["a ", "an ", "the "]:
        if food.startswith(prefix):
            food = food[len(prefix):].strip()

    if not food or food in _EXCLUDED_FOODS or food in _EXCLUDED_NOUNS:
        return None

    doc = nlp(food)
    lemmatized = " ".join([token.lemma_ for token in doc]).strip().lower()

    if not lemmatized or lemmatized in _EXCLUDED_FOODS or lemmatized in _EXCLUDED_NOUNS:
        return None

    return lemmatized

def _resolve_generic_foods(curr_foods: list[str], historical_foods: list[str]) -> list[str]:
    """Resolve generic food names (e.g. 'chicken') to more specific historical names (e.g. 'chicken breast')."""
    resolved = []
    for food in curr_foods:
        food_lower = food.lower()
        food_words = set(re.findall(r'\b\w+\b', food_lower))
        if not food_words:
            resolved.append(food)
            continue

        matched = False
        for hist in historical_foods:
            hist_lower = hist.lower()
            hist_words = set(re.findall(r'\b\w+\b', hist_lower))
            if food_words.issubset(hist_words) and food_lower != hist_lower:
                resolved.append(hist)
                matched = True
                break
        if not matched:
            resolved.append(food)
    return resolved


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

        self.use_llm_rewriter = cfg.get("use_llm_rewriter", False)
        self.rewriter_model = cfg.get("rewriter_model", "llama3.1:8b")

        self.lazy_rerank = cfg.get("lazy_rerank", False)
        self.lazy_rerank_threshold = cfg.get("lazy_rerank_threshold", 0.05)

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
        use_llm_rewriter = getattr(self, "use_llm_rewriter", False)

        # 1. Run NER on current query
        curr_entities = self.ner.predict(query)

        # 2. Extract and aggregate historical entities from previous turns
        historical_entities = {
            "FOOD": [],
            "DISEASE": [],
            "NUTRIENT": [],
            "SYMPTOM": []
        }

        if history:
            # Look back through up to 5 turns (max 10 user/assistant messages)
            for msg in reversed(history[-10:]):
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", "")
                if content:
                    ent_pred = self.ner.predict(content)
                    for etype in ["FOOD", "DISEASE", "NUTRIENT", "SYMPTOM"]:
                        for ent in ent_pred.get(etype, []):
                            if ent not in historical_entities[etype]:
                                historical_entities[etype].append(ent)

        # Keep at most 5 unique items of each type
        for etype in ["FOOD", "DISEASE", "NUTRIENT", "SYMPTOM"]:
            historical_entities[etype] = historical_entities[etype][:5]

        # Debug print current and accumulated entities
        print(f"\n[DEBUG NER] Current Entities: {curr_entities}")
        print(f"[DEBUG NER] Accumulated Historical Entities (last 5 turns): {historical_entities}\n")

        # 3. Formulate retrieval query and entities for DB lookup
        if use_llm_rewriter:
            search_query = self._condense_query(query, history)
            print(f"[DEBUG] Original Query: '{query}' -> Condensed/Rewritten Query: '{search_query}'")
            entities_for_lookup = self.ner.predict(search_query)
            retrieval_query = search_query
        else:
            search_query = query
            entities_for_lookup = curr_entities.copy()

            # Backfill missing entity types using historical entities
            for etype in ["FOOD", "DISEASE", "NUTRIENT", "SYMPTOM"]:
                if not entities_for_lookup.get(etype) and historical_entities[etype]:
                    entities_for_lookup[etype] = historical_entities[etype]

            # Enrich retrieval query text for Vector/BM25 retrievers
            retrieval_query = query
            added_words = []
            for etype in ["FOOD", "DISEASE", "SYMPTOM"]:
                if not curr_entities.get(etype) and historical_entities[etype]:
                    most_recent = historical_entities[etype][0]
                    if most_recent.lower() not in retrieval_query.lower():
                        added_words.append(most_recent)
            if added_words:
                retrieval_query += " " + " ".join(added_words)

        # Resolve generic food terms using historical context
        if entities_for_lookup.get("FOOD") and historical_entities.get("FOOD"):
            entities_for_lookup["FOOD"] = _resolve_generic_foods(
                entities_for_lookup["FOOD"], historical_entities["FOOD"]
            )

        intent = self.clf.classify(search_query)
        nutrition = None
        chunks = []

        # 4. Database Lookup (always run if food entities are present to support LLM reasoning)
        foods = entities_for_lookup.get("FOOD", [])
        cleaned_foods = []
        for f in foods:
            cf = _clean_food_entity(f)
            if cf and cf not in cleaned_foods:
                cleaned_foods.append(cf)

        # Fallback to spaCy noun chunks if no clean food entities are found
        if not cleaned_foods:
            fallback_foods = _extract_foods_from_query(search_query)
            for f in fallback_foods:
                cf = _clean_food_entity(f)
                if cf and cf not in cleaned_foods:
                    cleaned_foods.append(cf)

        if cleaned_foods:
            nutrition = []
            # Lookup database using top 3 candidate foods (current or historical resolved)
            for food in cleaned_foods[:3]:
                nut_data = self.db.lookup_en(food)
                if nut_data:
                    nutrition.append(nut_data)
                else:
                    # Append placeholder to explicitly inform LLM that the food is missing from DB
                    nutrition.append({
                        "food_description": food,
                        "error": "Not found in USDA database"
                    })
            if not nutrition:
                nutrition = None

        # 5. Reference Document Retrieval
        if intent in ("HEALTH_ADVICE", "BOTH"):
            ret_q = search_query if use_llm_rewriter else retrieval_query
            candidates = self.retriever.retrieve(ret_q, top_k=20)

            # Check if we should use lazy reranking based on retrieval confidence gap
            if self.lazy_rerank and len(candidates) > 1 and (candidates[0].score - candidates[1].score) >= self.lazy_rerank_threshold:
                chunks = candidates[:self.top_k]
                print(f"[DEBUG] Reranker Bypassed (Lazy Confident: Gap {candidates[0].score - candidates[1].score:.4f} >= {self.lazy_rerank_threshold})")
            else:
                chunks = self.reranker.rerank(ret_q, candidates, top_k=self.top_k)

        # 6. Response Generation (pass active_context if bypassing LLM rewriter)
        active_context_metadata = historical_entities if not use_llm_rewriter else None

        result = self.generator.generate(
            query=query,
            nutrition_data=nutrition,
            health_chunks=chunks,
            query_type=intent,
            history=history,
            active_context=active_context_metadata
        )
        result.update({"intent": intent, "entities": curr_entities})
        return result
