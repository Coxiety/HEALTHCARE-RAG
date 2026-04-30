from __future__ import annotations

import requests


class Generator:
    """Gọi Ollama local để sinh câu trả lời từ retrieved context."""

    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        query: str,
        nutrition_data: dict | None,
        health_chunks: list,
        query_type: str | None = None,
    ) -> str:
        """
        Ghép prompt từ kết quả retrieval.

        nutrition_data: dict từ SqliteManager.lookup() hoặc None
        health_chunks : list of RetrievedChunk (có .text và .source)
        query_type    : "NUTRITION_LOOKUP" | "HEALTH_ADVICE" | "BOTH" | None
        """
        sections = []

        need_nutrition = query_type in (None, "NUTRITION_LOOKUP", "BOTH")
        need_health    = query_type in (None, "HEALTH_ADVICE",    "BOTH")

        if need_nutrition and nutrition_data:
            food_desc = nutrition_data.get("food_description", "Unknown")
            fdc_id    = nutrition_data.get("fdc_id", "")
            nutrition_text = f"Food: {food_desc}\n"
            nutrients = nutrition_data.get("nutrients_per_100g")
            if nutrients:
                for name, v in nutrients.items():
                    nutrition_text += f"  {name}: {v['amount']} {v['unit']} / 100g\n"
            elif nutrition_data.get("nutrient_name"):
                nutrition_text += f"  {nutrition_data['nutrient_name']}: {nutrition_data['amount_per_100g']} {nutrition_data['unit']} / 100g\n"
            nutrition_text += f"Source: USDA FoodData Central (fdc_id={fdc_id})"
            sections.append(
                f"[Nutrition Data — USDA FoodData Central]\n"
                f"IMPORTANT: Use ONLY these exact values in your answer. Do NOT use any other numbers.\n"
                f"{nutrition_text}"
            )

        if need_health and health_chunks:
            context_text = "\n\n".join(
                f"[{c.source}]\n{c.text[:500]}" for c in health_chunks
            )
            sections.append(f"[Reference Documents]\n{context_text}")

        body = "\n\n".join(sections) if sections else "No reference data available."

        return (
            "You are a professional nutrition and health assistant. Answer DIRECTLY in English, concisely and accurately.\n"
            "Structure: (1) direct answer, (2) explanation or mechanism, (3) practical food suggestions if relevant.\n"
            "CRITICAL: Use ONLY the exact numbers from the provided data. Never invent or estimate nutritional values.\n\n"
            f"{body}\n\n"
            f"Question: {query}\n\n"
            "Answer (cite sources at the end):"
        )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _format_nutrition_answer(self, nutrition_data: dict) -> str:
        food = nutrition_data.get("food_description", "Unknown")
        nutrients = nutrition_data.get("nutrients_per_100g", {})
        fdc_id = nutrition_data.get("fdc_id", "")
        lines = [f"**{food}** (per 100g) — USDA FoodData Central"]
        for name, v in nutrients.items():
            lines.append(f"- {name}: {v['amount']} {v['unit']}")
        lines.append(f"\nSource: USDA FoodData Central (fdc_id={fdc_id})")
        return "\n".join(lines)

    def generate(
        self,
        query: str,
        nutrition_data: dict | None,
        health_chunks: list,
        query_type: str | None = None,
    ) -> dict:
        """
        Gọi Ollama và trả về kết quả.

        Returns:
            {
                "answer": "...",
                "sources": [...],
                "used_llm": True/False
            }
        """
        # NUTRITION_LOOKUP với data USDA: trả thẳng, không qua LLM
        if query_type == "NUTRITION_LOOKUP" and nutrition_data and nutrition_data.get("nutrients_per_100g"):
            answer = self._format_nutrition_answer(nutrition_data)
            sources = [f"USDA FoodData Central (fdc_id={nutrition_data['fdc_id']})"]
            return {"answer": answer, "sources": sources, "used_llm": False}

        prompt = self.build_prompt(query, nutrition_data, health_chunks, query_type)
        answer = self._call_ollama(prompt)

        sources = [c.source for c in health_chunks]
        if nutrition_data:
            sources.append(f"USDA FoodData Central (fdc_id={nutrition_data['fdc_id']})")

        if answer is None:
            answer = self._fallback_answer(query, nutrition_data, health_chunks, query_type)
            used_llm = False
        else:
            used_llm = True

        return {
            "answer": answer,
            "sources": sorted(set(sources)),
            "used_llm": used_llm,
        }

    def _call_ollama(self, prompt: str) -> str | None:
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model":   self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a professional nutrition and health advisor. "
                                "Answer questions based on the provided scientific documents. "
                                "Always respond in English, directly and thoroughly. "
                                "Never refuse to answer — always use the provided context to give helpful information."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream":  False,
                    "options": {
                        "num_ctx":     4096,
                        "num_predict": 2000,
                        "temperature": 0.3,
                    },
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("message", {}).get("content", "").strip()
            return self._strip_thinking(answer) or None
        except Exception:
            return None

    def _call_ollama_generate(self, prompt: str) -> str | None:
        """Fallback dùng /api/generate nếu /api/chat không khả dụng."""
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model":   self.model,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {
                        "num_ctx":     4096,
                        "num_predict": 2000,
                        "temperature": 0.3,
                    },
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response", "").strip()
            if not answer:
                answer = data.get("thinking", "").strip()
            return self._strip_thinking(answer) or None
        except Exception:
            return None

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Safety net: strip <think> tags nếu có. qwen2.5 không có thinking mode."""
        import re
        if not text:
            return text
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip() or text

    @staticmethod
    def _fallback_answer(
        query: str,
        nutrition_data: dict | None,
        health_chunks: list,
        query_type: str | None = None,
    ) -> str:
        parts = [f"Question: {query}\n"]

        need_nutrition = query_type in (None, "NUTRITION_LOOKUP", "BOTH")
        need_health    = query_type in (None, "HEALTH_ADVICE",    "BOTH")

        if need_nutrition:
            if nutrition_data:
                parts.append(
                    f"USDA data: {nutrition_data.get('nutrient_name', '')} of "
                    f"'{nutrition_data.get('food_description', '')}' is "
                    f"{nutrition_data.get('amount_per_100g', '')} {nutrition_data.get('unit', '')} / 100g."
                )
            else:
                parts.append("No USDA nutrition data found.")

        if need_health:
            if health_chunks:
                parts.append("\nRelevant medical references:")
                for c in health_chunks:
                    parts.append(f"  - {c.text[:300]}  (Source: {c.source})")
            else:
                parts.append("No relevant medical documents found.")

        return "\n".join(parts)


if __name__ == "__main__":
    g = Generator()
    print("Testing Ollama connection...")
    result = g._call_ollama("Xin chao, ban co hoat dong khong?")
    if result:
        print("OK:", result[:100])
    else:
        print("Ollama offline. Fallback active.")
