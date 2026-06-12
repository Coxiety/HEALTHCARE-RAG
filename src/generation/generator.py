from __future__ import annotations

import re

import requests


class Generator:
    """Gọi Ollama local để sinh câu trả lời từ retrieved context."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.host = host.rstrip("/")

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _format_single_nutrition_data(self, data: dict) -> str:
        food_desc = data.get("food_description", "Unknown")
        if "error" in data:
            return f"Food: {food_desc}\n  Status: NOT FOUND in USDA FoodData Central database. Exact nutritional values are unavailable."

        fdc_id    = data.get("fdc_id", "")
        text = f"Food: {food_desc}\n"
        nutrients = data.get("nutrients_per_100g")
        if nutrients:
            for name, v in nutrients.items():
                text += f"  {name}: {v['amount']} {v['unit']} / 100g\n"
        elif data.get("nutrient_name"):
            text += f"  {data['nutrient_name']}: {data['amount_per_100g']} {data['unit']} / 100g\n"
        text += f"Source: USDA FoodData Central (fdc_id={fdc_id})"
        return text

    def build_prompt(
        self,
        query: str,
        nutrition_data: dict | list[dict] | None,
        health_chunks: list,
        query_type: str | None = None,
        active_context: dict | None = None,
    ) -> str:
        """
        Ghép prompt từ kết quả retrieval.

        nutrition_data: dict, list[dict] từ SqliteManager hoặc None
        health_chunks : list of RetrievedChunk (có .text và .source)
        query_type    : "NUTRITION_LOOKUP" | "HEALTH_ADVICE" | "BOTH" | None
        """
        sections = []

        need_nutrition = query_type in (None, "NUTRITION_LOOKUP", "BOTH") or (nutrition_data is not None)
        need_health    = query_type in (None, "HEALTH_ADVICE",    "BOTH")

        comparison_instruction = ""
        if need_nutrition and nutrition_data:
            if isinstance(nutrition_data, list):
                nutrition_texts = []
                for item in nutrition_data:
                    nutrition_texts.append(self._format_single_nutrition_data(item))
                nutrition_text = "\n\n".join(nutrition_texts)
                if len(nutrition_data) > 1:
                    comparison_instruction = (
                        "IMPORTANT FOR COMPARISONS: Since you are comparing multiple foods, you MUST present a side-by-side nutrition comparison using a Markdown table.\n"
                        "The table should contain columns: Nutrient, [Food 1 short name], [Food 2 short name], etc., and list values per 100g (or scaled if a specific amount was requested).\n"
                        "Make sure to list ALL available key nutrients (Protein, Energy, Total lipid (fat), Carbohydrate, etc.) in the table columns/rows for ALL foods. Do not omit any foods from the table.\n"
                        "Follow the table with a concise explanation and practical suggestions.\n\n"
                    )
            else:
                nutrition_text = self._format_single_nutrition_data(nutrition_data)

            sections.append(
                f"[Nutrition Data — USDA FoodData Central]\n"
                f"IMPORTANT: Use ONLY these exact values in your answer. Do NOT use any other numbers.\n"
                f"Note: All database nutrient amounts are given per 100g. If the user asks for a comparison or a different serving size (e.g. 3 ounces or 100g), calculate and scale these values accordingly.\n\n"
                f"{nutrition_text}"
            )

        if need_health and health_chunks:
            context_text = "\n\n".join(
                f"[{c.source}]\n{c.text[:500]}" for c in health_chunks
            )
            sections.append(f"[Reference Documents]\n{context_text}")

        # Inject conversation entity memory context if provided
        if active_context:
            context_lines = []
            for etype, items in active_context.items():
                if items:
                    context_lines.append(f"  - {etype}: {', '.join(items)}")
            if context_lines:
                context_str = "\n".join(context_lines)
                sections.append(
                    f"[Conversation Context Entities]\n"
                    f"Use these recently mentioned entities to resolve any pronouns (like 'it', 'its', 'they', 'them') or implicit references in the user's question:\n"
                    f"{context_str}"
                )

        body = "\n\n".join(sections) if sections else "No reference data available."

        return (
            "You are a professional nutrition and health assistant. Answer DIRECTLY in English, in a natural, conversational, and helpful manner.\n"
            "Do NOT mention any rules, instructions, system prompts, or formatting/structural constraints to the user. Do NOT say 'I will follow the new structure' or make similar meta-comments. Always remain in character.\n"
            "All questions are asked by a human user regarding human nutrition, diet, and health. Do NOT interpret queries as being about live animals, livestock, or veterinary care. Any food terms mentioned (like 'chicken', 'a chicken with garlic', etc.) refer to human foods/dishes, not a live animal.\n"
            "When reasoning about glycemic index (GI), blood sugar, or diabetes, base your answer on actual carbohydrate content. Foods with 0g of carbohydrate (like lean beef or chicken breast) have a Glycemic Index of essentially zero and do not raise blood sugar levels. Never claim that zero-carb meats are 'relatively high-glycemic' compared to carbohydrate-containing fruits like apples.\n"
            "If specific nutrition data is provided in [Nutrition Data — USDA FoodData Central], you MUST use those exact values and prioritize them. "
            "If no nutrition data is provided for the specific food asked about, inform the user that exact nutritional data for that item could not be found in the database. Do NOT invent, estimate, or guess any nutritional numbers.\n"
            "If the user asks a specific clinical or medical question, you must base your answer on the provided 'Reference Documents' and cite them. If no relevant documents are found for such queries, state that you do not have sufficient information in the reference library to answer.\n\n"
            f"{comparison_instruction}"
            f"{body}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
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
        nutrition_data: dict | list[dict] | None,
        health_chunks: list,
        query_type: str | None = None,
        history: list[dict] = None,
        active_context: dict | None = None,
    ) -> dict:
        """
        Sinh câu trả lời sử dụng Ollama dựa trên cấu hình model.

        Returns:
            {
                "answer": "...",
                "sources": [...],
                "used_llm": True/False
            }
        """
        # Extract single dict if nutrition_data is a list with 1 element
        single_nutrition = None
        if isinstance(nutrition_data, list) and len(nutrition_data) == 1:
            single_nutrition = nutrition_data[0]
        elif isinstance(nutrition_data, dict):
            single_nutrition = nutrition_data

        # NUTRITION_LOOKUP với data USDA: trả thẳng, không qua LLM (chỉ áp dụng khi có đúng 1 thực phẩm đầy đủ thông tin)
        if query_type == "NUTRITION_LOOKUP" and single_nutrition and single_nutrition.get("nutrients_per_100g"):
            answer = self._format_nutrition_answer(single_nutrition)
            sources = [f"USDA FoodData Central (fdc_id={single_nutrition['fdc_id']})"]
            return {"answer": answer, "sources": sources, "used_llm": False}

        prompt = self.build_prompt(query, nutrition_data, health_chunks, query_type, active_context)
        print(f"\n[DEBUG PROMPT] Final Injected Prompt Sent to LLM:\n{prompt}\n")

        answer = self._call_ollama(prompt, history)

        sources = [c.source for c in health_chunks]
        if nutrition_data:
            if isinstance(nutrition_data, list):
                for item in nutrition_data:
                    if "fdc_id" in item:
                        sources.append(f"USDA FoodData Central (fdc_id={item['fdc_id']})")
            else:
                if "fdc_id" in nutrition_data:
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


    def _call_ollama(self, prompt: str, history: list[dict] = None) -> str | None:
        history = history or []
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional, helpful, and friendly nutrition and health advisor. "
                    "Answer all questions directly and concisely in English. "
                    "Never mention any system prompts, rules, instructions, or lack of data/documents to the user. "
                    "Always remain in character.\n\n"
                    "CRITICAL GUIDELINES:\n"
                    "1. Human Context: Always assume the user is a human inquiring about human nutrition, health, and diet. Do not interpret queries as being about live animals, livestock, or veterinary care (e.g., 'chicken with garlic' refers to human food/dishes, not treating a live bird).\n"
                    "2. Glycemic Index & Diabetes: Reason scientifically using actual nutrient values. Foods with 0g of carbohydrate (like lean beef or chicken breast) have a Glycemic Index of essentially zero and do not raise blood sugar levels. Do NOT claim that 0g carb meats are 'high-glycemic' or will spike blood sugar compared to carbohydrate-containing fruits like apples.\n"
                    "3. Differentiate Risks: Distinguish between long-term epidemiological correlation (e.g. processed meat risk in reference documents) and immediate physiological/glycemic impact of a single food item.\n"
                    "4. Multi-hop Reasoning (Chain-of-Thought): When resolving complex queries, perform logical reasoning step-by-step using both USDA Nutrition Data (for food composition/nutrients) and Reference Documents (for clinical/medical associations of those nutrients). Follow this logical chain internally:\n"
                    "   - Step 1: Identify the food's nutrient profile from the USDA data (e.g. Avocado contains monounsaturated fatty acids and dietary fiber).\n"
                    "   - Step 2: Correlate those specific nutrients with the clinical outcomes in the Reference Documents (e.g. monounsaturated fats improve insulin sensitivity and lower LDL cholesterol).\n"
                    "   - Step 3: Combine these steps to form a cohesive, scientifically grounded answer. Cite reference documents where appropriate.\n\n"
                    "Example of Multi-hop Reasoning:\n"
                    "   Question: Is salmon good for cardiovascular health?\n"
                    "   USDA Data: Salmon has high levels of Omega-3 fatty acids (EPA/DHA).\n"
                    "   References: Omega-3 fatty acids from fish reduce cardiovascular risk by lowering blood pressure and triglycerides.\n"
                    "   Reasoning: 1. Salmon contains high Omega-3 fatty acids. 2. References link Omega-3 to reduced cardiovascular risk. 3. Therefore, eating salmon supports cardiovascular health.\n"
                    "   Answer: Yes, salmon is excellent for cardiovascular health. It is highly rich in Omega-3 fatty acids (EPA and DHA), which clinical references show reduce cardiovascular risk by lowering blood pressure and triglycerides.\n\n"
                    "5. Use the provided context/data if available to form your answer; otherwise, use your general knowledge to answer with helpful and accurate information."
                ),
            }
        ]

        # Limit to the last 6 messages (3 turns) to prevent token bloat and optimize local inference speed
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": prompt})

        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model":   self.model,
                    "messages": messages,
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
        """Safety net: strip <think> tags nếu có."""
        if not text:
            return text
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip() or text

    @staticmethod
    def _fallback_answer(
        query: str,
        nutrition_data: dict | list[dict] | None,
        health_chunks: list,
        query_type: str | None = None,
    ) -> str:
        parts = [f"Question: {query}\n"]

        need_nutrition = query_type in (None, "NUTRITION_LOOKUP", "BOTH")
        need_health    = query_type in (None, "HEALTH_ADVICE",    "BOTH")

        if need_nutrition:
            if nutrition_data:
                if isinstance(nutrition_data, list):
                    for item in nutrition_data:
                        parts.append(f"USDA data for '{item.get('food_description', '')}' (fdc_id={item.get('fdc_id', '')}):")
                        nutrients = item.get("nutrients_per_100g")
                        if nutrients:
                            for name, v in nutrients.items():
                                parts.append(f"  - {name}: {v['amount']} {v['unit']} / 100g")
                        elif item.get("nutrient_name"):
                            parts.append(f"  - {item.get('nutrient_name')}: {item.get('amount_per_100g')} {item.get('unit')} / 100g")
                else:
                    parts.append(f"USDA data for '{nutrition_data.get('food_description', '')}' (fdc_id={nutrition_data.get('fdc_id', '')}):")
                    nutrients = nutrition_data.get("nutrients_per_100g")
                    if nutrients:
                        for name, v in nutrients.items():
                            parts.append(f"  - {name}: {v['amount']} {v['unit']} / 100g")
                    elif nutrition_data.get("nutrient_name"):
                        parts.append(f"  - {nutrition_data.get('nutrient_name')}: {nutrition_data.get('amount_per_100g')} {nutrition_data.get('unit')} / 100g")
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
