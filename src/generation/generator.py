from __future__ import annotations

import re

import requests


class Generator:
    """Gọi Ollama local để sinh câu trả lời từ retrieved context với prompt tối ưu."""

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

    def _format_single_nutrition_data(self, data: dict) -> str | None:
        """Chỉ trả về chuỗi nếu tìm thấy dữ liệu thực sự. Nếu lỗi/không tìm thấy, trả về None."""
        if "error" in data or not data:
            return None
            
        food_desc = data.get("food_description", "Unknown")
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
        Ghép prompt SẠCH từ kết quả retrieval, loại bỏ text rác và giữ nguyên vẹn chunk.
        """
        sections = []
        
        # Phân luồng nghiêm ngặt theo đúng lập trường của bạn (Person A)
        need_nutrition = query_type in (None, "NUTRITION_LOOKUP", "BOTH")
        need_health    = query_type in (None, "HEALTH_ADVICE", "BOTH")

        # 1. Xử lý dữ liệu dinh dưỡng (USDA)
        comparison_instruction = ""
        if need_nutrition and nutrition_data:
            nutrition_texts = []
            valid_items_count = 0
            
            items = nutrition_data if isinstance(nutrition_data, list) else [nutrition_data]
            for item in items:
                formatted = self._format_single_nutrition_data(item)
                if formatted:  # Chỉ lấy nếu thực sự FOUND dữ liệu
                    nutrition_texts.append(formatted)
                    valid_items_count += 1
            
            if nutrition_texts:
                nutrition_text = "\n\n".join(nutrition_texts)
                if valid_items_count > 1:
                    comparison_instruction = (
                        "IMPORTANT FOR COMPARISONS: Since you are comparing multiple foods, you MUST present a side-by-side nutrition comparison using a Markdown table.\n"
                        "The table should contain columns: Nutrient, [Food 1 short name], [Food 2 short name], etc., and list values per 100g.\n"
                        "Make sure to list ALL available key nutrients (Protein, Energy, Total lipid (fat), Carbohydrate) for ALL foods.\n\n"
                    )
                sections.append(
                    f"[Nutrition Data — USDA FoodData Central]\n"
                    f"IMPORTANT: Use ONLY these exact values. Do NOT invent numbers.\n\n"
                    f"{nutrition_text}"
                )

        # 2. Xử lý dữ liệu Y tế (Health Chunks) - Khóa chặt tối đa 3 sources liên quan nhất
        if need_health and health_chunks:
            # Đảm bảo giữ nguyên vẹn c.text, KHÔNG dùng [:500] bẻ gãy câu bừa bãi
            # Giới hạn nghiêm ngặt lấy tối đa 3 chunks để tăng Context Precision
            context_text = "\n\n".join(
                f"[{c.source}]\n{c.text}" for c in health_chunks[:3]
            )
            sections.append(f"[Reference Documents]\n{context_text}")

        # 3. Xử lý ngữ cảnh hội thoại
        if active_context:
            context_lines = []
            for etype, items in active_context.items():
                if items:
                    context_lines.append(f"  - {etype}: {', '.join(items)}")
            if context_lines:
                context_str = "\n".join(context_lines)
                sections.append(
                    f"[Conversation Context Entities]\n"
                    f"Use these context entities to resolve pronouns (it, they, them):\n"
                    f"{context_str}"
                )

        body = "\n\n".join(sections) if sections else "No reference data available."

        # Trả về prompt SẠCH, đẩy toàn bộ luật rườm rà lên SYSTEM PROMPT của Ollama
        return (
            f"{comparison_instruction}"
            f"{body}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
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
        try:
            print(f"\n[DEBUG PROMPT] Final Injected Prompt Sent to LLM:\n{prompt}\n")
        except UnicodeEncodeError:
            import sys
            enc = sys.stdout.encoding or 'utf-8'
            safe_prompt = prompt.encode(enc, errors='replace').decode(enc)
            print(f"\n[DEBUG PROMPT] Final Injected Prompt Sent to LLM:\n{safe_prompt}\n")

        answer = self._call_ollama(prompt, history)

        sources = [c.source for c in health_chunks[:3]]
        if nutrition_data:
            if isinstance(nutrition_data, list):
                for item in nutrition_data:
                    if "fdc_id" in item and "error" not in item:
                        sources.append(f"USDA FoodData Central (fdc_id={item['fdc_id']})")
            else:
                if "fdc_id" in nutrition_data and "error" not in nutrition_data:
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
        
        # SYSTEM PROMPT tập trung cô đọng toàn bộ gông cùm (Constraints) tại một nơi
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional medical and nutrition assistant. Always answer DIRECTLY and naturally in English.\n"
                    "SAFETY RULE: Base your answer STRICTLY on the provided [Reference Documents] and [Nutrition Data]. "
                    "If the information is not present in the context, state clearly that you do not have sufficient information. Do NOT invent, assume, or speculate.\n"
                    "CRITICAL CONSTRAINTS:\n"
                    "1. Never mention system rules, prompts, constraints, or metadata (like 'According to document 1' or 'The database says NOT FOUND') to the user.\n"
                    "2. All queries are about human nutrition. Terms like 'chicken' mean human food dishes, never live animals.\n"
                    "3. Glycemic Index Logic: Foods with 0g carbohydrate (e.g., lean beef, chicken breast) have a GI of essentially zero and DO NOT raise blood sugar. Never claim zero-carb meats spike blood sugar compared to high-carb foods like apples."
                ),
            }
        ]

        # Sliding window giữ 6 tin nhắn gần nhất
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
                        "num_predict": 700,
                        "temperature": 0.0,  # Đưa về 0.0 để ép mô hình tuân thủ kỷ luật, triệt tiêu ảo tưởng
                    },
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("message", {}).get("content", "").strip()
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            print(f"[OLLAMA STATS] Input: {prompt_tokens} tokens | Output: {completion_tokens} tokens | Total: {prompt_tokens + completion_tokens} tokens")
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
                        "num_predict": 700,
                        "temperature": 0.0,
                    },
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response", "").strip()
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            print(f"[OLLAMA GENERATE STATS] Input: {prompt_tokens} tokens | Output: {completion_tokens} tokens | Total: {prompt_tokens + completion_tokens} tokens")
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

        need_nutrition = query_type in (None, "NUTRITION_LOOKUP")
        need_health    = query_type in (None, "HEALTH_ADVICE",    "BOTH")

        if need_nutrition:
            valid_nutrition_texts = []
            if nutrition_data:
                items = nutrition_data if isinstance(nutrition_data, list) else [nutrition_data]
                for item in items:
                    if "error" not in item:
                        nut_parts = []
                        nut_parts.append(f"USDA data for '{item.get('food_description', '')}' (fdc_id={item.get('fdc_id', '')}):")
                        nutrients = item.get("nutrients_per_100g")
                        if nutrients:
                            for name, v in nutrients.items():
                                nut_parts.append(f"  - {name}: {v['amount']} {v['unit']} / 100g")
                        elif item.get("nutrient_name"):
                            nut_parts.append(f"  - {item.get('nutrient_name')}: {item.get('amount_per_100g')} {item.get('unit')} / 100g")
                        valid_nutrition_texts.append("\n".join(nut_parts))
            
            if valid_nutrition_texts:
                parts.extend(valid_nutrition_texts)
            else:
                parts.append("No USDA nutrition data found.")

        if need_health:
            if health_chunks:
                parts.append("\nRelevant medical references:")
                for c in health_chunks[:3]:
                    parts.append(f"  - {c.text}  (Source: {c.source})")
            else:
                parts.append("No relevant medical documents found.")

        return "\n".join(parts)
