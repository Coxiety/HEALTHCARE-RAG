"""One-shot generator: produce synthetic_intent.csv + intent_data.csv from templates.

Internal script — not part of production pipeline.
Run: python src/data_pipeline/_gen_intent_direct.py
"""
from __future__ import annotations

import itertools
import random
from pathlib import Path

import pandas as pd

random.seed(42)

# ── Word banks ─────────────────────────────────────────────────────────────────

FOODS = [
    "chicken breast", "salmon", "tuna", "beef", "pork loin", "shrimp", "sardines",
    "tilapia", "cod", "turkey breast", "lamb", "duck", "crab", "lobster",
    "broccoli", "spinach", "kale", "carrots", "tomatoes", "sweet potatoes",
    "potatoes", "onions", "garlic", "bell peppers", "zucchini", "cauliflower",
    "cucumber", "celery", "asparagus", "mushrooms", "beets", "cabbage",
    "apples", "bananas", "oranges", "grapes", "strawberries", "blueberries",
    "avocado", "mango", "pineapple", "watermelon", "kiwi", "peaches",
    "brown rice", "white rice", "oats", "quinoa", "whole wheat bread",
    "white bread", "pasta", "barley", "lentils", "black beans", "chickpeas",
    "soybeans", "kidney beans", "pinto beans", "green peas",
    "almonds", "walnuts", "cashews", "peanuts", "sunflower seeds", "chia seeds",
    "flaxseeds", "pumpkin seeds",
    "whole milk", "skim milk", "Greek yogurt", "plain yogurt", "cheddar cheese",
    "mozzarella", "cottage cheese", "butter", "eggs",
    "olive oil", "coconut oil", "peanut butter", "almond butter", "dark chocolate",
    "tofu", "tempeh", "edamame",
]

NUTRIENTS = [
    "calories", "protein", "carbohydrates", "total fat", "saturated fat",
    "fiber", "sodium", "potassium", "calcium", "iron", "magnesium", "zinc",
    "vitamin C", "vitamin D", "vitamin A", "vitamin B12", "folate",
    "omega-3 fatty acids", "omega-6 fatty acids", "cholesterol", "sugar",
    "phosphorus", "selenium", "thiamine", "riboflavin", "niacin",
]

AMOUNTS = [
    "100g", "100 grams", "one serving", "half a cup", "one cup", "one ounce",
    "a 3-ounce portion", "200 calories worth", "a medium portion", "one tablespoon",
    "a small bowl", "one piece", "a 150g portion", "two tablespoons", "a handful of",
]

CONDITIONS = [
    "diabetes", "type 2 diabetes", "type 1 diabetes", "gestational diabetes",
    "hypertension", "high blood pressure", "low blood pressure",
    "high cholesterol", "elevated LDL cholesterol",
    "heart disease", "coronary artery disease", "cardiovascular disease",
    "gout", "kidney disease", "chronic kidney disease",
    "celiac disease", "gluten intolerance", "lactose intolerance",
    "anemia", "iron-deficiency anemia", "osteoporosis",
    "obesity", "metabolic syndrome", "fatty liver disease",
    "hypothyroidism", "hyperthyroidism",
    "irritable bowel syndrome", "inflammatory bowel disease",
    "GERD", "acid reflux", "gastritis",
    "arthritis", "rheumatoid arthritis", "gout attacks",
    "depression", "anxiety", "insomnia",
    "PCOS", "polycystic ovary syndrome",
]

# ── Templates ──────────────────────────────────────────────────────────────────

NL_TEMPLATES = [
    "How many {nutrient} are in {amount} of {food}?",
    "What is the {nutrient} content of {food}?",
    "How much {nutrient} does {food} contain per serving?",
    "What is the exact {nutrient} in {amount} {food}?",
    "How many grams of {nutrient} are in {food}?",
    "Can you tell me the {nutrient} in {amount} of {food}?",
    "What percentage of daily {nutrient} is in {food}?",
    "How much {nutrient} is there in {amount} of {food}?",
    "What are the {nutrient} levels in {food}?",
    "Give me the {nutrient} value for {amount} of {food}.",
    "I need the {nutrient} count for {food}.",
    "What is the nutritional value of {food} in terms of {nutrient}?",
    "How many {nutrient} calories does {food} have?",
    "What is the {nutrient} breakdown for {food}?",
    "Is {food} high in {nutrient}?",
    "Does {food} have a lot of {nutrient}?",
    "How does the {nutrient} in {food} compare to the daily requirement?",
    "What is the total {nutrient} per {amount} of {food}?",
    "How rich is {food} in {nutrient}?",
    "What is the {nutrient} macro for {food}?",
]

HA_TEMPLATES = [
    "What foods should people with {condition} avoid?",
    "What should I eat if I have {condition}?",
    "Best foods for managing {condition}?",
    "Is it safe to eat {food} with {condition}?",
    "Diet recommendations for {condition} patients?",
    "What foods help reduce symptoms of {condition}?",
    "Can people with {condition} eat {food}?",
    "What is a healthy diet for someone with {condition}?",
    "Should I avoid {food} if I have {condition}?",
    "What fruits are safe for {condition} patients?",
    "What vegetables are recommended for {condition}?",
    "Are there foods that worsen {condition}?",
    "What is the best diet for controlling {condition}?",
    "Which foods are beneficial for people with {condition}?",
    "Is {food} recommended or not recommended for {condition}?",
    "How should someone with {condition} adjust their diet?",
    "What eating habits help manage {condition}?",
    "Are grains good for someone with {condition}?",
    "What proteins are best for a person with {condition}?",
    "Should someone with {condition} limit dairy intake?",
    "Is a low-carb diet good for {condition}?",
    "What snacks are suitable for someone with {condition}?",
    "How does diet affect {condition}?",
    "Can eating {food} help with {condition}?",
    "What dietary changes help with {condition}?",
]

BOTH_TEMPLATES = [
    "Is {food} good for {condition}? How much {nutrient} does it contain?",
    "Can {condition} patients eat {food}? What is the {nutrient} content?",
    "How many {nutrient} are in {food} and is it safe for {condition}?",
    "Is {food} high in {nutrient} and beneficial for {condition}?",
    "What is the {nutrient} in {food} and is it good for {condition}?",
    "For {condition} management, how much {food} can I eat given its {nutrient} content?",
    "Does {food} have enough {nutrient} to help with {condition}?",
    "Is {food} a good source of {nutrient} for people with {condition}?",
    "How does {food}'s {nutrient} level relate to {condition}?",
    "Can the {nutrient} in {food} benefit {condition} patients?",
    "Is {food} recommended for {condition}? What is its {nutrient} content per serving?",
    "How much {nutrient} is in {amount} of {food} and is that good for {condition}?",
    "Should someone with {condition} eat {food} given its {nutrient} content?",
    "What makes {food} suitable or unsuitable for {condition} based on its {nutrient}?",
    "Is the {nutrient} level in {food} appropriate for someone with {condition}?",
    "How does eating {food} affect {condition} given its high {nutrient} content?",
    "Can {food} help control {condition} because of its {nutrient}?",
    "Is {amount} of {food} too much {nutrient} for a {condition} patient?",
    "What role does the {nutrient} in {food} play for people with {condition}?",
    "Is the {nutrient} content of {food} beneficial or harmful for {condition}?",
]


def fill(template: str) -> str:
    result = template
    if "{food}" in result:
        result = result.replace("{food}", random.choice(FOODS))
    if "{nutrient}" in result:
        result = result.replace("{nutrient}", random.choice(NUTRIENTS))
    if "{amount}" in result:
        result = result.replace("{amount}", random.choice(AMOUNTS))
    if "{condition}" in result:
        result = result.replace("{condition}", random.choice(CONDITIONS))
    return result


def generate(templates: list[str], target: int) -> list[str]:
    seen: set[str] = set()
    questions: list[str] = []
    max_attempts = target * 20

    for _ in range(max_attempts):
        if len(questions) >= target:
            break
        q = fill(random.choice(templates))
        if q not in seen:
            seen.add(q)
            questions.append(q)

    return questions[:target]


def main() -> None:
    Path("data/en").mkdir(parents=True, exist_ok=True)

    nl  = generate(NL_TEMPLATES,   600)
    ha  = generate(HA_TEMPLATES,   600)
    bot = generate(BOTH_TEMPLATES, 600)

    rows = (
        [{"question": q, "intent_label": "NUTRITION_LOOKUP"} for q in nl]
        + [{"question": q, "intent_label": "HEALTH_ADVICE"}   for q in ha]
        + [{"question": q, "intent_label": "BOTH"}            for q in bot]
    )
    random.shuffle(rows)

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv("data/en/synthetic_intent.csv", index=False)
    print(f"synthetic_intent.csv — {len(df_raw)} rows")
    print(df_raw["intent_label"].value_counts().to_string())

    # Training-ready version: 500/class, rename columns for training notebook
    sampled_dfs = []
    for label, group in df_raw.groupby("intent_label"):
        sampled_dfs.append(group.sample(500, random_state=42))
    df_train = pd.concat(sampled_dfs).reset_index(drop=True)
    df_train = (
        df_train
        .rename(columns={"question": "text", "intent_label": "label"})
        [["text", "label"]]
    )
    df_train.to_csv("data/en/intent_data.csv", index=False)
    print(f"\nintent_data.csv — {len(df_train)} rows")
    print(df_train["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
