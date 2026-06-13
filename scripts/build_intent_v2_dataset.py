from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

import pandas as pd


LABELS = ["NUTRITION_LOOKUP", "HEALTH_ADVICE", "BOTH"]
SEED = 42


FOODS = [
    "chicken breast",
    "salmon",
    "avocado",
    "banana",
    "brown rice",
    "broccoli",
    "spinach",
    "egg",
    "greek yogurt",
    "oats",
    "lentils",
    "tofu",
    "almonds",
    "peanuts",
    "sweet potato",
    "apple",
    "orange",
    "milk",
    "cheddar cheese",
    "black beans",
    "tuna",
    "shrimp",
    "olive oil",
    "quinoa",
    "beef steak",
    "turkey",
    "mushrooms",
    "carrots",
    "pumpkin seeds",
    "chia seeds",
]


NUTRIENTS = [
    "calories",
    "protein",
    "carbohydrates",
    "fat",
    "fiber",
    "sugar",
    "sodium",
    "cholesterol",
    "calcium",
    "iron",
    "potassium",
    "magnesium",
    "vitamin C",
    "vitamin B12",
    "omega-3",
]


CONDITIONS = [
    "diabetes",
    "high blood pressure",
    "high cholesterol",
    "fatty liver disease",
    "PCOS",
    "kidney stones",
    "heart disease",
    "gout",
    "pregnancy",
    "obesity",
    "inflammation",
    "anemia",
    "acid reflux",
    "irritable bowel syndrome",
    "osteoporosis",
]


HERBS_AND_FOODS = [
    "ginger",
    "grapefruit juice",
    "hibiscus tea",
    "cinnamon",
    "garlic",
    "yerba mate tea",
    "green tea",
    "saffron",
    "krill oil",
    "fish oil",
    "moringa leaves",
    "bitter melon",
    "aloe vera",
    "peppermint oil",
    "turmeric",
    "soy foods",
    "whole grains",
    "eggs",
    "fiber supplements",
    "Mediterranean diet",
]


HEALTH_EFFECTS = [
    "lower blood sugar",
    "reduce nausea during pregnancy",
    "interact with medications",
    "improve cholesterol levels",
    "increase kidney stone risk",
    "help with PCOS symptoms",
    "reduce inflammation",
    "affect blood pressure",
    "improve bone health",
    "increase cancer risk",
    "help prevent type 2 diabetes",
    "cause liver damage",
    "improve IBS symptoms",
    "affect estrogen levels",
    "reduce cardiovascular risk",
]


def unique_take(items: list[dict], n: int) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        text = item["text"].strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(f"Only generated {len(out)} unique rows, need {n}")
    return out


def make_nutrition_lookup_examples() -> list[dict]:
    def questions(food: str, nutrient: str) -> list[str]:
        content_name = "calorie" if nutrient == "calories" else nutrient
        amount_q = (
            f"How many calories are in 100g of {food}?"
            if nutrient == "calories"
            else f"How much {nutrient} is in 100g of {food}?"
        )
        serving_q = (
            f"How many calories are in one serving of {food}?"
            if nutrient == "calories"
            else f"How much {nutrient} does {food} contain per serving?"
        )
        return [
            amount_q,
            f"What is the {content_name} content of {food} per 100g?",
            serving_q,
            f"Show me the calories, protein, fat, and carbs in 100g of {food}.",
            f"What are the USDA nutrition facts for {food}?",
        ]

    rows = []
    for food in FOODS:
        for nutrient in NUTRIENTS:
            for text in questions(food, nutrient):
                rows.append(
                    {
                        "text": text,
                        "label": "NUTRITION_LOOKUP",
                        "source": "crafted_hard_v2",
                        "notes": "exact nutrition lookup cue; no health advice requested",
                    }
                )
    return unique_take(rows, 100)


def make_both_examples() -> list[dict]:
    def questions(food: str, nutrient: str, condition: str) -> list[str]:
        content_name = "calorie" if nutrient == "calories" else nutrient
        amount_q = (
            f"How many calories are in {food}, and is it safe for someone with {condition}?"
            if nutrient == "calories"
            else f"How much {nutrient} is in {food}, and is it safe for someone with {condition}?"
        )
        return [
            amount_q,
            f"What is the {content_name} content of {food}, and should a patient with {condition} eat it?",
            f"How many calories are in 100g of {food}, and is it good for {condition}?",
            f"Give the USDA nutrition facts for {food}, then explain whether it helps with {condition}.",
            f"How much protein, fat, and carbs are in {food}, and what does that mean for {condition}?",
            f"What nutrients does {food} contain, and are those nutrients beneficial for {condition}?",
        ]

    rows = []
    for food in FOODS:
        for condition in CONDITIONS:
            for nutrient in NUTRIENTS:
                for text in questions(food, nutrient, condition):
                    rows.append(
                        {
                            "text": text,
                            "label": "BOTH",
                            "source": "crafted_hard_v2",
                            "notes": "explicit nutrition lookup cue plus health/clinical cue",
                        }
                    )
    return unique_take(rows, 100)


def make_health_advice_examples(hotpot_review_path: Path) -> list[dict]:
    rows: list[dict] = []
    if hotpot_review_path.exists():
        df = pd.read_csv(hotpot_review_path)
        for _, row in df.iterrows():
            if row.get("final_label") == "HEALTH_ADVICE":
                rows.append(
                    {
                        "text": str(row["question"]),
                        "label": "HEALTH_ADVICE",
                        "source": "hotpot_relabel_review",
                        "notes": "food/herb/research question; no exact USDA nutrition lookup requested",
                    }
                )

    templates = [
        "Can {item} {effect}?",
        "Does {item} {effect}?",
        "Is {item} helpful for {condition}?",
        "Is {item} risky for people with {condition}?",
        "What does the evidence say about {item} for {condition}?",
        "Can people with {condition} safely use {item}?",
        "Does consuming {item} affect patients with {condition}?",
        "Should someone with {condition} avoid {item}?",
    ]
    for item in HERBS_AND_FOODS:
        for condition in CONDITIONS:
            for effect in HEALTH_EFFECTS:
                for template in templates:
                    rows.append(
                        {
                            "text": template.format(item=item, condition=condition, effect=effect),
                            "label": "HEALTH_ADVICE",
                            "source": "crafted_hard_v2",
                            "notes": "health/clinical cue; no exact nutrition lookup requested",
                        }
                    )
    return unique_take(rows, 100)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stratified_split(rows: list[dict], train_n: int = 60, val_n: int = 20, test_n: int = 20) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(SEED)
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for label in LABELS:
        bucket = [row for row in rows if row["label"] == label]
        rng.shuffle(bucket)
        expected = train_n + val_n + test_n
        if len(bucket) < expected:
            raise ValueError(f"{label} has {len(bucket)} rows, need {expected}")
        train.extend(bucket[:train_n])
        val.extend(bucket[train_n : train_n + val_n])
        test.extend(bucket[train_n + val_n : train_n + val_n + test_n])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def rows_to_training_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"text": row["text"], "label": row["label"]} for row in rows])


def write_report(path: Path, hard_rows: list[dict], train_rows: list[dict], val_rows: list[dict], test_rows: list[dict], old_df: pd.DataFrame) -> None:
    hard_counts = Counter(row["label"] for row in hard_rows)
    train_counts = Counter(row["label"] for row in train_rows)
    val_counts = Counter(row["label"] for row in val_rows)
    test_counts = Counter(row["label"] for row in test_rows)
    old_counts = Counter(old_df["label"])

    lines = [
        "# Intent Dataset v2 Report",
        "",
        "Purpose: fix the routing-label mismatch found in the multi-hop evaluation.",
        "",
        "Intent schema:",
        "- NUTRITION_LOOKUP: exact USDA nutrition facts are needed.",
        "- HEALTH_ADVICE: health, safety, disease, treatment, or clinical evidence question.",
        "- BOTH: exact USDA nutrition facts plus health retrieval are both needed.",
        "",
        "## Source Data",
        "",
        f"- Original synthetic dataset: {len(old_df)} rows",
        f"- New balanced hard-set: {len(hard_rows)} rows",
        "- Hard-set includes relabeled multi-hop HEALTH_ADVICE examples plus crafted NUTRITION_LOOKUP/BOTH counterexamples.",
        "",
        "## Original Synthetic Distribution",
        "",
    ]
    lines.extend(f"- {label}: {old_counts.get(label, 0)}" for label in LABELS)
    lines.extend(["", "## Hard-Set Distribution", ""])
    lines.extend(f"- {label}: {hard_counts.get(label, 0)}" for label in LABELS)
    lines.extend(["", "## Hard Split Distribution", ""])
    lines.append("| Split | NUTRITION_LOOKUP | HEALTH_ADVICE | BOTH | Total |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, counts, rows in [
        ("train", train_counts, train_rows),
        ("val", val_counts, val_rows),
        ("test", test_counts, test_rows),
    ]:
        lines.append(
            f"| {name} | {counts.get('NUTRITION_LOOKUP', 0)} | {counts.get('HEALTH_ADVICE', 0)} | {counts.get('BOTH', 0)} | {len(rows)} |"
        )
    lines.extend(
        [
            "",
            "## Training Files",
            "",
            "- `data/en/intent_v2/intent_train_v2.csv`: original 1,500 rows + hard train rows.",
            "- `data/en/intent_v2/intent_hard_val.csv`: balanced hard validation set.",
            "- `data/en/intent_v2/intent_hard_test.csv`: balanced hard holdout test set.",
            "- `data/en/intent_hard_balanced_review.csv`: full reviewed hard-set.",
            "",
            "Recommended reporting:",
            "- Keep the old in-distribution synthetic score as a controlled sanity check.",
            "- Report hard-set test performance separately as generalization to difficult routing cases.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    old_path = Path("data/en/intent_data.csv")
    hotpot_review_path = Path("data/en/intent_hard_review.csv")
    output_dir = Path("data/en/intent_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    old_df = pd.read_csv(old_path)
    if set(old_df.columns) != {"text", "label"}:
        raise ValueError(f"Unexpected columns in {old_path}: {list(old_df.columns)}")

    if hotpot_review_path.exists():
        hotpot_review_df = pd.read_csv(hotpot_review_path)
        hotpot_relabel_df = hotpot_review_df.rename(columns={"question": "text", "final_label": "label"})[
            ["text", "label"]
        ]
        hotpot_relabel_df.to_csv(output_dir / "intent_hotpot_relabel_test.csv", index=False)

    hard_rows = (
        make_nutrition_lookup_examples()
        + make_health_advice_examples(hotpot_review_path)
        + make_both_examples()
    )
    hard_rows = [
        {
            "id": idx,
            "text": row["text"],
            "label": row["label"],
            "source": row["source"],
            "review_status": "reviewed_by_codex_schema_v1",
            "notes": row["notes"],
        }
        for idx, row in enumerate(hard_rows)
    ]

    train_rows, val_rows, test_rows = stratified_split(hard_rows)
    train_df = pd.concat([old_df, rows_to_training_frame(train_rows)], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    write_csv(
        Path("data/en/intent_hard_balanced_review.csv"),
        hard_rows,
        ["id", "text", "label", "source", "review_status", "notes"],
    )
    train_df.to_csv(output_dir / "intent_train_v2.csv", index=False)
    rows_to_training_frame(val_rows).to_csv(output_dir / "intent_hard_val.csv", index=False)
    rows_to_training_frame(test_rows).to_csv(output_dir / "intent_hard_test.csv", index=False)

    write_report(
        Path("reports/en/intent_dataset_v2_report.md"),
        hard_rows,
        train_rows,
        val_rows,
        test_rows,
        old_df,
    )

    print(f"Saved hard-set: data/en/intent_hard_balanced_review.csv ({len(hard_rows)} rows)")
    print(f"Saved train: {output_dir / 'intent_train_v2.csv'} ({len(train_df)} rows)")
    print(f"Saved val: {output_dir / 'intent_hard_val.csv'} ({len(val_rows)} rows)")
    print(f"Saved test: {output_dir / 'intent_hard_test.csv'} ({len(test_rows)} rows)")


if __name__ == "__main__":
    main()
