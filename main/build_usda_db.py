from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from pipeline import SQLITE_PATH, USDA_CSV_DIR


TARGET_NUTRIENTS = [
    "Protein",
    "Energy",
    "Carbohydrate, by difference",
    "Total lipid (fat)",
    "Fiber, total dietary",
    "Sugars, Total",
    "Sodium, Na",
]

TARGET_DATA_TYPES = {"foundation_food", "sr_legacy_food", "survey_fndds_food"}


def assert_required_files(data_dir: Path) -> None:
    required = ["food.csv", "nutrient.csv", "food_nutrient.csv"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing USDA files: {missing}")


def build_database() -> None:
    data_dir = USDA_CSV_DIR
    assert_required_files(data_dir)

    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()

    print(f"[1/5] Loading food table from {data_dir}")
    food = pd.read_csv(
        data_dir / "food.csv",
        usecols=["fdc_id", "description", "data_type"],
        low_memory=False,
    )
    food = food[food["data_type"].isin(TARGET_DATA_TYPES)].copy()
    food["description"] = food["description"].fillna("").astype(str)
    fdc_ids = set(food["fdc_id"].astype(int).tolist())

    print(f"[2/5] Loading nutrient table")
    nutrient = pd.read_csv(
        data_dir / "nutrient.csv",
        usecols=["id", "name", "unit_name"],
        low_memory=False,
    )
    nutrient = nutrient[nutrient["name"].isin(TARGET_NUTRIENTS)].copy()
    nutrient_ids = set(nutrient["id"].astype(int).tolist())

    print(f"[3/5] Filtering food_nutrient in chunks (this may take a while)")
    filtered_chunks: list[pd.DataFrame] = []
    chunks = pd.read_csv(
        data_dir / "food_nutrient.csv",
        usecols=["fdc_id", "nutrient_id", "amount"],
        chunksize=1_000_000,
        low_memory=False,
    )
    for i, chunk in enumerate(chunks, start=1):
        chunk = chunk[
            chunk["fdc_id"].isin(fdc_ids) & chunk["nutrient_id"].isin(nutrient_ids)
        ].copy()
        if not chunk.empty:
            filtered_chunks.append(chunk)
        if i % 5 == 0:
            print(f"  - processed {i} chunks")

    if not filtered_chunks:
        raise RuntimeError("No nutrient rows matched filters. Check USDA schema.")

    food_nutrient = pd.concat(filtered_chunks, ignore_index=True)

    print(f"[4/5] Writing SQLite database to {SQLITE_PATH}")
    with sqlite3.connect(SQLITE_PATH) as conn:
        food.to_sql("foods", conn, if_exists="replace", index=False)
        nutrient.to_sql("nutrients", conn, if_exists="replace", index=False)
        food_nutrient.to_sql("food_nutrients", conn, if_exists="replace", index=False)

        conn.execute("CREATE INDEX idx_food_desc ON foods(description)")
        conn.execute("CREATE INDEX idx_food_nutrients_fdc ON food_nutrients(fdc_id)")
        conn.execute(
            "CREATE INDEX idx_food_nutrients_fdc_nutr ON food_nutrients(fdc_id, nutrient_id)"
        )
        conn.commit()

    print("[5/5] Done. SQLite database is ready.")


if __name__ == "__main__":
    build_database()
