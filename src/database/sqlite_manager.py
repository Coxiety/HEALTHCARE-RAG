from __future__ import annotations

import sqlite3
from pathlib import Path


class UsdaLookupError(Exception):
    pass


class SqliteManager:
    """USDA SQLite lookup for English food queries."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise UsdaLookupError(
                f"DB not found: {self.db_path}. Run main/build_usda_db.py first."
            )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _find_food(self, food_name: str) -> sqlite3.Row | None:
        """Find the best-matching food by keyword search on description."""
        words = [w.strip().lower() for w in food_name.split() if w.strip()]
        if not words:
            return None
        where_clauses = ["LOWER(f.description) LIKE ?" for _ in words]
        params = [f"%{w}%" for w in words]
        sql = f"""
            SELECT f.fdc_id, f.description, f.data_type
            FROM foods f
            LEFT JOIN food_nutrients fn ON f.fdc_id = fn.fdc_id
            WHERE {" AND ".join(where_clauses)}
            GROUP BY f.fdc_id
            ORDER BY
                CASE WHEN LOWER(f.description) LIKE '%sausage%'
                       OR LOWER(f.description) LIKE '%frankfurter%'
                       OR LOWER(f.description) LIKE '%lunchmeat%'
                       OR LOWER(f.description) LIKE '%salami%'
                       OR LOWER(f.description) LIKE '%bologna%'
                       OR LOWER(f.description) LIKE '%hot dog%'
                     THEN 1 ELSE 0 END ASC,
                CASE WHEN f.data_type = 'foundation_food' THEN 0 ELSE 1 END,
                COUNT(fn.nutrient_id) DESC,
                LENGTH(f.description) ASC
            LIMIT 1
        """
        with self._connect() as conn:
            return conn.execute(sql, params).fetchone()

    def _get_nutrient(self, fdc_id: int, nutrient_name: str) -> dict | None:
        sql = """
            SELECT n.name, n.unit_name, fn.amount
            FROM food_nutrients fn
            JOIN nutrients n ON fn.nutrient_id = n.id
            WHERE fn.fdc_id = ? AND LOWER(n.name) = LOWER(?)
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (fdc_id, nutrient_name)).fetchone()
        if not row:
            return None
        return {
            "nutrient_name": row["name"],
            "unit": row["unit_name"],
            "amount_per_100g": row["amount"],
        }

    def lookup_en(self, food_name: str, nutrient_name: str | None = None) -> dict | None:
        """Look up a food by English name. Returns all key nutrients if no specific nutrient requested."""
        food = self._find_food(food_name)
        if food is None:
            return None

        result = {"fdc_id": food["fdc_id"], "food_description": food["description"]}

        if nutrient_name:
            nutrient = self._get_nutrient(food["fdc_id"], nutrient_name)
            if nutrient:
                result.update(nutrient)
        else:
            key_nutrients = [
                "Protein", "Energy", "Total lipid (fat)",
                "Carbohydrate, by difference", "Fiber, total dietary",
            ]
            nutrients = {}
            for name in key_nutrients:
                row = self._get_nutrient(food["fdc_id"], name)
                if row:
                    nutrients[name] = {"amount": row["amount_per_100g"], "unit": row["unit"]}
            if nutrients:
                result["nutrients_per_100g"] = nutrients

        return result
