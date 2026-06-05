from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


class UsdaLookupError(Exception):
    pass


class SqliteManager:
    """USDA SQLite lookup. Supports both Vietnamese (via mapping) and direct English queries."""

    def __init__(self, db_path: str | Path, vi_mapping_path: str | Path | None = None):
        self.db_path = Path(db_path)
        self.vi_mapping = self._load_mapping(Path(vi_mapping_path)) if vi_mapping_path else {}

    @staticmethod
    def _load_mapping(mapping_path: Path) -> dict[str, str]:
        if not mapping_path.exists():
            return {}
        mapping: dict[str, str] = {}
        with mapping_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vi = row.get("vi_name", "").strip().lower()
                en = row.get("en_keyword", "").strip().lower()
                if vi and en:
                    mapping[vi] = en
        return mapping

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise UsdaLookupError(
                f"DB not found: {self.db_path}. Run main/build_usda_db.py first."
            )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _resolve_keyword(self, vi_name: str) -> str:
        return self.vi_mapping.get(vi_name.strip().lower(), vi_name.strip().lower())

    def find_food(self, vi_name: str) -> sqlite3.Row | None:
        keyword = self._resolve_keyword(vi_name)
        words = [w.strip().lower() for w in keyword.split() if w.strip()]
        if not words:
            return None
        where_clauses = [ "LOWER(f.description) LIKE ?" for _ in words ]
        params = [ f"%{w}%" for w in words ]
        sql = f"""
            SELECT f.fdc_id, f.description, f.data_type
            FROM foods f
            LEFT JOIN food_nutrients fn ON f.fdc_id = fn.fdc_id
            WHERE {" AND ".join(where_clauses)}
            GROUP BY f.fdc_id
            ORDER BY
                CASE WHEN LOWER(f.description) LIKE '%sausage%' OR LOWER(f.description) LIKE '%frankfurter%' OR LOWER(f.description) LIKE '%lunchmeat%' OR LOWER(f.description) LIKE '%salami%' OR LOWER(f.description) LIKE '%bologna%' OR LOWER(f.description) LIKE '%hot dog%' THEN 1 ELSE 0 END ASC,
                CASE WHEN f.data_type = 'foundation_food' THEN 0 ELSE 1 END,
                COUNT(fn.nutrient_id) DESC,
                LENGTH(f.description) ASC
            LIMIT 1
        """
        with self._connect() as conn:
            return conn.execute(sql, params).fetchone()

    def get_nutrient(self, fdc_id: int, nutrient_name: str) -> dict | None:
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

    def lookup(self, vi_name: str, nutrient_name: str) -> dict | None:
        """Vietnamese food + nutrient name → USDA row, or None if not found."""
        food = self.find_food(vi_name)
        if not food:
            return None
        nutrient = self.get_nutrient(food["fdc_id"], nutrient_name)
        if not nutrient:
            return None
        return {
            "fdc_id": food["fdc_id"],
            "food_description": food["description"],
            "data_type": food["data_type"],
            **nutrient,
        }

    def lookup_en(self, food_name: str, nutrient_name: str | None = None) -> dict | None:
        """Direct English food name lookup — no VN mapping step."""
        words = [w.strip().lower() for w in food_name.split() if w.strip()]
        if not words:
            return None
        where_clauses = [ "LOWER(f.description) LIKE ?" for _ in words ]
        params = [ f"%{w}%" for w in words ]
        sql = f"""
            SELECT f.fdc_id, f.description, f.data_type
            FROM foods f
            LEFT JOIN food_nutrients fn ON f.fdc_id = fn.fdc_id
            WHERE {" AND ".join(where_clauses)}
            GROUP BY f.fdc_id
            ORDER BY
                CASE WHEN LOWER(f.description) LIKE '%sausage%' OR LOWER(f.description) LIKE '%frankfurter%' OR LOWER(f.description) LIKE '%lunchmeat%' OR LOWER(f.description) LIKE '%salami%' OR LOWER(f.description) LIKE '%bologna%' OR LOWER(f.description) LIKE '%hot dog%' THEN 1 ELSE 0 END ASC,
                CASE WHEN f.data_type = 'foundation_food' THEN 0 ELSE 1 END,
                COUNT(fn.nutrient_id) DESC,
                LENGTH(f.description) ASC
            LIMIT 1
        """
        with self._connect() as conn:
            food = conn.execute(sql, params).fetchone()
        if food is None:
            return None
        result = {"fdc_id": food["fdc_id"], "food_description": food["description"]}
        if nutrient_name:
            nutrient = self.get_nutrient(food["fdc_id"], nutrient_name)
            if nutrient:
                result.update(nutrient)
        else:
            key_nutrients = ["Protein", "Energy", "Total lipid (fat)", "Carbohydrate, by difference", "Fiber, total dietary"]
            nutrients = {}
            for name in key_nutrients:
                row = self.get_nutrient(food["fdc_id"], name)
                if row:
                    nutrients[name] = {"amount": row["amount_per_100g"], "unit": row["unit"]}
            if nutrients:
                result["nutrients_per_100g"] = nutrients
        return result

    def list_mapped_foods(self) -> list[str]:
        return sorted(self.vi_mapping.keys())


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2]
    db = SqliteManager(
        db_path=BASE / "data/usda_food.db",
        vi_mapping_path=BASE / "data/vi_food_mapping.csv",
    )

    print("mapped foods:", db.list_mapped_foods())
    print()

    result = db.lookup("uc ga", "Protein")
    if result:
        print(f"{result['food_description']}")
        print(f"  {result['nutrient_name']}: {result['amount_per_100g']} {result['unit']} / 100g")
    else:
        print("not found.")
