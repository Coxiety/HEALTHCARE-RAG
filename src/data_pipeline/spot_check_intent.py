"""CLI tool for manual spot-check of synthetic intent labels.

Reviews a stratified random sample and computes human-annotator accuracy.
Results saved to reports/en/spot_check_results.csv.

Usage:
    python src/data_pipeline/spot_check_intent.py
    python src/data_pipeline/spot_check_intent.py --n 100 --input data/en/synthetic_intent.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_DEFAULT_INPUT = "data/en/synthetic_intent.csv"
_DEFAULT_OUTPUT = "reports/en/spot_check_results.csv"
_N = 100

_LABEL_COLORS = {
    "NUTRITION_LOOKUP": "\033[94m",  # blue
    "HEALTH_ADVICE": "\033[92m",     # green
    "BOTH": "\033[93m",              # yellow
}
_RESET = "\033[0m"


def _colored(text: str, label: str) -> str:
    color = _LABEL_COLORS.get(label, "")
    return f"{color}{text}{_RESET}"


def spot_check(input_path: str, output_path: str, n: int) -> None:
    df = pd.read_csv(input_path)
    if "intent_label" not in df.columns:
        raise ValueError(f"Expected 'intent_label' column in {input_path}")

    labels = sorted(df["intent_label"].unique())
    per_class = n // len(labels)

    sample = (
        df.groupby("intent_label", group_keys=False)
        .apply(lambda g: g.sample(min(per_class, len(g)), random_state=42))
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    print(f"\n=== Spot Check — {len(sample)} questions ===")
    print("Keys:  1 = correct   0 = wrong   s = skip   q = quit\n")

    results: list[dict] = []
    for i, row in sample.iterrows():
        label = row["intent_label"]
        print(f"[{i+1}/{len(sample)}] {_colored(label, label)}")
        print(f"  {row['question']}")

        while True:
            key = input("  → ").strip().lower()
            if key in ("1", "0", "s", "q"):
                break
            print("  Invalid. Use 1, 0, s, or q.")

        if key == "q":
            print("\nQuit early.")
            break
        if key != "s":
            results.append({
                "question": row["question"],
                "intent_label": label,
                "correct": int(key),
            })

    if not results:
        print("No results recorded.")
        return

    out_df = pd.DataFrame(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    accuracy = out_df["correct"].mean()
    print(f"\n=== Results ({len(out_df)} reviewed) ===")
    print(f"Overall accuracy: {accuracy:.1%}")
    print("\nPer-class accuracy:")
    print(out_df.groupby("intent_label")["correct"].mean().apply(lambda x: f"{x:.1%}").to_string())
    print(f"\nSaved → {output_path}")

    if accuracy < 0.80:
        print("\n⚠️  Accuracy < 80% — consider regenerating with stricter prompts")
    else:
        print("\n✅  Accuracy ≥ 80% — dataset quality acceptable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spot-check synthetic intent labels")
    parser.add_argument("--n", type=int, default=_N,
                        help="Number of questions to review (default: 100)")
    parser.add_argument("--input", default=_DEFAULT_INPUT,
                        help="Input CSV (synthetic_intent.csv)")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT,
                        help="Output CSV for results")
    args = parser.parse_args()
    spot_check(args.input, args.output, args.n)
