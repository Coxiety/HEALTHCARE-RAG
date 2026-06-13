# Intent Hard-Set Relabel Audit

This file audits the multi-hop nutrition dataset against the project routing intent schema.

Schema:
- NUTRITION_LOOKUP: exact nutrition facts from USDA are needed.
- HEALTH_ADVICE: health, disease, safety, or clinical advice/research question.
- BOTH: both exact USDA nutrition facts and health retrieval are needed.

Total rows: 71
Rows whose suggested label differs from old label: 36
Rows flagged for manual review: 60

## Old Label Distribution

- BOTH: 37
- HEALTH_ADVICE: 34

## Suggested Label Distribution

- BOTH: 1
- HEALTH_ADVICE: 70

## Current Classifier Prediction Distribution

- BOTH: 46
- HEALTH_ADVICE: 23
- NUTRITION_LOOKUP: 2

## Review Guidance

Open `data/en/intent_hard_review.csv` and fill `final_label`.
Keep BOTH only when the question truly needs exact USDA nutrition data and health retrieval.
If the question is about evidence, safety, disease risk, treatment, or clinical effect only, use HEALTH_ADVICE.
