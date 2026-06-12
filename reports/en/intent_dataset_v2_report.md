# Intent Dataset v2 Report

Purpose: fix the routing-label mismatch found in the multi-hop evaluation.

Intent schema:
- NUTRITION_LOOKUP: exact USDA nutrition facts are needed.
- HEALTH_ADVICE: health, safety, disease, treatment, or clinical evidence question.
- BOTH: exact USDA nutrition facts plus health retrieval are both needed.

## Source Data

- Original synthetic dataset: 1500 rows
- New balanced hard-set: 300 rows
- Hard-set includes relabeled multi-hop HEALTH_ADVICE examples plus crafted NUTRITION_LOOKUP/BOTH counterexamples.

## Original Synthetic Distribution

- NUTRITION_LOOKUP: 500
- HEALTH_ADVICE: 500
- BOTH: 500

## Hard-Set Distribution

- NUTRITION_LOOKUP: 100
- HEALTH_ADVICE: 100
- BOTH: 100

## Hard Split Distribution

| Split | NUTRITION_LOOKUP | HEALTH_ADVICE | BOTH | Total |
|---|---:|---:|---:|---:|
| train | 60 | 60 | 60 | 180 |
| val | 20 | 20 | 20 | 60 |
| test | 20 | 20 | 20 | 60 |

## Training Files

- `data/en/intent_v2/intent_train_v2.csv`: original 1,500 rows + hard train rows.
- `data/en/intent_v2/intent_hard_val.csv`: balanced hard validation set.
- `data/en/intent_v2/intent_hard_test.csv`: balanced hard holdout test set.
- `data/en/intent_hard_balanced_review.csv`: full reviewed hard-set.

Recommended reporting:
- Keep the old in-distribution synthetic score as a controlled sanity check.
- Report hard-set test performance separately as generalization to difficult routing cases.
