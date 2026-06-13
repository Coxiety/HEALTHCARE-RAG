# Intent Hard-Set Evaluation

Evaluation uses `final_label` from `data/en/intent_hard_review.csv`.
`current_predicted_intent` is taken from the latest multi-hop evaluation cases file.

Total rows: 71
Accuracy: 0.3239
Macro-F1: 0.1631

## Label Distribution

| Label | Gold | Predicted |
|---|---:|---:|
| BOTH | 0 | 46 |
| HEALTH_ADVICE | 71 | 23 |
| NUTRITION_LOOKUP | 0 | 2 |

## Per-Label Metrics

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BOTH | 0.0000 | 0.0000 | 0.0000 | 0 |
| HEALTH_ADVICE | 1.0000 | 0.3239 | 0.4894 | 71 |
| NUTRITION_LOOKUP | 0.0000 | 0.0000 | 0.0000 | 0 |

## Confusion Matrix

| Gold \ Pred | BOTH | HEALTH_ADVICE | NUTRITION_LOOKUP |
|---|---|---|---|
| BOTH | 0 | 0 | 0 |
| HEALTH_ADVICE | 46 | 23 | 2 |
| NUTRITION_LOOKUP | 0 | 0 | 0 |
