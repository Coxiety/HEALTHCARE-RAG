# Intent v2 Baseline Before Retraining

Current model: `models/classifier_bert`

## Balanced Hard Test

File: `data/en/intent_v2/intent_hard_test.csv`

- Rows: 60
- Accuracy: 0.8833
- Macro-F1: 0.8796

Confusion matrix, rows = gold, columns = predicted:

| Gold \ Pred | NUTRITION_LOOKUP | HEALTH_ADVICE | BOTH |
|---|---:|---:|---:|
| NUTRITION_LOOKUP | 20 | 0 | 0 |
| HEALTH_ADVICE | 0 | 13 | 7 |
| BOTH | 0 | 0 | 20 |

Interpretation: the model handles clear nutrition lookup and clear BOTH examples well, but still sends some health-only food/herb questions to BOTH.

## Hotpot Relabel Hard Negatives

File: `data/en/intent_v2/intent_hotpot_relabel_test.csv`

- Rows: 71
- Accuracy: 0.3239
- Macro-F1: 0.1631

Confusion matrix:

| Gold \ Pred | NUTRITION_LOOKUP | HEALTH_ADVICE | BOTH |
|---|---:|---:|---:|
| HEALTH_ADVICE | 2 | 23 | 46 |

Interpretation: this is the main issue. Food/herb + clinical-effect questions are often over-routed to BOTH, even when no exact USDA nutrition lookup is requested.

## Decision

Retrain intent classifier with `intent_train_v2.csv`, then evaluate on:

- `intent_hard_test.csv`
- `intent_hotpot_relabel_test.csv`

Use the new model only if it improves the hotpot hard negatives without hurting the balanced hard test.
