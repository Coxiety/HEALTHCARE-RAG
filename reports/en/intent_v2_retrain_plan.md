# Intent Classifier v2 Retrain Plan

## Goal

Fix the routing-label mismatch found in multi-hop evaluation.

Important: this retrains only the intent classifier. It does not touch the NER BERT+CRF+Viterbi model.

## Intent Schema

- `NUTRITION_LOOKUP`: exact USDA nutrition facts are needed.
- `HEALTH_ADVICE`: health, safety, disease, treatment, or clinical evidence question.
- `BOTH`: exact USDA nutrition facts plus health retrieval are both needed.

Examples:

| Query | Label |
|---|---|
| How many calories are in 100g chicken breast? | NUTRITION_LOOKUP |
| Can ginger help nausea during pregnancy? | HEALTH_ADVICE |
| How much protein is in salmon, and is it good for cholesterol? | BOTH |

## Files Created

| File | Purpose |
|---|---|
| `data/en/intent_hard_review.csv` | Relabeled 71 multi-hop questions under the routing schema |
| `data/en/intent_hard_balanced_review.csv` | 300-row balanced hard-set, 100/class |
| `data/en/intent_v2/intent_train_v2.csv` | Training data: original 1,500 + hard train 180 = 1,680 rows |
| `data/en/intent_v2/intent_hard_val.csv` | Balanced hard validation set, 60 rows |
| `data/en/intent_v2/intent_hard_test.csv` | Balanced hard holdout test set, 60 rows |
| `data/en/intent_v2/intent_hotpot_relabel_test.csv` | 71 relabeled multi-hop HEALTH_ADVICE hard negatives |

## Baseline Current Model

Current `models/classifier_bert`:

| Eval Set | Accuracy | Macro-F1 | Main Issue |
|---|---:|---:|---|
| Balanced hard test | 0.8833 | 0.8796 | 7/20 HEALTH_ADVICE predicted as BOTH |
| Hotpot relabel hard negatives | 0.3239 | 0.1631 | 46/71 HEALTH_ADVICE predicted as BOTH |

Interpretation:

The model is not globally broken. It mainly overpredicts `BOTH` for food/herb + clinical-effect questions.

## Colab Training Steps

Notebook version:

Open `notebooks/en/train_intent_bert_v2_colab.ipynb` in Colab and run cells top to bottom.

Command-line version:

1. Upload or sync the project folder to Colab/Drive.

2. Install dependencies if needed:

```bash
pip uninstall -y torchvision
pip install -U "transformers==4.44.2" "accelerate>=0.33.0" scikit-learn pandas torch
```

3. Run from the project root:

```bash
python scripts/build_intent_v2_dataset.py
python scripts/train_intent_bert_v2.py \
  --train_csv data/en/intent_v2/intent_train_v2.csv \
  --hard_val_csv data/en/intent_v2/intent_hard_val.csv \
  --hard_test_csv data/en/intent_v2/intent_hard_test.csv \
  --output_dir models/classifier_bert_v2 \
  --report_dir reports/en/intent_v2 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5
```

4. Evaluate the new model on the extra hotpot hard-negative set:

```bash
python scripts/evaluate_intent_model.py \
  --model_path models/classifier_bert_v2 \
  --csv data/en/intent_v2/intent_hotpot_relabel_test.csv \
  --output reports/en/intent_v2/v2_on_hotpot_relabel.json
```

5. If the new model improves hotpot hard negatives without hurting the balanced hard test, replace the runtime model:

```bash
cp -r models/classifier_bert_v2/* models/classifier_bert/
```

Or update `configs/config.yaml`:

```yaml
classifier_model_path: "models/classifier_bert_v2/"
```

## Reporting Guidance

Do not present the old `F1=1.000` as a broad real-world claim.

Use this wording:

> The original classifier reached near-perfect performance on the controlled synthetic validation set. After discovering a routing-label mismatch in multi-hop food-health queries, we added a balanced hard-set and evaluated generalization separately.

For the slide:

- Keep synthetic score as "in-distribution sanity check".
- Add hard-set score after retraining.
- Do not show `Intent Accuracy = 47.9%` from the hotpot chart as the main classifier score, because that dataset used a different label interpretation.
