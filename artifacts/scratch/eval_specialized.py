import json
import os
import sys
import torch
import numpy as np
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def evaluate_specialized():
    bc5cdr_path = "data/en/bc5cdr_bio.jsonl"
    food_path = "data/en/food_bio.jsonl"
    
    # Load data
    rows = []
    if os.path.exists(bc5cdr_path):
        with open(bc5cdr_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if os.path.exists(food_path):
        with open(food_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    
    # Replicate exact split
    import random
    random.seed(42)
    random.shuffle(rows)
    test_idx = int(len(rows) * 0.9)
    test_rows = rows[test_idx:]
    
    print(f"Test split size: {len(test_rows)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqeval = evaluate.load("seqeval")
    
    # Load tokenizers
    disease_tok = AutoTokenizer.from_pretrained("alvaroalon2/biobert_diseases_ner")
    food_tok = AutoTokenizer.from_pretrained("Dizex/FoodBaseBERT-NER")
    nut_tok = AutoTokenizer.from_pretrained("sgarbi/bert-fda-nutrition-ner")
    
    # Load models
    disease_mod = AutoModelForTokenClassification.from_pretrained("alvaroalon2/biobert_diseases_ner").to(device).eval()
    food_mod = AutoModelForTokenClassification.from_pretrained("Dizex/FoodBaseBERT-NER").to(device).eval()
    nut_mod = AutoModelForTokenClassification.from_pretrained("sgarbi/bert-fda-nutrition-ner").to(device).eval()
    
    dis_id2label = disease_mod.config.id2label
    food_id2label = food_mod.config.id2label
    nut_id2label = nut_mod.config.id2label
    
    true_labels = []
    merged_baseline_preds = []
    
    print("Running specialized ensemble evaluation...")
    for idx, row in enumerate(test_rows):
        tokens = row["tokens"]
        labels = row["labels"]
        true_labels.append(labels)
        
        # 1. Disease
        inputs_dis = disease_tok(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            preds_dis = torch.argmax(disease_mod(**{k: v.to(device) for k, v in inputs_dis.items()}).logits, dim=-1)[0].tolist()
        
        # 2. Food
        inputs_food = food_tok(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            preds_food = torch.argmax(food_mod(**{k: v.to(device) for k, v in inputs_food.items()}).logits, dim=-1)[0].tolist()
            
        # 3. Nutrient
        inputs_nut = nut_tok(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            preds_nut = torch.argmax(nut_mod(**{k: v.to(device) for k, v in inputs_nut.items()}).logits, dim=-1)[0].tolist()
            
        # Align word ids for each and merge
        w_dis = inputs_dis.word_ids(batch_index=0)
        w_food = inputs_food.word_ids(batch_index=0)
        w_nut = inputs_nut.word_ids(batch_index=0)
        
        # Helper to align subword tags to word level, handling digit '0' as 'O'
        def align_tags(w_ids, predictions, id2label, class_name):
            prev_wid = None
            aligned = ["O"] * len(labels)
            for i, wid in enumerate(w_ids):
                if wid is None or wid >= len(labels):
                    continue
                if wid != prev_wid:
                    raw_tag = id2label[predictions[i]]
                    # Map to class_name, handling '0' as 'O'
                    if raw_tag not in ["O", "0"]:
                        bio = raw_tag[0]
                        aligned[wid] = f"{bio}-{class_name}"
                prev_wid = wid
            return aligned
            
        aligned_dis = align_tags(w_dis, preds_dis, dis_id2label, "DISEASE")
        aligned_food = align_tags(w_food, preds_food, food_id2label, "FOOD")
        aligned_nut = align_tags(w_nut, preds_nut, nut_id2label, "NUTRIENT")
        
        # Merge predictions
        row_merged = []
        for d_p, f_p, n_p in zip(aligned_dis, aligned_food, aligned_nut):
            if d_p != "O":
                row_merged.append(d_p)
            elif f_p != "O":
                row_merged.append(f_p)
            elif n_p != "O":
                row_merged.append(n_p)
            else:
                row_merged.append("O")
        merged_baseline_preds.append(row_merged)
        
    results = seqeval.compute(predictions=merged_baseline_preds, references=true_labels)
    print("\nEnsemble Results:")
    for key in ["DISEASE", "FOOD", "NUTRIENT"]:
        print(f"{key}: {results[key]}")
    print(f"Overall: P={results['overall_precision']:.4f}, R={results['overall_recall']:.4f}, F1={results['overall_f1']:.4f}")

if __name__ == "__main__":
    evaluate_specialized()
