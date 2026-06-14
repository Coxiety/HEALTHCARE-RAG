import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer
from src.en.ner import BertCRFForTokenClassification

def evaluate_ner():
    model_path = "models/ner_bert"
    bc5cdr_path = "data/en/bc5cdr_bio.jsonl"
    food_path = "data/en/food_bio.jsonl"
    
    LABEL_LIST = ["O", "B-NUTRIENT", "I-NUTRIENT", "B-DISEASE", "I-DISEASE", "B-FOOD", "I-FOOD"]
    label2id = {l: i for i, l in enumerate(LABEL_LIST)}
    id2label = {i: l for l, i in label2id.items()}
    
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
                    
    # Split 80/10/10. Let's seed and shuffle to replicate the split
    import random
    random.seed(42)
    random.shuffle(rows)
    
    # Test split is the last 10%
    test_idx = int(len(rows) * 0.9)
    test_rows = rows[test_idx:]
    
    print(f"Total rows: {len(rows)} | Test split size: {len(test_rows)}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertCRFForTokenClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    seqeval = evaluate.load("seqeval")
    
    true_labels = []
    pred_labels = []
    
    for idx, row in enumerate(test_rows):
        tokens = row["tokens"]
        labels = row["labels"]
        
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        model_inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**model_inputs)
            
        logits = outputs.logits
        mask = (model_inputs["attention_mask"] == 1)
        
        decoded_paths = model.crf.decode(logits, mask)
        path = decoded_paths[0]
        
        # Align labels
        word_ids = inputs.word_ids(batch_index=0)
        prev_word_id = None
        
        row_true = []
        row_pred = []
        
        # We need to map subwords back to word level tags
        # Or evaluate at the subword token level using the -100 mask
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                # First subword of word
                if word_id < len(labels):
                    row_true.append(labels[word_id])
                    # Get prediction for this subword index
                    pred_tag = id2label[path[i]]
                    row_pred.append(pred_tag)
            prev_word_id = word_id
            
        true_labels.append(row_true)
        pred_labels.append(row_pred)
        
    result = seqeval.compute(predictions=pred_labels, references=true_labels)
    print("\nSeqeval Classification Report:")
    for key, val in result.items():
        if isinstance(val, dict):
            print(f"{key}: {val}")
        else:
            print(f"{key}: {val:.4f}")

if __name__ == "__main__":
    evaluate_ner()
