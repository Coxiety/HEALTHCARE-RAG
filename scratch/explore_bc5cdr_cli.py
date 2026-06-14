import json
import os
from collections import Counter

data_path = 'data/en/bc5cdr_bio.jsonl'

if not os.path.exists(data_path):
    data_path = '../../data/en/bc5cdr_bio.jsonl'

rows = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

def extract_entities(rows, ent_type):
    entities = []
    for row in rows:
        tokens = row['tokens']
        labels = row['labels']
        current_ent = []
        for t, l in zip(tokens, labels):
            if l == f"B-{ent_type}":
                if current_ent:
                    entities.append(" ".join(current_ent).lower())
                current_ent = [t]
            elif l == f"I-{ent_type}":
                if current_ent:
                    current_ent.append(t)
            else:
                if current_ent:
                    entities.append(" ".join(current_ent).lower())
                    current_ent = []
        if current_ent:
            entities.append(" ".join(current_ent).lower())
    return entities

diseases = extract_entities(rows, 'DISEASE')
nutrients = extract_entities(rows, 'NUTRIENT')

print("TOP_20_DISEASES:")
for idx, (d, c) in enumerate(Counter(diseases).most_common(20)):
    print(f"{idx+1}. {d} ({c})")

print("\nTOP_20_NUTRIENTS:")
for idx, (n, c) in enumerate(Counter(nutrients).most_common(20)):
    print(f"{idx+1}. {n} ({c})")
