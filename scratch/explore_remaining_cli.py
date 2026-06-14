import json
import os
from collections import Counter

food_path = 'data/en/food_bio.jsonl'
if not os.path.exists(food_path):
    food_path = '../../data/en/food_bio.jsonl'

food_rows = []
with open(food_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            food_rows.append(json.loads(line))

unique_foods = set()
food_list = []

for r in food_rows:
    tokens = r["tokens"]
    labels = r["labels"]
    current_food = []
    for t, l in zip(tokens, labels):
        if l == "B-FOOD":
            if current_food:
                food_str = " ".join(current_food).lower()
                unique_foods.add(tuple(current_food))
                food_list.append(food_str)
            current_food = [t]
        elif l == "I-FOOD":
            if current_food:
                current_food.append(t)
        else:
            if current_food:
                food_str = " ".join(current_food).lower()
                unique_foods.add(tuple(current_food))
                food_list.append(food_str)
                current_food = []
    if current_food:
        food_str = " ".join(current_food).lower()
        unique_foods.add(tuple(current_food))
        food_list.append(food_str)

print("TOP_20_FOODS_FOODBASE:")
for idx, (f, c) in enumerate(Counter(food_list).most_common(20)):
    print(f"{idx+1}. {f} ({c})")

# Food length stats
len_counts = Counter([len(food) for food in unique_foods])
print("\nFOOD_TOKEN_LENGTH_DISTRIBUTION:")
for length, count in sorted(len_counts.items()):
    print(f"{length} tokens: {count} foods")
