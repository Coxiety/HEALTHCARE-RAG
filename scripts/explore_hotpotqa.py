import sys
import os

# Set output encoding to UTF-8 to handle unicode printing in terminal
sys.stdout.reconfigure(encoding='utf-8')

try:
    from datasets import load_dataset
except ImportError:
    print("[HF Dataset] Installing 'datasets' and 'huggingface_hub' libraries...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub", "fsspec>=2023.5.0"])
    from datasets import load_dataset

print("Downloading HotpotQA validation set (distractor configuration) from Hugging Face...")
try:
    # Use split='validation' (7,405 cases) instead of train to download quickly
    dataset = load_dataset("KOJKO/hotpot_qa", "distractor", split="validation")
    print(f"\nSuccessfully downloaded! Total cases in validation set: {len(dataset)}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    sys.exit(1)

# Health and nutrition keywords to filter the dataset
keywords = ["food", "nutrition", "diet", "disease", "health", "vitamin", "symptom", "medicine", "doctor", "clinical", "cancer", "diabetes", "calcium"]

filtered_cases = []
for item in dataset:
    q = item["question"].lower()
    if any(kw in q for kw in keywords):
        filtered_cases.append(item)

print(f"\nFiltered cases containing health/nutrition keywords: {len(filtered_cases)}")

print("\n=== SAMPLE GENERAL QUESTIONS IN HOTPOTQA ===")
for i in range(min(5, len(dataset))):
    item = dataset[i]
    print(f"{i+1}. Q: {item['question']}")
    print(f"   A: {item['answer']}\n")

if filtered_cases:
    print("=== SAMPLE HEALTH/NUTRITION RELATED QUESTIONS IN HOTPOTQA ===")
    for i in range(min(5, len(filtered_cases))):
        item = filtered_cases[i]
        print(f"{i+1}. Q: {item['question']}")
        print(f"   A: {item['answer']}\n")
else:
    print("No health/nutrition-related questions found in the validation subset.")
