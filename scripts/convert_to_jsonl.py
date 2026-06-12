import json
import os
import sys

file_path = r"d:\Data\File for Google Drive real\Project\Nutrition_RAG\HEALTHCARE-RAG\data\en\eval_hotpot_nutrition.jsonl"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    sys.exit(1)

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read().strip()

# Strip markdown backticks if present
if content.startswith("```"):
    # Strip opening backticks (e.g. ```json)
    first_newline = content.find("\n")
    if first_newline != -1:
        content = content[first_newline:].strip()
if content.endswith("```"):
    content = content[:-3].strip()

try:
    # Parse the standard JSON array
    records = json.loads(content)
except json.JSONDecodeError as e:
    print(f"JSON Decode Error: {e}")
    # Print a snippet of where it failed
    snippet_start = max(0, e.pos - 50)
    snippet_end = min(len(content), e.pos + 50)
    print(f"Error snippet: ... {content[snippet_start:snippet_end]} ...")
    sys.exit(1)

if not isinstance(records, list):
    print("Error: Expected a JSON list of objects.")
    sys.exit(1)

# Write back as JSON Lines (one JSON object per line)
with open(file_path, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Successfully converted {len(records)} records to JSONL format!")
