import sys
import os
import json
import yaml

# Thêm root dir vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.database.vector_store import VectorStore

def main():
    try:
        cfg = yaml.safe_load(open("configs/config.yaml"))
    except Exception as e:
        print("Cannot load config.yaml:", e)
        return

    vs = VectorStore(cfg["chroma_persist_dir"], cfg["chroma_collection"], cfg["embedding_model"])
    
    print(f"Clearing existing ChromaDB collection '{vs.collection_name}'...")
    try:
        vs.clear()
    except Exception as e:
        print(f"Warning on clear: {e}")
    
    filepath = "data/en/corpus.jsonl"
    print(f"Reading documents from {filepath}...")
    
    docs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            obj = json.loads(line)
            docs.append({
                "id": obj["id"],
                "text": obj["text"],
                "source": obj.get("source", "unknown")
            })
            
    print(f"Loaded {len(docs)} documents. Re-indexing into ChromaDB...")
    
    BATCH_SIZE = 5000
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        vs.add(batch)
        print(f"  Indexed {min(i + BATCH_SIZE, len(docs))} / {len(docs)}")
        
    print(f"ChromaDB total: {vs.count()}")
    print("Rebuild completed successfully!")

if __name__ == "__main__":
    main()
