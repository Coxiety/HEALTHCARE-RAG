# HealthCare-RAG

English-language Retrieval-Augmented Generation system for nutrition and health Q&A.

---

## Architecture

```
[User Query]
      │
      ▼
[Preprocessor]        NLTK / spaCy
      │
      ▼
[Classifier BERT]     3-class: NUTRITION_LOOKUP · HEALTH_ADVICE · BOTH
      │
      ▼
[NER BioBERT]         FOOD · DISEASE · NUTRIENT · SYMPTOM
      │
      ├── NUTRITION ──▶ USDA SQLite (13,661 foods)
      │
      └── HEALTH ────▶ Retriever
                        TF-IDF · BM25 · Dense · Dense fine-tuned · Hybrid RRF
                              │
                              ▼
                        Reranker (cross-encoder/ms-marco-MiniLM)
                              │
                              ▼
[Generator]           Ollama llama3.1:8b
```

---

## Retrieval Eval — NFCorpus 323 test queries

| Method | MRR |
|---|---|
| TF-IDF | 0.38 |
| BM25 | 0.47 |
| Dense (vanilla) | 0.50 |
| Dense fine-tuned | 0.47 |
| Hybrid RRF | 0.50 |
| Hybrid + Reranker | **0.55** |

---

## Models

| Component | Model | Status |
|---|---|---|
| NER | `dmis-lab/biobert-base-cased-v1.2` fine-tuned on BC5CDR | ✅ F1 = 0.893 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` fine-tuned on NFCorpus triplets | ✅ MRR = 0.47 |
| Classifier | `bert-base-uncased` + 3-class head | ✅ F1 = 1.000 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (pretrained) | ✅ |
| LLM | `llama3.1:8b` via Ollama | ✅ |

---

## Datasets

| Dataset | Purpose | Size |
|---|---|---|
| USDA FoodData Central | Nutrition lookup | 13,661 foods |
| NFCorpus (BEIR) | RAG corpus + retrieval eval | 3,633 docs · 323 test queries |
| BC5CDR (`tner/bc5cdr`) | NER training — BIO tags | 16,423 sentences |
| Synthetic intent | Classifier training | 1,500 questions (500 × 3 class) |

---

## Training Notebooks

All notebooks run on **Google Colab GPU**. Workflow: upload data to Drive → open notebook → train → download zip → extract to `models/`.

| Notebook | Input (upload to Drive) | Output |
|---|---|---|
| `notebooks/en/training_ner.ipynb` | `data/en/bc5cdr_bio.jsonl` | `models/ner_bert/` |
| `notebooks/en/training_embedding.ipynb` | `data/en/triplets.jsonl` | `models/embedding_domain/` |
| `notebooks/en/training_classifier.ipynb` | `data/en/intent_data.csv` | `models/classifier_bert/` |

> **Note:** `bc5cdr_bio.jsonl` and `triplets.jsonl` are not tracked by git (too large).
> `intent_data.csv` is tracked — `git pull` is enough for the classifier notebook.

After downloading, extract to the corresponding `models/` subfolder:

```
models/
├── ner_bert/            ← extract ner_bert.zip here
├── embedding_domain/    ← extract embedding_domain.zip here
└── classifier_bert/     ← extract classifier_bert.zip here
```

---

## Project Structure

```
├── src/
│   ├── en/               # EN pipeline
│   ├── data_pipeline/    # data loading, chunking, embedding, synthesis
│   ├── database/         # sqlite_manager, vector_store
│   └── generation/       # Ollama generator
├── main/
│   ├── rag_server.py     # FastAPI server (port 8000)
│   └── build_usda_db.py  # Build USDA SQLite from CSV
├── notebooks/en/
│   ├── training_ner.ipynb
│   ├── training_embedding.ipynb
│   ├── training_classifier.ipynb
│   ├── eval_retrieval.ipynb
│   └── eval_synthetic_intent.ipynb
├── data/en/
│   ├── intent_data.csv   # tracked — 1,500 synthetic intent questions
│   ├── corpus.jsonl      # tracked — 3,633 NFCorpus docs
│   ├── bc5cdr_bio.jsonl  # NOT tracked — share manually for NER training
│   └── triplets.jsonl    # NOT tracked — share manually for embedding training
├── models/               # NOT tracked — extract from training zip
├── reports/en/           # eval charts + ablation results
├── chatbot/              # Spring Boot UI (port 8081)
└── configs/config.yaml
```

---

## Setup

**Requirements:** Python 3.10+, JDK 21+, Maven, [Ollama](https://ollama.com)

```bash
conda create -n nutrition-rag python=3.10
conda activate nutrition-rag
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull llama3.1:8b
```

Build USDA database (requires raw CSV in `FoodData_Central_csv_*/`):

```bash
python main/build_usda_db.py
```

Download and index NFCorpus (~50 MB, runs once):

```bash
python -m src.data_pipeline.load_nfcorpus
```

---

## Running

```bash
# Terminal 1
ollama serve

# Terminal 2
python main/rag_server.py        # FastAPI at http://localhost:8000

# Terminal 3
cd chatbot && mvn spring-boot:run # UI at http://localhost:8081
```

---

## Team

| Member | Responsibility |
|---|---|
| TV1 | EN pipeline · retrieval baselines · NER training · eval |
| TV2 | Triplet mining · embedding fine-tune |
| TV3 | Classifier fine-tune · Spring Boot UI · report |
