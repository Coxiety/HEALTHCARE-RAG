# HealthCare-RAG

English-language's Retrieval-Augmented Generation system for nutrition and health Q&A, featuring intelligent query rewriting, hybrid search, rerankers, and dual LLM support (local Ollama or cloud-based Google Gemini).

---

## Architecture

```
                                      [User Query]
                                           │
                                           ▼
                                [Query Rewriter (Gemini/Ollama)]
                                           │
                                           ▼
                                    [Preprocessor]
                                           │
                                           ▼
                                    [Classifier BERT] (3-class: NUTRITION_LOOKUP, HEALTH_ADVICE, BOTH)
                                           │
                                           ▼
                                     [NER BioBERT] (FOOD, DISEASE, NUTRIENT, SYMPTOM)
                                           │
                                           ├─► NUTRITION ──► USDA SQLite (13,661 foods)
                                           │
                                           └─► HEALTH ─────► Retriever (BM25 + Dense Hybrid RRF)
                                                                 │
                                                                 ▼
                                                             [Reranker] (cross-encoder/ms-marco-MiniLM)
                                                                 │
                                                                 ▼
                                                      [Generator (Ollama/Gemini)]
                                                                 │
                                                                 ▼
                                                              [Answer]
```

---

## Key Features

1. **Context-Aware Query Rewriting**: Resolves pronouns and contextual references in chat history (e.g. converting "Is it healthy?" after a discussion about "apple" into "Is apple healthy for human consumption?"). Supports Google Gemini (with the new `google-genai` SDK) and local Ollama, with automated failover from Gemini to Ollama.
2. **Hybrid Retrieval**: Combines sparse BM25 retrieval with dense vector search (ChromaDB using fine-tuned `all-MiniLM-L6-v2` embeddings).
3. **Re-Ranking**: Filters and re-orders the top retrieved passages using `ms-marco-MiniLM-L-6-v2` cross-encoder.
4. **Structured Knowledge Base**: Fast local SQLite lookups for USDA nutrition facts.
5. **Flexible LLM Engine**: Can be configured to run entirely locally with Ollama or leverage Google Gemini APIs for superior performance.

---

## Configuration (`configs/config.yaml`)

Configure your model paths, database locations, and LLM backends in `configs/config.yaml`:

```yaml
# --- Paths ---
sqlite_path: "data/usda_food.db"
chroma_persist_dir: "data/chroma_db"
chroma_collection: "nfcorpus"

# --- Models ---
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
ner_model_path: "models/ner_bert/"
classifier_model_path: "models/classifier_bert/"

# --- LLM Generator ---
llm_model: "llama3.1:8b"        # local Ollama model

# --- Query Rewriter ---
rewriter_model: "llama3.1:8b"

# --- Retrieval ---
top_k: 5
similarity_threshold: 0.5
```

---

## Setup & Installation

**Requirements:** Python 3.10+, JDK 21+, Maven, [Ollama](https://ollama.com) (if running locally).

### 1. Auto Setup (Windows PowerShell)
Run the setup script from the root directory:
```powershell
.\setup_project.ps1 -UpgradePip
```
This script will initialize the virtual environment `.venv` and install all necessary dependencies listed in `requirements.txt`.

### 2. Manual Setup
Alternatively, set up manually:
```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull local LLM model (if using Ollama)
ollama pull llama3.1:8b
```

### 3. Downloads
Download the required model weights and database from the shared folder and extract them to their respective locations:
*   [Download Link (Google Drive)](https://drive.google.com/drive/folders/1N42YIeaPXBH9S9hyRiGw42SHcyzTvn8L?usp=sharing)

| File / Folder | Target Path |
|---|---|
| `ner_bert.zip` | `models/ner_bert/` (Extract here) |
| `classifier_bert.zip` | `models/classifier_bert/` (Extract here) |
| `embedding_domain.zip` | `models/embedding_domain/` (Extract here) |
| `usda_food.db` | `data/usda_food.db` |

---

## Running the Application

To run the full system with a web user interface, start the backend and frontend services:

```bash
# 1. Start Ollama (if using local LLM)
ollama serve

# 2. Start RAG Backend Server (FastAPI)
python main/rag_server.py

# 3. Start Frontend App (Spring Boot)
cd chatbot
mvn spring-boot:run
```
Access the web UI at `http://localhost:8081`.

## Evaluation and Testing

### 1. Run Automatic RAG Evaluation
Run evaluation on a test dataset (e.g. `eval_200.jsonl`):
```bash
python -m src.evaluation.rag_evaluator --input data/en/eval_200.jsonl --output-dir reports/en/rag_eval
```
The evaluator outputs `summary.json` and `cases.csv` containing performance and accuracy metrics.

---

## Project Structure

```
├── src/
│   ├── en/               # English RAG pipeline (processor, classifier, NER, retriever, etc.)
│   ├── data_pipeline/    # Data indexing and chunking
│   ├── database/         # SQLite manager and Vector Store (Chroma)
│   └── generation/       # LLM Generator (Ollama wrappers)
├── main/
│   ├── build_usda_db.py  # Builds the USDA SQLite database from CSVs
│   └── rag_server.py     # FastAPI Server
├── notebooks/en/         # Jupyter Notebooks for training model parts
├── data/                 # Raw/processed data (Chroma DB files, USDA, etc.)
├── models/               # Model weights directory (NER, Classifier, etc.)
├── reports/              # Benchmark and evaluation reports
├── chatbot/              # Spring Boot UI (port 8081)
└── configs/config.yaml   # Main configuration file
```
