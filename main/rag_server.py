from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import uvicorn

from src.en.pipeline import ENPipeline

_en_pipeline: ENPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _en_pipeline
    print("[RAG Server] Starting EN pipeline...")
    _en_pipeline = ENPipeline()
    print("[RAG Server] EN pipeline ready.")
    yield
    print("[RAG Server] Shutting down.")


app = FastAPI(title="HealthCare RAG", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


@app.post("/ask")
async def ask(req: AskRequest):
    if _en_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    history_dicts = [h.model_dump() for h in req.history]
    result = await run_in_threadpool(_en_pipeline.answer, req.message, history_dicts)
    ner_entities = result.get("entities", {})
    
    # Debug print to terminal
    print(f"\n[DEBUG RAG Server] ========== REQUEST PROCESSED ==========")
    print(f"User Message:       '{req.message}'")
    print(f"Classified Intent:  {result.get('intent', '')}")
    print(f"Extracted Entities (BERT NER):")
    print(f"  - FOOD:      {ner_entities.get('FOOD', [])}")
    print(f"  - DISEASE:   {ner_entities.get('DISEASE', [])}")
    print(f"  - NUTRIENT:  {ner_entities.get('NUTRIENT', [])}")
    print(f"  - SYMPTOM:   {ner_entities.get('SYMPTOM', [])}")
    
    # If NUTRITION_LOOKUP but NER found no food, mention the spaCy fallback
    if result.get("intent") in ("NUTRITION_LOOKUP", "BOTH") and not ner_entities.get("FOOD"):
        from src.en.pipeline import _extract_foods_from_query
        fallback_foods = _extract_foods_from_query(req.message)
        print(f"  -> [Fallback spaCy Noun Chunks used for DB lookup]: {fallback_foods}")
        
    print(f"Used LLM:           {result.get('used_llm', True)}")
    print(f"Sources:            {sorted(set(result.get('sources', [])))}")
    print(f"==========================================================\n")
    
    return {
        "answer":   result.get("answer", ""),
        "intent":   result.get("intent", ""),
        "entities": {
            "foods":     ner_entities.get("FOOD",     []),
            "diseases":  ner_entities.get("DISEASE",  []),
            "nutrients": ner_entities.get("NUTRIENT", []),
            "symptoms":  ner_entities.get("SYMPTOM",  []),
        },
        "sources": sorted(set(result.get("sources", []))),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": _en_pipeline is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
