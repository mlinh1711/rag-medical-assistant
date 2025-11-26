from typing import List, Dict, Any

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.core.config import settings
from backend.core.rag_pipeline import run_rag

app = FastAPI(title="RAG Medical Assistant")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    start = time.time()

    try:
        answer, sources = run_rag(
            question=payload.question,
            top_k=payload.top_k,
        )
    except Exception as e:
        print(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail="Internal error in RAG pipeline")

    latency = round((time.time() - start) * 1000, 2)

    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": latency,
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "data_path": settings.data_path}


@app.get("/")
def root():
    return {"message": "RAG Medical Assistant is running!"}
