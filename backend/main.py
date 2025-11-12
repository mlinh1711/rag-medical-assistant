from fastapi import FastAPI
from backend.core.config import settings

app = FastAPI(title="RAG Medical Assistant")

@app.get("/health")
def health_check():
    return {"status": "ok", "data_path": settings.data_path}

@app.get("/")
def root():
    return {"message": "RAG Medical Assistant is running!"}