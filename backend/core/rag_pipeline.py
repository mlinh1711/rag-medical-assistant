from typing import List, Dict, Any, Tuple
import os
import requests

from backend.core.retrieval import retrieve


# Cấu hình Ollama từ biến môi trường
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def build_context_from_docs(docs) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Nhận list Document và:
    - Ghép thành context text cho LLM
    - Trả thêm list sources cho frontend
    """
    context_blocks: List[str] = []
    sources: List[Dict[str, Any]] = []

    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = (
            metadata.get("source")
            or metadata.get("doc_id")
            or "unknown_source"
        )
        page = (
            metadata.get("page")
            or metadata.get("page_label")
            or "N/A"
        )

        block = f"[{idx}] Source: {source} - page {page}\n{doc.page_content}"
        context_blocks.append(block)

        snippet = doc.page_content[:400]

        sources.append(
            {
                "id": idx,
                "source": source,
                "page": page,
                "snippet": snippet,
            }
        )

    context_text = "\n\n".join(context_blocks)
    return context_text, sources


def call_ollama_with_context(question: str, context: str) -> str:
    """
    Gọi Ollama qua /api/generate với prompt đầy đủ.
    Dùng kiểu generate đơn giản, ổn định hơn /api/chat.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"

    prompt = (
        "You are a medical information assistant. "
        "Answer ONLY based on the provided context. "
        "If the context does not contain enough information, say you do not know. "
        "Always remind that your answer is general information and not a substitute "
        "for professional medical advice.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and concisely."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=120)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("Ollama error status:", resp.status_code)
        print("Ollama response:", resp.text)
        raise

    data = resp.json()

    # Với stream=False, Ollama trả về field "response"
    answer = data.get("response", "").strip()
    if not answer:
        answer = "Sorry, the answer can not be generated from Ollama model right now."

    return answer


def run_rag(
    question: str,
    top_k: int = 5,
    score_threshold: float | None = 0.27,
):
    """
    Pipeline RAG chính:
    - Retrieve từ Chroma
    - Build context
    - Gọi Ollama
    - Trả answer + sources
    """

    # Dùng score_threshold để lọc bớt đoạn không liên quan
    docs = retrieve(query=question, k=top_k, score_threshold=score_threshold)

    if not docs:
        answer = (
            "Sorry, I couldn't find enough relevant information in the documents to answer this question accurately."
            "You should consult a doctor or healthcare professional."
        )
        return answer, []

    context_text, sources = build_context_from_docs(docs)
    answer = call_ollama_with_context(question, context_text)

    return answer, sources
