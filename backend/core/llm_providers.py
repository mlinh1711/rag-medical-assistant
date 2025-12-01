import os
import requests
from typing import List, Dict


# Cấu hình Ollama từ biến môi trường
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def _call_ollama_generate(prompt: str, timeout: int = 120) -> str:
    """
    Hàm tiện ích gọi Ollama /api/generate với prompt thô.
    Các hàm khác (RAG, rewrite, v.v.) sẽ build prompt rồi gọi hàm này.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=timeout)

    try:
        resp.raise_for_status()
    except requests.HTTPError:
        print("Ollama error status:", resp.status_code)
        print("Ollama response:", resp.text)
        raise

    data = resp.json()
    answer = data.get("response", "").strip()
    if not answer:
        answer = (
            "Sorry, the answer can not be generated from the LLM provider right now. "
            "Please try again later or consult a healthcare professional."
        )
    return answer


def call_llm_with_context(question: str, context: str) -> str:
    """
    Gọi LLM để trả lời câu hỏi y khoa dựa trên context (RAG).
    Dùng chung cho RAG bình thường và history-aware RAG.
    """
    prompt = (
        "You are a medical information assistant.\n"
        "Answer ONLY based on the provided context below.\n"
        "If the context does not contain enough information, say clearly that you do not know.\n"
        "Your answer is for general information only and is NOT a substitute for professional medical advice.\n\n"
        "Answering style:\n"
        "- Write for a non-medical person (patients, caregivers).\n"
        "- Start with a direct, practical answer in 1–2 sentences.\n"
        "- Overall length: at most 3–4 sentences or about 80 words.\n"
        "- Do NOT copy long passages or tables from the context. Summarise in your own words.\n"
        "- If doses or numbers are relevant, mention the key ones clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Now provide the answer."
    )

    return _call_ollama_generate(prompt)


def call_llm_for_rewrite(history: List[Dict[str, str]], question: str) -> str:
    """
    Gọi LLM để rewrite câu hỏi mơ hồ thành câu hỏi đầy đủ, tự chứa.

    history: list các message {"role": "user"/"assistant", "content": "..."}
    question: câu hỏi mới nhất (có thể mơ hồ, ví dụ: "What about the dosage?")
    """
    # Chuyển history thành text để LLM nắm ngữ cảnh hội thoại
    history_lines: List[str] = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)

    prompt = (
        "You are helping to rewrite ambiguous medical questions into clear, "
        "standalone questions.\n\n"
        "Conversation history:\n"
        f"{history_text}\n\n"
        f"New user question: {question}\n\n"
        "Task:\n"
        "- Rewrite the new user question into a standalone, explicit medical question.\n"
        "- Include key details from the history if needed (e.g. drug name, patient, route, etc.).\n"
        "- The rewritten question must make sense on its own without the history.\n"
        "- Answer ONLY with the rewritten question, nothing else.\n"
    )

    rewritten = _call_ollama_generate(prompt)
    # Làm sạch kết quả (tránh trường hợp LLM thêm ngoặc kép hoặc tiền tố)
    return rewritten.strip().strip('"').strip()
