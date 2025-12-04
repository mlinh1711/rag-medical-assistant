import os
import requests
from typing import Literal

from backend.core.config import settings

# Đọc provider từ env: "deepseek" hoặc "ollama"
LLM_PROVIDER: Literal["deepseek", "ollama"] = os.getenv(
    "LLM_PROVIDER", "deepseek"
).lower()  # type: ignore

# Cấu hình DeepSeek từ .env (qua pydantic Settings)
DEEPSEEK_API_KEY = settings.deepseek_api_key
DEEPSEEK_MODEL = settings.deepseek_model  # ví dụ: "deepseek-chat"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Optional: Ollama nếu bạn muốn dùng song song
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def build_prompt(question: str, context: str) -> str:
    """
    Prompt chung cho mọi LLM: siết chặt việc bám vào Context,
    tránh trả lời linh tinh ngoài tài liệu.
    """
    return (
        "You are a careful medical assistant for a Retrieval Augmented Generation system.\n"
        "Follow these rules strictly:\n"
        "1. Use only the medical information inside the Context below. Do not use outside knowledge.\n"
        "2. If the Context does not clearly contain the answer, reply exactly: "
        "\"I do not know. Please consult a doctor for more information.\"\n"
        "3. If the Question and the Context talk about different diseases, medicines, or topics, "
        "you must also reply with the same sentence in rule 2.\n"
        "4. Keep the answer short and clear, about 2 to 5 sentences.\n"
        "5. Answer in the same language as the Question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def call_deepseek_with_context(question: str, context: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is empty. Check your .env file.")

    prompt = build_prompt(question, context)

    headers = {
        # Quan trọng: phải có "Bearer "
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # In nội dung lỗi thật để debug
        print("DeepSeek error:", resp.text)
        raise e

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_ollama_with_context(question: str, context: str) -> str:
    """
    Hàm gọi Ollama, nếu bạn vẫn muốn dùng local model.
    """
    prompt = build_prompt(question, context)

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Tùy theo format Ollama, chỉnh lại cho đúng
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"].strip()

    # Fallback
    return str(data)


def call_llm_with_context(question: str, context: str) -> str:
    """
    Hàm chung, RAG pipeline chỉ gọi hàm này.
    """
    provider = LLM_PROVIDER

    if provider == "deepseek":
        return call_deepseek_with_context(question, context)
    elif provider == "ollama":
        return call_ollama_with_context(question, context)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
