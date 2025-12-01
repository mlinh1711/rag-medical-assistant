from typing import List, Dict, Any, Tuple, Optional

from backend.core.rag_pipeline import build_context_from_docs
from backend.core.retrieval import retrieve
from backend.core.llm_providers import call_llm_with_context, call_llm_for_rewrite


def rewrite_question(history: List[Dict[str, str]], question: str) -> str:
    """
    history: list các message {"role": "user"/"assistant", "content": "..."}
    Gọi LLM để rewrite question thành câu đầy đủ, không mơ hồ.
    """
    # Ủy quyền cho hàm call_llm_for_rewrite trong llm_providers
    return call_llm_for_rewrite(history, question)


def run_rag_with_history(
    history: List[Dict[str, str]],
    question: str,
    top_k: int = 5,
    score_threshold: Optional[float] = 0.27,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    History-aware RAG pipeline:
    - Rewrite câu hỏi dựa trên lịch sử hội thoại.
    - Retrieve theo câu hỏi đã rewrite.
    - Build context và gọi LLM để trả lời.
    - Trả về: answer, sources, rewritten_question.
    """
    # Rewrite câu hỏi mơ hồ dựa trên history
    rewritten = rewrite_question(history, question)

    # Retrieve dựa trên câu hỏi đã được rewrite
    docs = retrieve(query=rewritten, k=top_k, score_threshold=score_threshold)

    if not docs:
        answer = (
            "Sorry, I couldn't find enough relevant information in the documents to "
            "answer this question accurately. You should consult a doctor or "
            "healthcare professional."
        )
        return answer, [], rewritten

    # Xây dựng context và sources giống RAG pipeline thường
    context_text, sources = build_context_from_docs(docs)

    # Gọi LLM để trả lời dựa trên context
    answer = call_llm_with_context(rewritten, context_text)

    return answer, sources, rewritten
