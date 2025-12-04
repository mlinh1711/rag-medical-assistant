from typing import List, Dict, Any, Tuple

from backend.core.retrieval import retrieve
from backend.core.llm_providers import call_llm_with_context
from backend.core.multi_query import multi_query_retrieve


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


def run_rag(
    question: str,
    top_k: int = 5,
    score_threshold: float | None = 0.27,
):
    """
    Pipeline RAG chính:
    - Retrieve từ vector store
    - Build context
    - Gọi LLM qua call_llm_with_context
    - Trả answer + sources
    """

    docs = retrieve(query=question, k=top_k, score_threshold=score_threshold)

    if not docs:
        answer = (
            "Sorry, I couldn't find enough relevant information in the documents to "
            "answer this question accurately. You should consult a doctor or "
            "healthcare professional."
        )
        return answer, []

    context_text, sources = build_context_from_docs(docs)
    answer = call_llm_with_context(question, context_text)

    return answer, sources


def run_rag_multi_query(
    question: str,
    variations: List[str],
    top_k: int = 5,
    score_threshold: float | None = 0.27,
):
    """
    Phiên bản RAG dùng multi query:
    - Nhận câu hỏi gốc và các biến thể (variations)
    - Gọi multi_query_retrieve để lấy nhiều bộ docs
    - Build context chung và gọi LLM
    """
    docs = multi_query_retrieve(
        query=question,
        variations=variations,
        k=top_k,
        score_threshold=score_threshold,
    )

    if not docs:
        answer = (
            "Sorry, I couldn't find enough relevant information in the documents to "
            "answer this question accurately. You should consult a doctor or "
            "healthcare professional."
        )
        return answer, []

    context_text, sources = build_context_from_docs(docs)
    answer = call_llm_with_context(question, context_text)

    return answer, sources
