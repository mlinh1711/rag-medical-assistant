from typing import List, Dict, Any, Tuple, Optional

from backend.core.rag_pipeline import build_context_from_docs
from backend.core.retrieval import retrieve
from backend.core.llm_providers import call_llm_with_context, call_llm_for_rewrite


def rewrite_question(history: List[Dict[str, str]], question: str) -> str:
    return call_llm_for_rewrite(history, question)


def run_rag_with_history(
    history: List[Dict[str, str]],
    question: str,
    top_k: int = 5,
    score_threshold: Optional[float] = 0.27,
) -> Tuple[str, List[Dict[str, Any]], str]:
    
    rewritten = rewrite_question(history, question)

    docs = retrieve(query=rewritten, k=top_k, score_threshold=score_threshold)

    if not docs:
        answer = (
            "Sorry, I couldn't find enough relevant information in the documents to "
            "answer this question accurately. You should consult a doctor or "
            "healthcare professional."
        )
        return answer, [], rewritten

    context_text, sources = build_context_from_docs(docs)

    answer = call_llm_with_context(rewritten, context_text)

    return answer, sources, rewritten
