from typing import List, Optional

from backend.core.retrieval import get_vectorstore


def multi_query_retrieve(
    query: str,
    variations: List[str],
    k: int = 5,
    score_threshold: Optional[float] = None,
):
    """
    Multi-query retrieval:
    - query: câu hỏi chính
    - variations: list câu hỏi phụ (paraphrase / mở rộng)
    - Trả về list Document đã loại trùng theo (source, page)
    """
    db = get_vectorstore()

    # Tạo retriever giống như trong retrieval.py
    if score_threshold is None:
        retriever = db.as_retriever(
            search_kwargs={"k": k},
        )
    else:
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold,
            },
        )

    all_docs = []

    # Query chính
    main_docs = retriever.invoke(query)
    all_docs.extend(main_docs)

    # Query phụ
    for q in variations:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    # Deduplicate theo (source, page)
    unique = {}
    for d in all_docs:
        metadata = d.metadata or {}
        source = metadata.get("source") or metadata.get("doc_id")
        page = metadata.get("page") or metadata.get("page_label")
        key = (source, page)
        if key not in unique:
            unique[key] = d

    return list(unique.values())
