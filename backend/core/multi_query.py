from typing import List, Optional

from backend.core.retrieval import get_vectorstore


def multi_query_retrieve(
    query: str,
    variations: List[str],
    k: int = 5,
    score_threshold: Optional[float] = None,
):
    db = get_vectorstore()

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

    main_docs = retriever.invoke(query)
    all_docs.extend(main_docs)

    for q in variations:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    unique = {}
    for d in all_docs:
        metadata = d.metadata or {}
        source = metadata.get("source") or metadata.get("doc_id")
        page = metadata.get("page") or metadata.get("page_label")
        key = (source, page)
        if key not in unique:
            unique[key] = d

    return list(unique.values())
