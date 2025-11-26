import os
import sys

sys.path.append("/kaggle/working/rag-medical-assistant")

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


DB_PATH = "db/chroma_db"

# Dùng chung embedding với retrieval
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)


def get_vectorstore():
    if not os.path.exists(DB_PATH) or len(os.listdir(DB_PATH)) == 0:
        raise RuntimeError(
            f"Vector store not found in '{DB_PATH}'. "
            "Run ingest/ingest_final.py first."
        )

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return db


def multi_query_retrieve(query, variations, k=5):
    """
    Multi-query retrieval:
    - query: câu hỏi chính
    - variations: list câu hỏi phụ (paraphrase / mở rộng)
    - Trả về list Document đã loại trùng theo (source, page)
    """
    db = get_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": k})

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
        source = d.metadata.get("source")
        page = d.metadata.get("page")
        key = (source, page)
        if key not in unique:
            unique[key] = d

    return list(unique.values())
