import os
from typing import Any, List, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


DB_PATH = "db/chroma_db"

# Embedding model miễn phí, chạy trên CPU
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)


def get_vectorstore() -> Chroma:
    """
    Load a persisted Chroma vector store from disk.
    """
    # Kiểm tra thư mục DB có tồn tại và không rỗng
    if not os.path.exists(DB_PATH) or len(os.listdir(DB_PATH)) == 0:
        raise RuntimeError(
            f"Vector store does not exist at '{DB_PATH}'. "
            "Run ingest/ingest_final.py first."
        )

    # Khởi tạo Chroma với embedding model đã cấu hình
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return db


def get_retriever(k: int = 5, score_threshold: float | None = None):
    """
    Return a LangChain retriever object.

    If score_threshold is None: standard top-k search.
    If score_threshold is not None: similarity_score_threshold search.
    """
    # Lấy instance Chroma đang lưu trên đĩa
    db = get_vectorstore()

    if score_threshold is None:
        # Chỉ lấy top-k gần nhất
        retriever = db.as_retriever(
            search_kwargs={"k": k},
        )
    else:
        # Lọc theo ngưỡng similarity score
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold,
            },
        )
    return retriever


def retrieve(query: str, k: int = 5, score_threshold: float | None = None):
    """
    Convenience wrapper that returns only Documents for a given query.
    """
    # Gọi retriever theo cấu hình k và score_threshold
    retriever = get_retriever(k=k, score_threshold=score_threshold)
    docs = retriever.invoke(query)
    return docs


def retrieve_with_scores(query: str, k: int = 5) -> List[Tuple[Any, float]]:
    """
    Return (Document, score) pairs for a given query using Chroma directly.
    """
    # Dùng trực tiếp similarity_search_with_score để xem thêm score
    db = get_vectorstore()
    docs_and_scores = db.similarity_search_with_score(query, k=k)
    return docs_and_scores


if __name__ == "__main__":
    # Ví dụ query test
    demo_query = "When should paracetamol be used?"
    print("Query:", demo_query)

    # Test top-k retrieval (không có score)
    docs = retrieve(demo_query, k=5)
    print("\n--- Top-k retrieval ---")
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source") or doc.metadata.get("doc_id")
        page = doc.metadata.get("page") or doc.metadata.get("page_label") or "N/A"
        print(f"\nDoc {i}")
        print(f"Source: {src} | Page: {page}")
        print(doc.page_content[:300], "...")

    # Test retrieval kèm score
    docs_scores = retrieve_with_scores(demo_query, k=5)
    print("\n--- Retrieval with scores ---")
    for i, (doc, score) in enumerate(docs_scores, start=1):
        src = doc.metadata.get("source") or doc.metadata.get("doc_id")
        page = doc.metadata.get("page") or doc.metadata.get("page_label") or "N/A"
        print(f"\nRank {i} | score={score:.4f}")
        print(f"Source: {src} | Page: {page}")
        print(doc.page_content[:300], "...")
