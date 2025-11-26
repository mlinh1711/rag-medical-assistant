import sys
from typing import List, Dict

# Thêm repo vào sys.path để import được backend.*
sys.path.append("/kaggle/working/rag-medical-assistant")

from backend.core.retrieval import get_vectorstore


# 10 câu hỏi để test retrieval
TEST_QUERIES: List[str] = [
    "What is the recommended adult dose of paracetamol?",
    "Which conditions require caution when using paracetamol?",
    "What are the main adverse effects of paracetamol?",
    "What is the recommended storage condition for injectable paracetamol?",
    "How is oral rehydration salts (ORS) used for dehydration?",
    "When are NSAIDs contraindicated?",
    "What is the therapeutic action of amoxicillin?",
    "What is the dosage of amoxicillin for children?",
    "When should diazepam be avoided?",
    "How to treat severe dehydration in children?",
]

# Ground truth đơn giản dùng keyword (bạn có thể bổ sung dần)
# Key = question string y như trong TEST_QUERIES
# Value = list keyword cần xuất hiện trong doc đúng
GROUND_TRUTHS: Dict[str, List[str]] = {
    # Ví dụ (tự thêm sau):
    # "What is the recommended adult dose of paracetamol?": [
    #     "500 mg", "3 to 4 times", "adult"
    # ],
}


def doc_contains_any_keyword(text: str, keywords: List[str]) -> bool:
    """Kiểm tra doc có chứa ít nhất một keyword (không phân biệt hoa thường)."""
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def main():
    print("=== Retrieval evaluation with scores ===\n")

    db = get_vectorstore()

    metrics = {
        "total": len(TEST_QUERIES),
        "hit@1": 0,
        "hit@3": 0,
        "hit@5": 0,
    }

    TOP_K = 5

    for idx, q in enumerate(TEST_QUERIES, start=1):
        print("=" * 100)
        print(f"QUESTION {idx}: {q}")
        print("=" * 100)

        # Lấy top-k docs + score cho từng câu hỏi
        docs_scores = db.similarity_search_with_score(q, k=TOP_K)

        if not docs_scores:
            print("No documents returned.\n")
            continue

        print("\n--- Top-k results ---")
        for rank, (doc, score) in enumerate(docs_scores, start=1):
            src = doc.metadata.get("source") or doc.metadata.get("doc_id")
            page = doc.metadata.get("page") or doc.metadata.get("page_label") or "N/A"
            snippet = doc.page_content[:400].replace("\n", " ")
            print(f"\nRank {rank} | score={score:.4f}")
            print(f"Source: {src} | Page: {page}")
            print(snippet, "...")

        # Đánh giá hit@k nếu có ground truth cho câu hỏi này
        gt_keywords = GROUND_TRUTHS.get(q)
        if gt_keywords:
            docs_only = [doc for doc, _ in docs_scores]

            for k in [1, 3, 5]:
                top_docs = docs_only[:k]
                is_hit = any(
                    doc_contains_any_keyword(doc.page_content, gt_keywords)
                    for doc in top_docs
                )
                if is_hit:
                    metrics[f"hit@{k}"] += 1

        print("\n")

    print("\n=== Summary ===")
    print("Total questions:", metrics["total"])
    for k in [1, 3, 5]:
        hit = metrics[f"hit@{k}"]
        acc = hit / metrics["total"] if metrics["total"] > 0 else 0.0
        print(f"hit@{k}: {hit}/{metrics['total']} = {acc:.2%}")


if __name__ == "__main__":
    main()
