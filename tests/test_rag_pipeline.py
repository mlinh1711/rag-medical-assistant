# tests/test_rag_pipeline.py

from backend.core import rag_pipeline


def test_run_rag_no_docs(monkeypatch):
    """
    Khi retrieve không trả về tài liệu nào, run_rag phải trả về câu xin lỗi.
    Ở đây mock retrieve để luôn trả về list rỗng, bất kể truyền tham số gì.
    """

    def fake_retrieve(*args, **kwargs):
        # Giả lập: không tìm thấy doc nào đủ liên quan
        return []

    # Gắn fake_retrieve vào đúng hàm retrieve mà run_rag đang gọi
    monkeypatch.setattr(rag_pipeline, "retrieve", fake_retrieve)

    result = rag_pipeline.run_rag("Any medical question")

    # run_rag bên bạn có thể trả về tuple, dict hoặc string -> xử lý linh hoạt
    if isinstance(result, tuple):
        answer, sources = result
    elif isinstance(result, dict):
        answer = result.get("answer", "")
        sources = result.get("sources", [])
    else:
        answer = result
        sources = []

    assert "Sorry" in answer
    assert isinstance(sources, list)
    assert len(sources) == 0


def test_run_rag_with_docs(monkeypatch):
    """
    Khi có docs, run_rag phải gọi LLM (đã mock) và trả về câu trả lời.
    Ở đây mock cả retrieve lẫn call_ollama_with_context.
    """

    class FakeDoc:
        def __init__(self, content, source, page):
            self.page_content = content
            self.metadata = {"source": source, "page": page}

    def fake_retrieve(*args, **kwargs):
        # Trả về 1 doc giả để build context
        return [
            FakeDoc(
                "Paracetamol is used to relieve pain and reduce fever.",
                "Instructions for the safe use of medicines.pdf",
                3,
            )
        ]

    def fake_call_ollama_with_context(*args, **kwargs):
        # Giả lập LLM trả về câu trả lời
        return "FAKE ANSWER FROM LLM"

    monkeypatch.setattr(rag_pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        rag_pipeline,
        "call_ollama_with_context",
        fake_call_ollama_with_context,
    )

    result = rag_pipeline.run_rag("When should I use paracetamol?")

    if isinstance(result, tuple):
        answer, sources = result
    elif isinstance(result, dict):
        answer = result.get("answer", "")
        sources = result.get("sources", [])
    else:
        answer = result
        sources = []

    assert "FAKE ANSWER" in answer
    assert isinstance(sources, list)
