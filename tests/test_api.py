from fastapi.testclient import TestClient
from backend.main import app
from backend import main as backend_main

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_query_endpoint(monkeypatch):
    def fake_run_rag(question: str, top_k: int = 5, score_threshold: float = 0.2):
        # Giống kiểu mà main.py đang unpack
        fake_answer = f"Fake answer for: {question}"
        fake_sources = [
            {"source": "test.pdf", "page": 1, "snippet": "This is a test snippet."}
        ]
        return fake_answer, fake_sources

    monkeypatch.setattr(backend_main, "run_rag", fake_run_rag)

    payload = {"question": "What is hypertension?"}
    resp = client.post("/query", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["answer"].startswith("Fake answer for")
