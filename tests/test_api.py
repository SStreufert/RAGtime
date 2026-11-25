import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

import api
import healthcheck
from healthcheck import ProbeResult


@pytest.fixture(autouse=True)
def _clean_health_cache():
    healthcheck._success_cache.clear()
    yield
    healthcheck._success_cache.clear()


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient with mocked dependencies."""
    # Mock healthcheck.run_check to return healthy endpoints
    monkeypatch.setattr(
        healthcheck,
        "run_check",
        lambda roles: [
            ProbeResult(role="chat", base_url="http://localhost:1234/v1", model="llama3.2", ok=True, detail="ok"),
            ProbeResult(role="embeddings", base_url="http://localhost:1234/v1", model="nomic-embed-text", ok=True, detail="ok"),
        ],
    )
    # Mock api.answer (where it's used in api.py) to return a deterministic response
    monkeypatch.setattr(
        api,
        "answer",
        lambda question, k=3: (
            "Test answer",
            [
                SimpleNamespace(metadata={"source": "doc1.pdf", "page": 1}),
                SimpleNamespace(metadata={"source": "doc2.pdf", "page": 5}),
            ],
            2,
        ),
    )
    return TestClient(api.app)


class TestHealthEndpoint:
    def test_health_ok_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert len(data["endpoints"]) == 2
        assert data["endpoints"][0]["role"] == "chat"
        assert data["endpoints"][1]["role"] == "embeddings"

    def test_health_degraded_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(
            healthcheck,
            "run_check",
            lambda roles: [
                ProbeResult(role="chat", base_url="http://h/v1", model="m", ok=False, detail="unreachable"),
                ProbeResult(role="embeddings", base_url="http://h/v1", model="m", ok=True, detail="ok"),
            ],
        )
        resp = client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["ok"] is False
        assert data["endpoints"][0]["ok"] is False
        assert data["endpoints"][1]["ok"] is True


class TestAskEndpoint:
    def test_ask_returns_answer_and_sources(self, client):
        resp = client.post("/ask", json={"question": "What is RAG?", "k": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Test answer"
        assert data["sources_used"] == 2
        assert len(data["sources"]) == 2
        assert data["sources"][0]["index"] == 1
        assert data["sources"][0]["source"] == "doc1.pdf"
        assert data["sources"][0]["page"] == 1
        assert data["sources"][1]["source"] == "doc2.pdf"
        assert data["sources"][1]["page"] == 5

    def test_ask_defaults_k_to_3(self, client):
        resp = client.post("/ask", json={"question": "What is RAG?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Test answer"

    def test_ask_validates_question_required(self, client):
        resp = client.post("/ask", json={})
        assert resp.status_code == 422  # Pydantic validation error

    def test_ask_validates_k_positive(self, client):
        resp = client.post("/ask", json={"question": "Q?", "k": 0})
        assert resp.status_code == 422  # Pydantic validation error (ge=1)

    def test_ask_validates_k_type(self, client):
        resp = client.post("/ask", json={"question": "Q?", "k": "not-a-number"})
        assert resp.status_code == 422


