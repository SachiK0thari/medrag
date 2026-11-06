"""FastAPI surface tests with the agent stubbed out."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from medrag import __version__
from medrag.api import main as api
from medrag.models import Answer
from medrag.ratelimit import RateLimiter


class StubAgent:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.questions: list[str] = []

    def answer(self, question: str) -> Answer:
        self.questions.append(question)
        if self.fail:
            raise RuntimeError("pinecone down")
        return Answer(
            question=question,
            text="Stub [1].",
            citations=[],
            confidence=0.5,
            n_retrieval_calls=1,
            total_chunks_considered=0,
        )


@pytest.fixture(autouse=True)
def limiter(monkeypatch: pytest.MonkeyPatch) -> RateLimiter:
    """A fresh limiter per test, so tests never share the process-wide counts."""
    fresh = RateLimiter(per_client=2, window_s=3600, global_per_day=100)
    monkeypatch.setattr(api, "get_limiter", lambda: fresh)
    return fresh


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    stub = StubAgent()
    monkeypatch.setattr(api, "get_agent", lambda: stub)
    return TestClient(api.app), stub


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": __version__}


def test_ask_returns_answer_model(client):
    c, stub = client
    r = c.post("/ask", json={"question": "What did LEADER show?"})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "What did LEADER show?"
    assert body["text"] == "Stub [1]."
    assert stub.questions == ["What did LEADER show?"]


@pytest.mark.parametrize("payload", [{"question": "ab"}, {"question": "x" * 501}, {}])
def test_ask_validates_input(client, payload):
    c, stub = client
    r = c.post("/ask", json=payload)
    assert r.status_code == 422
    assert stub.questions == []


def test_ask_rate_limits_a_client(client):
    c, stub = client
    for _ in range(2):
        assert c.post("/ask", json={"question": "What did LEADER show?"}).status_code == 200
    r = c.post("/ask", json={"question": "What did LEADER show?"})
    assert r.status_code == 429
    assert 0 < int(r.headers["Retry-After"]) <= 3600
    assert "limit" in r.json()["detail"]
    assert len(stub.questions) == 2  # the refused request never reached the agent


def test_invalid_input_does_not_use_up_the_limit(client):
    c, _ = client
    for _ in range(5):
        assert c.post("/ask", json={"question": "ab"}).status_code == 422
    assert c.post("/ask", json={"question": "What did LEADER show?"}).status_code == 200


def test_health_is_not_rate_limited(client):
    c, _ = client
    for _ in range(5):
        assert c.get("/health").status_code == 200


def test_ask_maps_agent_failure_to_500(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api, "get_agent", lambda: StubAgent(fail=True))
    c = TestClient(api.app, raise_server_exceptions=False)
    r = c.post("/ask", json={"question": "anything at all"})
    assert r.status_code == 500
    assert "pinecone down" in r.json()["detail"]


def test_get_agent_is_cached(monkeypatch: pytest.MonkeyPatch):
    created: list[int] = []

    class Counting:
        def __init__(self) -> None:
            created.append(1)

    monkeypatch.setattr(api, "MedRagAgent", Counting)
    monkeypatch.setattr(api, "_agent", None)
    a = api.get_agent()
    b = api.get_agent()
    assert a is b
    assert created == [1]
