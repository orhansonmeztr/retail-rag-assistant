from conftest import ScriptedChatClient, completion
from fastapi.testclient import TestClient

from retail_rag.agent import RetailAssistant
from retail_rag.api import app, assistant_or_503, get_api_keys


def test_chat_endpoint(retriever):
    client = ScriptedChatClient([completion(content="Merhaba!")])
    app.dependency_overrides[assistant_or_503] = lambda: RetailAssistant(
        client, "gpt-test", retriever, "system", "1.0.0"
    )
    app.dependency_overrides[get_api_keys] = lambda: ()  # auth tested in test_security.py
    try:
        http = TestClient(app)
        assert http.get("/health").json()["status"] == "ok"
        r = http.post("/chat", json={"question": "Selam"})
        assert r.status_code == 200
        assert r.json()["answer"] == "Merhaba!"
        assert http.post("/chat", json={"question": ""}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_startup_warmup_runs_one_retrieval(retriever, monkeypatch):
    from retail_rag import api

    calls = []

    class Spy:
        def retrieve(self, query, filter=None, top_k=None):
            calls.append((query, top_k))
            return []

    fake = type("A", (), {"retriever": Spy()})()
    monkeypatch.setattr(api, "get_assistant", lambda: fake)
    with TestClient(app):  # entering the context runs the lifespan
        pass
    assert calls == [("warmup", 1)]


def test_unconfigured_assistant_returns_503(monkeypatch):
    from retail_rag import api

    def broken():
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set")

    monkeypatch.setattr(api, "get_assistant", broken)
    app.dependency_overrides[get_api_keys] = lambda: ()
    try:
        r = TestClient(app).post("/chat", json={"question": "x"})
        assert r.status_code == 503 and r.json()["detail"] == "assistant not configured"
    finally:
        app.dependency_overrides.clear()


def test_unauthorized_call_does_not_build_assistant(monkeypatch):
    from retail_rag import api

    built = []
    monkeypatch.setattr(api, "get_assistant", lambda: built.append(1))
    app.dependency_overrides[get_api_keys] = lambda: ("secret",)
    try:
        assert TestClient(app).post("/chat", json={"question": "x"}).status_code == 401
        assert built == []
    finally:
        app.dependency_overrides.clear()
