from conftest import ScriptedChatClient, completion
from fastapi.testclient import TestClient

from retail_rag.agent import RetailAssistant
from retail_rag.api import app, get_assistant


def test_chat_endpoint(retriever):
    client = ScriptedChatClient([completion(content="Merhaba!")])
    app.dependency_overrides[get_assistant] = lambda: RetailAssistant(
        client, "gpt-test", retriever, "system", "1.0.0"
    )
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
