import pytest
from types import SimpleNamespace
from fastapi import HTTPException

import api.main as main


def test_chat_success(monkeypatch, test_client):
    # Override auth dependency to return a user
    async def _fake_user():
        return SimpleNamespace(id="u1", email="u1@example.com")

    main.app.dependency_overrides[main.get_current_user] = _fake_user

    async def fake_embed(q):
        return [0.0] * 384

    async def fake_search(q, top_k=5, threshold=0.7):
        return [SimpleNamespace(score=0.9, payload={"text": "doc", "source": "s"})]

    async def fake_llm(prompt):
        return "This is an answer"

    # Patch both the implementation module and the names imported into `api.main`
    monkeypatch.setattr("api.services.embeddings.get_embedding", fake_embed)
    monkeypatch.setattr(main, "get_embedding", fake_embed)

    monkeypatch.setattr("api.services.vector_store.search_vectors", fake_search)
    monkeypatch.setattr(main, "search_vectors", fake_search)

    monkeypatch.setattr("api.services.llm.get_llm_response", fake_llm)
    monkeypatch.setattr(main, "get_llm_response", fake_llm)

    resp = test_client.post("/api/chat?format=json", json={"query": "hello"}, headers={"Authorization": "Bearer valid"})
    assert resp.status_code == 200
    j = resp.json()
    assert "response" in j and j["response"] == "This is an answer"


def test_chat_unauthorized(monkeypatch, test_client):
    async def _raise_unauth():
        raise HTTPException(status_code=401, detail="unauth")

    main.app.dependency_overrides[main.get_current_user] = _raise_unauth
    resp = test_client.post("/api/chat", json={"query": "hi"})
    assert resp.status_code == 401


def test_chat_empty_results(monkeypatch, test_client):
    # Valid user
    async def _fake_user():
        return SimpleNamespace(id="u1", email="u1@example.com")

    async def fake_embed(q):
        return [0.0] * 384

    async def fake_search(q, top_k=5, threshold=0.7):
        return []

    main.app.dependency_overrides[main.get_current_user] = _fake_user
    monkeypatch.setattr("api.services.embeddings.get_embedding", fake_embed)
    monkeypatch.setattr(main, "get_embedding", fake_embed)
    monkeypatch.setattr("api.services.vector_store.search_vectors", fake_search)
    monkeypatch.setattr(main, "search_vectors", fake_search)

    # JSON format should return helpful message
    resp = test_client.post("/api/chat?format=json", json={"query": "no match"}, headers={"Authorization": "Bearer valid"})
    assert resp.status_code == 200
    j = resp.json()
    assert "couldn't find any relevant information" in j["response"].lower() or "couldn't find" in j["response"].lower()


def test_chat_streaming(monkeypatch, test_client):
    async def _fake_user():
        return SimpleNamespace(id="u1", email="u1@example.com")

    def fake_stream(prompt):
        async def _gen():
            yield "data: chunk1\n\n"
            yield "data: chunk2\n\n"

        return _gen()

    main.app.dependency_overrides[main.get_current_user] = _fake_user
    async def fake_embed(q):
        return [0.0] * 384

    async def fake_search(q, top_k=5, threshold=0.7):
        return [SimpleNamespace(score=0.9, payload={"text": "doc"})]

    monkeypatch.setattr("api.services.embeddings.get_embedding", fake_embed)
    monkeypatch.setattr(main, "get_embedding", fake_embed)
    monkeypatch.setattr("api.services.vector_store.search_vectors", fake_search)
    monkeypatch.setattr(main, "search_vectors", fake_search)
    monkeypatch.setattr("api.services.llm.stream_llm_response", fake_stream)
    monkeypatch.setattr(main, "stream_llm_response", fake_stream)

    resp = test_client.post("/api/chat", json={"query": "hello"}, headers={"Authorization": "Bearer valid"})
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("text/event-stream")
    body = resp.content.decode("utf-8")
    assert "chunk1" in body and "chunk2" in body
