import pytest
from fastapi import HTTPException
from types import SimpleNamespace

import api.main as main


def test_valid_token(test_client, mock_supabase_client):
    async def _valid_user():
        # simulate supabase verification
        info = mock_supabase_client.verify_token("valid")
        return SimpleNamespace(id=info["sub"], email="v@example.com")

    main.app.dependency_overrides[main.get_current_user] = _valid_user
    # Avoid calling external services — stub embeddings/vector/llm (async fakes)
    monkeypatch = pytest.MonkeyPatch()

    async def _fake_embed(q):
        return [0.0] * 384

    async def _fake_search(q, top_k=5, threshold=0.7):
        return [SimpleNamespace(score=0.9, payload={"text": "doc"})]

    async def _fake_llm(prompt):
        return "ok"

    monkeypatch.setattr(main, "get_embedding", _fake_embed)
    monkeypatch.setattr(main, "search_vectors", _fake_search)
    monkeypatch.setattr(main, "get_llm_response", _fake_llm)

    resp = test_client.post("/api/chat?format=json", json={"query": "hi"}, headers={"Authorization": "Bearer valid"})
    assert resp.status_code == 200
    monkeypatch.undo()


def test_invalid_token(test_client):
    async def _raise():
        raise HTTPException(status_code=401, detail="invalid token")

    main.app.dependency_overrides[main.get_current_user] = _raise
    resp = test_client.post("/api/chat", json={"query": "hi"})
    assert resp.status_code == 401


def test_expired_token(test_client):
    async def _expired():
        raise HTTPException(status_code=401, detail="token expired")

    main.app.dependency_overrides[main.get_current_user] = _expired
    resp = test_client.post("/api/chat", json={"query": "hi"})
    assert resp.status_code == 401
    assert "expired" in resp.json().get("detail", "").lower()
