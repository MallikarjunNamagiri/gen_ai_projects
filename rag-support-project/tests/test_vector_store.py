import asyncio
import types
import pytest

import api.services.vector_store as vs


class _FakePoint:
    def __init__(self, score, text):
        self.score = score
        self.payload = {"text": text, "source": "s.txt"}


class _FakeResults:
    def __init__(self, points):
        self.points = points


def test_search_vectors(monkeypatch, mock_qdrant_client):
    # Patch _get_client to return our fixture
    monkeypatch.setattr(vs, "_get_client", lambda: mock_qdrant_client)
    res = asyncio.run(vs.search_vectors([0.0] * 384, top_k=2, threshold=0.0))
    assert isinstance(res, list)
    assert len(res) == 2


def test_search_with_threshold(monkeypatch):
    # Fake client returning two points with different scores
    class FakeClient:
        def query_points(self, *args, **kwargs):
            return _FakeResults([_FakePoint(0.9, "hi"), _FakePoint(0.4, "lo")])

    monkeypatch.setattr(vs, "_get_client", lambda: FakeClient())
    res = asyncio.run(vs.search_vectors([0.0] * 384, top_k=2, threshold=0.8))
    assert len(res) == 1
    assert res[0].score >= 0.8


def test_connection_retry(monkeypatch):
    calls = {"n": 0}

    class FlakyClient:
        def query_points(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("connection error")
            return _FakeResults([_FakePoint(0.85, "ok")])

    monkeypatch.setattr(vs, "_get_client", lambda: FlakyClient())

    # Avoid real sleeps in test by monkeypatching to an async no-op
    async def _nosleep(*_args, **_kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", _nosleep)

    res = asyncio.run(vs.search_vectors([0.0] * 384, top_k=1, threshold=0.0))
    assert len(res) == 1
    assert calls["n"] >= 3
