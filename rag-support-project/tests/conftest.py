import asyncio
from types import SimpleNamespace
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient


# Ensure project src is importable when running tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.main as main


@pytest.fixture
def mock_qdrant_client(monkeypatch):
    class FakePoint:
        def __init__(self, score, text, source="s.txt"):
            self.score = score
            self.payload = {"text": text, "source": source}

    class FakeResults:
        def __init__(self, points):
            self.points = points

    class FakeClient:
        def __init__(self):
            self._calls = 0

        def query_points(self, collection_name, query, limit, with_payload):
            # Return some deterministic results
            pts = [FakePoint(0.9, "doc1"), FakePoint(0.6, "doc2")]
            return FakeResults(pts)

    client = FakeClient()
    yield client


@pytest.fixture
def mock_embedding_model(monkeypatch):
    class FakeModel:
        def encode(self, x):
            # Support both str and list of str
            if isinstance(x, list):
                return [[0.1] * 384 for _ in x]
            return [0.1] * 384

    # Patch the module-level _model so get_embedding uses this fake
    import api.services.embeddings as embeddings_mod

    monkeypatch.setattr(embeddings_mod, "_model", FakeModel())
    yield embeddings_mod._model


@pytest.fixture
def mock_supabase_client():
    # Provide a simple object that tests can use to simulate supabase
    class FakeSupabase:
        def verify_token(self, token):
            if token == "valid":
                return {"sub": "user_valid"}
            if token == "expired":
                raise Exception("token expired")
            raise Exception("invalid token")

    yield FakeSupabase()


@pytest.fixture
def test_client(monkeypatch):
    # Ensure dev mode is off by default for most tests so auth and services are exercised
    monkeypatch.setenv("DEV_MODE", "false")
    client = TestClient(main.app)
    yield client
