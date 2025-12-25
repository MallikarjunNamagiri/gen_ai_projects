import asyncio
from api.services.embeddings import get_embedding


def test_embedding_dimension(mock_embedding_model):
    vec = asyncio.run(get_embedding("hello world"))
    assert hasattr(vec, "__len__")
    assert len(vec) == 384


def test_embedding_consistency(mock_embedding_model):
    v1 = asyncio.run(get_embedding("same text"))
    v2 = asyncio.run(get_embedding("same text"))
    assert v1 == v2


def test_batch_embedding(mock_embedding_model):
    texts = ["one", "two", "three"]
    vecs = asyncio.run(get_embedding(texts))
    assert isinstance(vecs, list)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


def test_openai_modern_client_compat(monkeypatch):
    # Simulate modern OpenAI client with OpenAI(...).embeddings.create(...) returning an object with .data
    class FakeResp:
        def __init__(self):
            self.data = [{"embedding": [0.2] * 384}]

    class FakeEmbeddings:
        def create(self, model, input):
            return FakeResp()

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.embeddings = FakeEmbeddings()

    import sys
    fake_mod = type("m", (), {"OpenAI": FakeOpenAI})
    monkeypatch.setitem(sys.modules, "openai", fake_mod)

    import api.services.embeddings as emb_mod
    # Force re-init of model
    emb_mod._model = None
    v = asyncio.run(emb_mod.get_embedding("hello"))
    assert len(v) == 384


def test_openai_legacy_module_compat(monkeypatch):
    # Simulate legacy openai module with Embedding.create(...) returning a dict
    class FakeResp:
        def __init__(self):
            self.data = [{"embedding": [0.3] * 384}]

    class FakeEmbedding:
        @staticmethod
        def create(model, input):
            return {"data": [{"embedding": [0.3] * 384}]}

    fake_mod = type("m", (), {"Embedding": FakeEmbedding})
    import sys
    monkeypatch.setitem(sys.modules, "openai", fake_mod)

    import api.services.embeddings as emb_mod
    emb_mod._model = None
    v = asyncio.run(emb_mod.get_embedding("hello"))
    assert len(v) == 384


def test_groq_api_key_fallback(monkeypatch):
    # Verify the wrapper uses GROQ_API_KEY if OPENAI_API_KEY isn't set
    class FakeResp:
        def __init__(self):
            self.data = [{"embedding": [0.4] * 384}]

    class FakeEmbeddings:
        def create(self, model, input):
            return FakeResp()

    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key=None):
            captured['api_key'] = api_key
            self.embeddings = FakeEmbeddings()

    import sys
    fake_mod = type("m", (), {"OpenAI": FakeOpenAI})
    monkeypatch.setitem(sys.modules, "openai", fake_mod)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "grokey_123")

    import api.services.embeddings as emb_mod
    emb_mod._model = None
    v = asyncio.run(emb_mod.get_embedding("hello"))
    assert len(v) == 384
    assert captured.get('api_key') == "grokey_123"


def test_groq_wrapper_uses_embeddings_endpoint(monkeypatch):
    # Ensure the Groq wrapper posts to the embeddings endpoint and parses response
    called = {}

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    def fake_post(url, json=None, headers=None):
        called['url'] = url
        called['json'] = json
        called['headers'] = headers
        return FakeResponse({"data": [{"embedding": [0.5] * 384}]})

    monkeypatch.setenv("EMBEDDING_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "grok_test_key")
    import requests as requests_mod
    monkeypatch.setattr(requests_mod, "post", fake_post)

    import api.services.embeddings as emb_mod
    emb_mod._model = None
    v = asyncio.run(emb_mod.get_embedding("hello groq"))
    assert len(v) == 384
    assert called['url'].endswith('/openai/v1/embeddings')
