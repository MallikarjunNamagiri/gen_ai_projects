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
