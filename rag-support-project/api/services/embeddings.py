"""Embedding utilities.

This module used to depend on SentenceTransformers (heavy, requires torch).
To keep the runtime lightweight (suitable for Vercel), we use an external
embedding provider (OpenAI) by default. For tests the module-level `_model`
can still be monkeypatched with a fake object that implements `.encode(...)`.
"""

import asyncio
import os
from typing import Any

_model = None


class _OpenAIWrapper:
    """Minimal wrapper around the OpenAI embeddings endpoint.

    Provides an `encode` method that mirrors the behaviour of SentenceTransformers
    for compatibility with the rest of the code and tests:
      - If passed a single string -> returns a single list[float]
      - If passed list[str] -> returns list[list[float]]
    """

    def __init__(self, model_name: str | None = None):
        try:
            import openai
        except Exception as e:
            raise RuntimeError("OpenAI package not available: " + str(e))

        self._client = openai
        self._model = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        # allow explicit API key via env var (OpenAI lib will also pick this up)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                # some OpenAI versions expect openai.api_key
                setattr(self._client, "api_key", api_key)
            except Exception:
                # ignore - client may manage keys differently
                pass

    def encode(self, x: Any):
        # The openai client call is synchronous; keep it simple and return raw lists
        resp = self._client.Embedding.create(model=self._model, input=x)
        # `resp.data` is a list of objects with `embedding`
        emb_list = [d["embedding"] for d in resp["data"]]
        if isinstance(x, list):
            return emb_list
        return emb_list[0]


def _init_model():
    """Instantiate the OpenAI-based embedding provider wrapper."""
    return _OpenAIWrapper()


async def get_embedding(text: str | list[str]):
    """Return embedding(s) for a string or list of strings.

    The function prefers a monkeypatched `_model` (used by tests). If no model
    is present, it lazily initializes a network-backed OpenAI client wrapper.
    The underlying blocking encode call is executed with `asyncio.to_thread` so
    the event loop is not blocked.
    """
    global _model
    if _model is None:
        _model = _init_model()

    try:
        embedding = await asyncio.to_thread(_model.encode, text)
        return embedding
    except Exception as e:
        raise RuntimeError("Failed to generate embedding: " + str(e))

