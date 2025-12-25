"""Embedding utilities.

This module used to depend on SentenceTransformers (heavy, requires torch).
To keep the runtime lightweight (suitable for Vercel), we use an external
embedding provider (OpenAI) by default. For tests the module-level `_model`
can still be monkeypatched with a fake object that implements `.encode(...)`.
"""

import asyncio
import os
import requests
from typing import List
from typing import Any

_model = None

class _OpenAIWrapper:
    """Minimal wrapper for OpenAI embeddings."""

    def __init__(self, model_name: str | None = None):
        self._api_key = os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("Environment variable OPENAI_API_KEY must be set")
        
        self._model = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self._base_url = "https://api.openai.com/v1/embeddings"

    def encode(self, x: Any) -> List[float] | List[List[float]]:
        # Handle single string vs. list of strings
        is_single = isinstance(x, str)
        if is_single:
            texts = [x]
        else:
            texts = x  # list[str]
        
        payload = {
            "model": self._model,
            "input": texts
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        response = requests.post(self._base_url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Extract embeddings from response
        data = response.json().get("data", [])
        embeddings = [item["embedding"] for item in data]
        
        if is_single:
            return embeddings[0]
        return embeddings
    

class _GroqWrapper:
    """Minimal wrapper for Groq OpenAPI embeddings."""

    def __init__(self, model_name: str | None = None):
        self._api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("Environment variable GROQ_API_KEY or OPENAI_API_KEY must be set")
        
        self._model = model_name or os.getenv("EMBEDDING_MODEL", "default_groq")  # Use Groq's default model name
        # Use the embeddings-compatible endpoint. Keep this configurable via env if needed.
        self._base_url = os.getenv("GROQ_EMBEDDINGS_URL") or "https://api.groq.com/openai/v1/embeddings"

    def encode(self, x: Any) -> List[float] | List[List[float]]:
        # Handle single string vs. list of strings
        is_single = isinstance(x, str)
        if is_single:
            texts = [x]
        else:
            texts = x  # list[str]
        
        payload = {
            "model": self._model,
            "input": texts,
            "type": "query"  # Adjust based on Groq's payload requirements
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        response = requests.post(self._base_url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Extract embeddings based on possible Groq response structures
        resp_json = response.json()
        # Try common shapes: {'data': [{'embedding': [...]}, ...]} or {'embeddings': [...]}
        data = resp_json.get("data") or resp_json.get("embeddings") or []
        embeddings = []
        for item in data:
            if isinstance(item, dict):
                v = item.get("embedding") or item.get("vector") or item.get("embeddings")
                if isinstance(v, list):
                    embeddings.append(v)
                else:
                    # If the dict itself looks like an embedding list, try flatten
                    possible = [val for val in item.values() if isinstance(val, list)]
                    if possible:
                        embeddings.append(possible[0])
            elif isinstance(item, list):
                embeddings.append(item)
            else:
                # Unknown shape, skip
                continue
        
        if is_single:
            return embeddings[0] if embeddings else []
        return embeddings


def _init_model():
    """Use either OpenAI or Groq based on EMBEDDING_PROVIDER."""
    provider = os.getenv("EMBEDDING_PROVIDER").lower()
    if provider == "groq":
        return _GroqWrapper()
    else:
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
    