# api/services/embeddings.py
import asyncio
from typing import Optional

_model = None


def _init_model():
    """Lazy-init the SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        # Raise a helpful message so the caller can understand why import failed
        raise RuntimeError("Failed to import sentence_transformers: " + str(e))

    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


async def get_embedding(text: str):
    """Generate embedding for the given text.

    This function lazily initializes the heavy SentenceTransformer instance to
    avoid importing the package at module import-time (which can fail in some
    environments where binary wheels are mismatched). The sync encode call is
    run on a thread using asyncio.to_thread to avoid blocking the event loop.
    """
    global _model
    if _model is None:
        _model = _init_model()

    # model.encode is blocking; run it in a thread
    try:
        embedding = await asyncio.to_thread(_model.encode, text)
        # If the result is a numpy array, convert to list
        try:
            return embedding.tolist()
        except Exception:
            return embedding
    except Exception as e:
        raise RuntimeError("Failed to encode text: " + str(e))

