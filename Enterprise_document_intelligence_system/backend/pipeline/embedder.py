# ============================================================
# backend/pipeline/embedder.py
# Local embedding via SentenceTransformers — no API key needed.
# ============================================================

from sentence_transformers import SentenceTransformer
from backend.pipeline import Chunk, EmbeddedChunk
from backend.config import EMBEDDING_MODEL_NAME

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Returns a singleton embedding model instance.
    First call loads from local HuggingFace cache (~90 MB download
    on very first use); all subsequent calls return the cached model.
    """
    global _model
    if _model is None:
        print(f"[Embedder] Loading '{EMBEDDING_MODEL_NAME}' ...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"[Embedder] Model ready.")
    return _model


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Batch-encodes all chunks for efficiency."""
    model  = get_embedding_model()
    texts  = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    return [
        EmbeddedChunk(
            text=chunk.text,
            source=chunk.source,
            chunk_index=chunk.chunk_index,
            embedding=embedding
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]


def embed_query(query: str):
    """Encodes a single query string into a numpy embedding."""
    return get_embedding_model().encode([query])[0]


def embed_text(text: str):
    """Encodes any arbitrary text — used by the evaluator."""
    return get_embedding_model().encode(text, convert_to_tensor=True)
