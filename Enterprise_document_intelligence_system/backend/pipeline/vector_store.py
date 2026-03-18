# ============================================================
# backend/pipeline/vector_store.py
# In-memory FAISS index — rebuilt from CSV on each startup.
# ============================================================

import numpy as np
import faiss
from backend.pipeline import EmbeddedChunk, RetrievalResult
from backend.config import EMBEDDING_DIM


class VectorStore:
    """
    FAISS IndexFlatIP with a parallel EmbeddedChunk metadata list.
    Vectors are L2-normalised before insertion so that inner-product
    search is equivalent to cosine similarity.

    Rebuilt from scratch on every server startup — fast enough for
    CLT-scale data (hundreds of chunks). For millions of chunks,
    consider persisting the FAISS index to disk with faiss.write_index.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.index  = faiss.IndexFlatIP(embedding_dim)
        self.chunks: list[EmbeddedChunk] = []

    def add_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        vectors = np.array(
            [ec.embedding for ec in embedded_chunks]
        ).astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks.extend(embedded_chunks)
        print(f"[VectorStore] Index contains {self.index.ntotal} vectors.")

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> list[RetrievalResult]:
        query_vector = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                text=chunk.text,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                similarity_score=float(score)
            ))
        return results

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)


# Module-level singleton — populated during app startup
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def set_vector_store(store: VectorStore) -> None:
    global _store
    _store = store
