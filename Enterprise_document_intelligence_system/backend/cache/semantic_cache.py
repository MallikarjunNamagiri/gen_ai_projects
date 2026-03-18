# ============================================================
# backend/cache/semantic_cache.py
# Semantic cache backed by local on-disk Qdrant.
# No server process needed — qdrant-client handles file I/O.
# ============================================================

import time
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, Range,
    PayloadSchemaType
)

from backend.pipeline.embedder import get_embedding_model
from backend.config import (
    QDRANT_STORAGE_PATH, CACHE_COLLECTION,
    CACHE_SIMILARITY_THR, CACHE_TTL_SECONDS,
    RETRIEVAL_THRESHOLD
)
import os


class LocalSemanticCache:
    """
    On-disk semantic cache using Qdrant's embedded mode.

    Fixes carried forward:
    ✅ FIX 1: PayloadSchemaType.FLOAT index on 'timestamp' —
              required by Qdrant for Range filters on numeric fields.
    ✅ FIX 2: .query_points() replaces deprecated .search()
    ✅ FIX 3: Empty results guard before results[0] access
    """

    def __init__(
        self,
        storage_path:         str   = QDRANT_STORAGE_PATH,
        similarity_threshold: float = CACHE_SIMILARITY_THR,
        ttl_seconds:          float = CACHE_TTL_SECONDS
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds          = ttl_seconds
        self.total_lookups        = 0
        self.total_hits           = 0

        os.makedirs(storage_path, exist_ok=True)
        self.client = QdrantClient(path=storage_path)
        print(f"[Cache] Local Qdrant storage: '{storage_path}'")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]

        if CACHE_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"[Cache] Created collection: '{CACHE_COLLECTION}'")
        else:
            print(f"[Cache] Connected to existing: '{CACHE_COLLECTION}'")

        # ✅ FIX 1: idempotent — safe on every startup
        self.client.create_payload_index(
            collection_name=CACHE_COLLECTION,
            field_name="timestamp",
            field_schema=PayloadSchemaType.FLOAT
        )

    def lookup(self, query: str) -> dict | None:
        self.total_lookups += 1
        model               = get_embedding_model()
        query_embedding     = model.encode(query).tolist()
        min_ts              = time.time() - self.ttl_seconds

        ttl_filter = Filter(
            must=[FieldCondition(key="timestamp", range=Range(gte=min_ts))]
        )

        # ✅ FIX 2: query_points replaces deprecated search()
        results = self.client.query_points(
            collection_name=CACHE_COLLECTION,
            query=query_embedding,
            query_filter=ttl_filter,
            limit=1,
            with_payload=True
        ).points

        if results and results[0].score >= self.similarity_threshold:
            entry = results[0].payload
            self.total_hits += 1
            print(f"[Cache HIT] score={results[0].score:.4f} | '{entry['original_query']}'")
            return {
                "answer":            entry["answer"],
                "sources":           entry["sources"],
                "served_from_cache": True,
                "cache_similarity":  results[0].score,
                "original_query":    entry["original_query"]
            }

        # ✅ FIX 3: guard against empty list
        best = results[0].score if results else 0.0
        print(f"[Cache MISS] best score={best:.4f}")
        return None

    def store(self, query: str, rag_result: dict) -> None:
        if rag_result.get("top_retrieval_score", 0.0) < RETRIEVAL_THRESHOLD:
            return

        model           = get_embedding_model()
        query_embedding = model.encode(query).tolist()

        self.client.upsert(
            collection_name=CACHE_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=query_embedding,
                    payload={
                        "original_query": query,
                        "answer":         rag_result["answer"],
                        "sources":        rag_result.get("sources", []),
                        "timestamp":      time.time()
                    }
                )
            ]
        )
        print(f"[Cache STORE] '{query[:55]}...' | hit rate: {self.hit_rate:.1%}")

    def clear(self) -> None:
        """Wipes all entries — useful during development."""
        self.client.delete_collection(CACHE_COLLECTION)
        self._ensure_collection()
        self.total_lookups = 0
        self.total_hits    = 0
        print("[Cache] Cleared.")

    @property
    def hit_rate(self) -> float:
        return self.total_hits / self.total_lookups if self.total_lookups else 0.0

    @property
    def stats(self) -> dict:
        return {
            "total_lookups": self.total_lookups,
            "total_hits":    self.total_hits,
            "hit_rate":      round(self.hit_rate, 4)
        }


# Module-level singleton
_cache: LocalSemanticCache | None = None


def get_cache() -> LocalSemanticCache:
    global _cache
    if _cache is None:
        _cache = LocalSemanticCache()
    return _cache
