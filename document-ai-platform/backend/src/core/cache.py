import os
import json
import numpy as np
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import QUERY_CACHE_DIR, QUERY_JSON_PATH, SEMANTIC_SIMILARITY_THRESHOLD
from typing import Optional, Dict, Any, List
from upstash_redis import Redis
from src.core.models import get_embeddings


# Initialize Upstash Client using REST API
UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

redis_client = None
if UPSTASH_URL and UPSTASH_TOKEN:
    redis_client = Redis(url=UPSTASH_URL, token=UPSTASH_TOKEN)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_query_json() -> dict:
    if QUERY_JSON_PATH.exists():
        try:
            with open(QUERY_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_query_json(data: dict):
    with open(QUERY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_query_cache_vectorstore():
    embeddings = get_embeddings()
    index_file = QUERY_CACHE_DIR / "index.faiss"
    pickle_file = QUERY_CACHE_DIR / "index.pkl"
    if index_file.exists() and pickle_file.exists():
        try:
            return FAISS.load_local(
                str(QUERY_CACHE_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            return None
    return None

def check_semantic_cache(
    query: str, 
    doc_hash: str, 
    similarity_threshold: float = 0.88
) -> Optional[Dict[str, Any]]:
    """
    Checks Upstash Redis for semantically similar previous queries on the active document.
    """
    if not redis_client or not doc_hash:
        return None

    try:
        # Retrieve all cached entries for this document from Upstash
        redis_key = f"cache:{doc_hash}"
        cached_items = redis_client.hgetall(redis_key)
        if not cached_items:
            return None

        # Compute embedding for the incoming query
        embeddings_model = get_embeddings()
        query_vector = embeddings_model.embed_query(query)

        best_match = None
        highest_score = -1.0

        for item_key, raw_entry in cached_items.items():
            entry = json.loads(raw_entry) if isinstance(raw_entry, str) else raw_entry
            entry_vector = entry.get("vector")
            if not entry_vector:
                continue

            score = _cosine_similarity(query_vector, entry_vector)
            if score > highest_score:
                highest_score = score
                best_match = entry

        if highest_score >= similarity_threshold and best_match:
            return {
                "response": best_match.get("answer"),
                "similarity_score": round(highest_score, 4),
                "sources": best_match.get("sources", [])
            }

    except Exception as e:
        print(f"[Cache Warning] Failed to read from Upstash Redis: {e}")

    return None

def store_semantic_cache(
    query: str, 
    answer: str, 
    sources: List[Dict[str, Any]], 
    doc_hash: str
) -> None:
    """
    Persists query embedding, answer, and sources into Upstash Redis.
    """
    if not redis_client or not doc_hash:
        return

    try:
        embeddings_model = get_embeddings()
        query_vector = embeddings_model.embed_query(query)

        entry = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "vector": query_vector
        }

        redis_key = f"cache:{doc_hash}"
        # Use query string as field key within document hash
        redis_client.hset(redis_key, query, json.dumps(entry))
        # Set 7-day TTL on cached document entries
        redis_client.expire(redis_key, 60 * 60 * 24 * 7)

    except Exception as e:
        print(f"[Cache Warning] Failed to write to Upstash Redis: {e}")


def get_cached_queries_count(doc_hash: Optional[str] = None) -> int:
    """Returns the total number of cached queries in Upstash Redis."""
    if not redis_client:
        return 0
    try:
        if doc_hash:
            return redis_client.hlen(f"cache:{doc_hash}")
        keys = redis_client.keys("cache:*")
        total = 0
        for k in keys:
            total += redis_client.hlen(k)
        return total
    except Exception:
        return 0