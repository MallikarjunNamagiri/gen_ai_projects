# api/services/vector_store.py
from dotenv import load_dotenv
import os
from typing import List
import logging

load_dotenv()
logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy-create and validate a Qdrant client.

    Raises RuntimeError with a helpful message if env vars are missing.
    """
    global _client
    if _client is not None:
        return _client

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL is not set. Please configure your .env or env vars.")

    try:
        from qdrant_client import QdrantClient
    except Exception as e:
        raise RuntimeError("Failed to import qdrant_client: " + str(e))

    try:
        _client = QdrantClient(url=url, api_key=api_key)
    except Exception as e:
        raise RuntimeError("Failed to initialize Qdrant client: " + str(e))

    return _client


async def search_vectors(query_vector, top_k=5, threshold=0.7) -> List[object]:
    client = _get_client()
    try:
        # Use query_points method - it accepts query as a vector directly
        # For cosine distance, Qdrant returns similarity scores (higher is better, range ~0-1)
        # Temporarily remove threshold to debug, then we can add it back
        results = client.query_points(
            collection_name="support_docs",
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        # query_points returns a QueryResponse object with .points attribute
        logger.info(f"Found {len(results.points)} results from Qdrant")
        if results.points:
            logger.info(f"Top result score: {results.points[0].score}")
            # Filter by threshold if provided
            filtered = [p for p in results.points if p.score >= threshold]
            logger.info(f"After threshold {threshold} filtering: {len(filtered)} results")
            return filtered if filtered else results.points  # Return all if threshold filters everything
        return results.points
    except Exception as e:
        logger.error(f"Qdrant query error: {e}")
        raise RuntimeError("Qdrant query failed: " + str(e))
