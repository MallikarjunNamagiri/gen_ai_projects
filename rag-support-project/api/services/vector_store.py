# api/services/vector_store.py
from dotenv import load_dotenv
import os
from typing import List
import logging
import asyncio

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
    """Search vectors in Qdrant with simple retry logic for connection issues."""
    client = _get_client()
    last_error: Exception | None = None

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # Use query_points method - it accepts query as a vector directly
            # For cosine distance, Qdrant returns similarity scores (higher is better, range ~0-1)
            results = client.query_points(
                collection_name="support_docs",
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            # query_points returns a QueryResponse object with .points attribute
            logger.info("Found %d results from Qdrant", len(results.points))
            if results.points:
                logger.info("Top result score: %s", results.points[0].score)
                # Filter by threshold if provided
                filtered = [p for p in results.points if p.score >= threshold]
                logger.info("After threshold %s filtering: %d results", threshold, len(filtered))
                # Return all if threshold filters everything, to at least have something
                return filtered if filtered else results.points
            return results.points
        except Exception as e:
            last_error = e
            logger.error(
                "Qdrant query error on attempt %d/%d: %s",
                attempt,
                max_retries,
                e,
            )
            if attempt >= max_retries:
                break
            # Backoff a bit before retrying
            await asyncio.sleep(1 * attempt)

    raise RuntimeError("Qdrant query failed after retries: " + str(last_error))
