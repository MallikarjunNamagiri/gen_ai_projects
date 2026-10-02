from typing import List, Dict, Any
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document
from src.services.ingestion import get_qdrant_client, COLLECTION_NAME
from src.core.models import get_embeddings

def get_registered_documents_from_qdrant() -> List[Dict[str, Any]]:
    """Retrieves unique document profiles directly from Qdrant payloads without disk access."""
    client = get_qdrant_client()
    try:
        # Scroll payload keys to identify unique doc_hash and doc_name
        response, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=200,
            with_payload=["doc_name", "doc_hash"],
            with_vectors=False
        )
        seen = {}
        for point in response:
            payload = point.payload or {}
            h = payload.get("doc_hash")
            n = payload.get("doc_name")
            if h and n and h not in seen:
                seen[h] = {"name": n, "hash": h}
        return list(seen.values())
    except Exception:
        return []

def search_qdrant_with_doc_filter(query: str, doc_hash: str, top_k: int = 5) -> List[Document]:
    """Retrieves chunks isolated strictly to the target document hash."""
    client = get_qdrant_client()
    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)

    search_filter = Filter(
        must=[FieldCondition(key="doc_hash", match=MatchValue(value=doc_hash))]
    )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True
    )

    docs = []
    for hit in results:
        payload = hit.payload or {}
        docs.append(
            Document(
                page_content=payload.get("page_content", ""),
                metadata={
                    "source": payload.get("doc_name", "Document"),
                    "page_number": payload.get("page_number", 1),
                    "score": hit.score,
                    "token_count": payload.get("token_count", 0)
                }
            )
        )
    return docs