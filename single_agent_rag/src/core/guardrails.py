from src.core.models import get_reranker
from src.config import GUARDRAIL_CONFIDENCE_THRESHOLD

def rerank_documents_with_guardrail(
    query: str, 
    docs: list, 
    top_k: int = 3, 
    threshold: float = GUARDRAIL_CONFIDENCE_THRESHOLD
) -> list:
    """Reranks candidates and discards chunks falling below the logit confidence threshold."""
    if not docs:
        return []
    
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    filtered_docs = [doc for doc, score in scored_docs if score >= threshold]
    return filtered_docs[:top_k]