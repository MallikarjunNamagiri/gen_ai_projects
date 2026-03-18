# ============================================================
# backend/pipeline/rag.py
# End-to-end RAG orchestration: embed → retrieve → prompt → generate.
# ============================================================

from backend.pipeline import RetrievalResult
from backend.pipeline.embedder import embed_query
from backend.pipeline.vector_store import VectorStore
from backend.pipeline.prompt import build_rag_prompt
from backend.pipeline.generator import generate_answer
from backend.config import TOP_K, RETRIEVAL_THRESHOLD


def run_rag_pipeline(
    query: str,
    vector_store: VectorStore,
    top_k: int = TOP_K,
    retrieval_threshold: float = RETRIEVAL_THRESHOLD
) -> dict:
    """
    Connects all pipeline components in sequence:
      query → embed → FAISS retrieve → build prompt → Ollama generate

    Returns a unified result dict that the cache and API layer both
    understand. Low-confidence retrievals short-circuit before the
    LLM call to avoid wasting inference time.
    """
    query_embedding   = embed_query(query)
    retrieval_results = vector_store.retrieve(query_embedding, top_k=top_k)
    top_score         = retrieval_results[0].similarity_score if retrieval_results else 0.0

    if not retrieval_results or top_score < retrieval_threshold:
        return {
            "answer": "I could not find relevant information in the CLT data.",
            "reason": "low_retrieval_confidence",
            "top_retrieval_score": top_score,
            "sources": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "context_chunks_for_eval": []
        }

    prompt_package = build_rag_prompt(query, retrieval_results)
    result         = generate_answer(prompt_package)
    result["top_retrieval_score"]     = top_score
    result["context_chunks_for_eval"] = retrieval_results
    return result
