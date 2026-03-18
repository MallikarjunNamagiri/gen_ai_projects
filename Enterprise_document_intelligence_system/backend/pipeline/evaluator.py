# ============================================================
# backend/pipeline/evaluator.py
# RAG response evaluation — faithfulness, relevancy, precision.
# ============================================================

from sentence_transformers import util
from backend.pipeline import RetrievalResult, EvaluationResult
from backend.pipeline.embedder import get_embedding_model, embed_text
from backend.config import FAITHFULNESS_THR, RELEVANCY_THR


def evaluate_faithfulness(
    answer: str,
    context_chunks: list[RetrievalResult]
) -> float:
    """
    Max cosine similarity between the answer embedding and any
    retrieved context chunk. Measures whether the answer is
    grounded in the retrieved context (hallucination signal).
    """
    if not context_chunks:
        return 0.0

    model         = get_embedding_model()
    context_texts = [c.text for c in context_chunks]
    answer_emb    = embed_text(answer)
    context_embs  = model.encode(context_texts, convert_to_tensor=True)
    return float(util.cos_sim(answer_emb, context_embs).max())


def evaluate_answer_relevancy(query: str, answer: str) -> float:
    """
    Cosine similarity between query and answer embeddings.
    Measures whether the answer actually addresses what was asked.
    """
    query_emb  = embed_text(query)
    answer_emb = embed_text(answer)
    return float(util.cos_sim(query_emb, answer_emb))


def evaluate_rag_response(
    query: str,
    rag_result: dict,
    context_chunks: list[RetrievalResult],
    faithfulness_threshold: float = FAITHFULNESS_THR,
    relevancy_threshold:    float = RELEVANCY_THR
) -> EvaluationResult:
    """
    Runs all three metrics and returns a structured EvaluationResult
    with a pass/fail decision based on configured thresholds.
    """
    answer       = rag_result.get("answer", "")
    faithfulness = evaluate_faithfulness(answer, context_chunks)
    relevancy    = evaluate_answer_relevancy(query, answer)
    top_score    = rag_result.get("top_retrieval_score", 0.0)

    high_conf = [
        c for c in context_chunks
        if hasattr(c, "similarity_score") and c.similarity_score > 0.6
    ]
    retrieval_precision = (
        len(high_conf) / len(context_chunks) if context_chunks else 0.0
    )

    passed = (
        faithfulness >= faithfulness_threshold and
        relevancy    >= relevancy_threshold    and
        top_score    >= 0.3
    )

    return EvaluationResult(
        query=query,
        answer=answer,
        faithfulness_score=faithfulness,
        answer_relevancy_score=relevancy,
        retrieval_precision=retrieval_precision,
        top_retrieval_score=top_score,
        passed=passed
    )
