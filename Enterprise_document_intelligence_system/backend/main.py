# ============================================================
# backend/main.py
# FastAPI application — all routes + startup lifecycle.
# Run with:  uvicorn backend.main:app --reload --port 8000
# ============================================================

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


from backend import config
from backend.models import (
    QueryRequest, QueryResponse, EvaluationMetrics,
    CacheStats, IndexStats, HealthResponse
)
from backend.pipeline.loader import load_csv
from backend.pipeline.chunker import chunk_csv_by_tenure
from backend.pipeline.embedder import embed_chunks, get_embedding_model
from backend.pipeline.vector_store import VectorStore, set_vector_store, get_vector_store
from backend.pipeline.rag import run_rag_pipeline
from backend.pipeline.evaluator import evaluate_rag_response
from backend.pipeline.generator import check_ollama
from backend.cache.semantic_cache import get_cache


# ============================================================
# Startup — build index once when the server starts
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager: runs setup before the server
    accepts requests, and teardown when it shuts down.
    Index building happens here — once — not on every request.
    """
    print("\n" + "=" * 55)
    print("CLT RAG — Starting up")
    print("=" * 55)

    csv_path = config.CSV_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at '{csv_path}'. "
            f"Update CSV_PATH in backend/config.py"
        )

    # Load → chunk → embed → index
    df             = load_csv(csv_path)
    chunks         = chunk_csv_by_tenure(df, filepath=csv_path)
    embedded       = embed_chunks(chunks)
    store          = VectorStore()
    store.add_chunks(embedded)
    set_vector_store(store)

    # Warm up cache singleton
    get_cache()

    print("=" * 55)
    print(f"Ready — {store.total_vectors} vectors indexed")
    print("=" * 55 + "\n")

    yield   # server is running

    print("[Shutdown] CLT RAG server stopped.")


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="CLT RAG API",
    description="Customer Lifetime RAG pipeline — local Ollama + FAISS + Qdrant",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Routes
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Quick health check — verifies all subsystems are ready."""
    store      = get_vector_store()
    cache      = get_cache()
    ollama_ok  = check_ollama()

    return HealthResponse(
        status       = "ok" if ollama_ok and store.total_vectors > 0 else "degraded",
        ollama_ready = ollama_ok,
        index_ready  = store.total_vectors > 0,
        cache_ready  = cache is not None,
        ollama_model = config.OLLAMA_MODEL
    )


@app.get("/index/stats", response_model=IndexStats, tags=["System"])
def index_stats():
    """Returns FAISS index size and embedding model info."""
    store = get_vector_store()
    return IndexStats(
        total_vectors = store.total_vectors,
        total_chunks  = store.total_chunks,
        model_name    = config.EMBEDDING_MODEL_NAME
    )


@app.get("/cache/stats", response_model=CacheStats, tags=["Cache"])
def cache_stats():
    """Returns semantic cache hit rate and lookup counts."""
    cache = get_cache()
    return CacheStats(**cache.stats)


@app.delete("/cache", tags=["Cache"])
def clear_cache():
    """Wipes all entries from the semantic cache."""
    get_cache().clear()
    return {"message": "Cache cleared successfully."}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    """
    Main RAG endpoint.

    Flow:
      1. Check semantic cache (fast path)
      2. Embed query → FAISS retrieve → build prompt → Ollama generate
      3. Store result in cache
      4. Run evaluation metrics
      5. Return structured response
    """
    cache = get_cache()
    store = get_vector_store()

    if store.total_vectors == 0:
        raise HTTPException(status_code=503, detail="Index not ready.")

    # ── Cache lookup ──────────────────────────────────────────
    cached = cache.lookup(request.query)
    if cached:
        return QueryResponse(
            query             = request.query,
            answer            = cached["answer"],
            sources           = cached["sources"],
            served_from_cache = True,
            cache_similarity  = cached.get("cache_similarity"),
            original_query    = cached.get("original_query")
        )

    # ── RAG pipeline ──────────────────────────────────────────
    result = run_rag_pipeline(
        query               = request.query,
        vector_store        = store,
        top_k               = request.top_k,
        retrieval_threshold = request.retrieval_threshold
    )

    cache.store(request.query, result)

    # ── Evaluation ────────────────────────────────────────────
    eval_result = evaluate_rag_response(
        query         = request.query,
        rag_result    = result,
        context_chunks= result.get("context_chunks_for_eval", [])
    )

    return QueryResponse(
        query               = request.query,
        answer              = result["answer"],
        sources             = result.get("sources", []),
        served_from_cache   = False,
        top_retrieval_score = result.get("top_retrieval_score"),
        input_tokens        = result.get("input_tokens"),
        output_tokens       = result.get("output_tokens"),
        evaluation          = EvaluationMetrics(
            faithfulness_score     = eval_result.faithfulness_score,
            answer_relevancy_score = eval_result.answer_relevancy_score,
            retrieval_precision    = eval_result.retrieval_precision,
            top_retrieval_score    = eval_result.top_retrieval_score,
            passed                 = eval_result.passed
        )
    )
