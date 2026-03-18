# ============================================================
# backend/models.py
# Pydantic models for FastAPI request/response contracts.
# Keeping these separate from pipeline dataclasses means the
# API contract is stable even if internal logic changes.
# ============================================================

from pydantic import BaseModel
from typing import List, Optional


# ── Request ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    retrieval_threshold: float = 0.3


# ── Response ─────────────────────────────────────────────────

class EvaluationMetrics(BaseModel):
    faithfulness_score:    float
    answer_relevancy_score: float
    retrieval_precision:   float
    top_retrieval_score:   float
    passed:                bool


class QueryResponse(BaseModel):
    query:              str
    answer:             str
    sources:            List[str]
    served_from_cache:  bool
    top_retrieval_score: Optional[float] = None
    cache_similarity:   Optional[float]  = None
    original_query:     Optional[str]    = None
    input_tokens:       Optional[int]    = None
    output_tokens:      Optional[int]    = None
    evaluation:         Optional[EvaluationMetrics] = None


class CacheStats(BaseModel):
    total_lookups: int
    total_hits:    int
    hit_rate:      float


class IndexStats(BaseModel):
    total_vectors: int
    total_chunks:  int
    model_name:    str


class HealthResponse(BaseModel):
    status:        str
    ollama_ready:  bool
    index_ready:   bool
    cache_ready:   bool
    ollama_model:  str
