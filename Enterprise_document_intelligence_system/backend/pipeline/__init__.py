# ============================================================
# backend/pipeline/__init__.py
# Shared dataclasses used across all pipeline modules.
# ============================================================

from dataclasses import dataclass
import numpy as np


@dataclass
class Chunk:
    text:        str
    source:      str
    chunk_index: int


@dataclass
class EmbeddedChunk:
    text:        str
    source:      str
    chunk_index: int
    embedding:   np.ndarray


@dataclass
class RetrievalResult:
    text:             str
    source:           str
    chunk_index:      int
    similarity_score: float


@dataclass
class PromptPackage:
    system_prompt:  str
    user_prompt:    str
    context_chunks: list


@dataclass
class EvaluationResult:
    query:                  str
    answer:                 str
    faithfulness_score:     float
    answer_relevancy_score: float
    retrieval_precision:    float
    top_retrieval_score:    float
    passed:                 bool
