# ============================================================
# backend/config.py
# All configuration constants — edit here, nowhere else.
# ============================================================

import os

# --- Ollama ---
OLLAMA_HOST  = "http://127.0.0.1:11434"   # use 127.0.0.1 on Windows (not localhost)
OLLAMA_MODEL = "llama3.2"                 # must be pulled: ollama pull llama3.2

# --- Embedding model (HuggingFace, runs locally, ~90 MB) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM        = 384

# --- FAISS / Retrieval ---
TOP_K               = 5
RETRIEVAL_THRESHOLD = 0.3     # lower than plain text — CLT scores are compressed

# --- Qdrant (local on-disk semantic cache) ---
QDRANT_STORAGE_PATH  = "./qdrant_storage"
CACHE_COLLECTION     = "clt_semantic_cache"
CACHE_SIMILARITY_THR = 0.92
CACHE_TTL_SECONDS    = 3600.0

# --- Chunking ---
TENURE_BAND_SIZE = 10

# --- Evaluation thresholds ---
FAITHFULNESS_THR = 0.40
RELEVANCY_THR    = 0.60

# --- Data ---
# Path to the CLT CSV file. Can also be set via the CSV_PATH env variable.
CSV_PATH = os.getenv("CSV_PATH", "./data/sample_clt_data.csv")
