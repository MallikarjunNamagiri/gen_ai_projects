import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=True)

# File and Directory Paths
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
QUERY_CACHE_DIR = BASE_DIR / "faiss_query_cache"
QUERY_JSON_PATH = QUERY_CACHE_DIR / "query_payloads.json"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
QUERY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Guardrails & Cache Parameters
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.88"))
GUARDRAIL_CONFIDENCE_THRESHOLD = float(os.getenv("GUARDRAIL_CONFIDENCE_THRESHOLD", "-2.5"))

# Model Providers
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Application Settings
AUTO_REFRESH_ENABLED = os.getenv("AUTO_REFRESH_ENABLED", "false").lower() == "true"
AUTO_REFRESH_SECONDS = int(os.getenv("AUTO_REFRESH_SECONDS", "60"))