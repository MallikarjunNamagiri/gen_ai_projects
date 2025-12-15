import os
import logging

from fastapi import FastAPI

# Configure model/cache directories to use the ephemeral but writable /tmp
# filesystem provided by Vercel serverless. This allows heavy model files
# (e.g. Hugging Face / sentence-transformers) to be cached across warm
# invocations within the same lambda instance.
TMP_DIR = "/tmp"

# Common environment variables used by Hugging Face / sentence-transformers
os.environ.setdefault("HF_HOME", os.path.join(TMP_DIR, "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(TMP_DIR, "transformers"))
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    os.path.join(TMP_DIR, "sentence_transformers"),
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info(
    "Initialized /tmp-based model cache dirs: HF_HOME=%s, TRANSFORMERS_CACHE=%s, SENTENCE_TRANSFORMERS_HOME=%s",  # noqa: E501
    os.environ.get("HF_HOME"),
    os.environ.get("TRANSFORMERS_CACHE"),
    os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
)

# Import the existing FastAPI app that defines the /api routes
from api.main import app as fastapi_app  # noqa: E402

# This is the ASGI app that Vercel will invoke.
app: FastAPI = fastapi_app


@app.get("/health", include_in_schema=False)
async def health_root():
    """
    Lightweight root health check for Vercel.

    This lives at /health in addition to the existing /api/health defined in
    api.main, giving both a simple platform-level probe and the more detailed
    app-level health endpoint.
    """
    return {"status": "ok"}


