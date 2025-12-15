# api/main.py
from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import Form
from dotenv import load_dotenv
import os
import json
import logging
from api.models import ChatRequest, User
from api.services.embeddings import get_embedding
from api.services.vector_store import search_vectors
from api.services.llm import stream_llm_response, get_llm_response
from api.middleware.auth import get_current_user

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()


def dev_mode_enabled() -> bool:
    return os.getenv("DEV_MODE", "true").lower() in ("1", "true", "yes")


@app.get("/api/health")
async def health_check():
    """Simple health endpoint that attempts to detect whether optional services are available."""
    checks = {
        "dev_mode": dev_mode_enabled(),
        "qdrant_url": bool(os.getenv("QDRANT_URL")),
        "groq_api_key": bool(os.getenv("GROQ_API_KEY")),
    }
    return {"status": "ok", "checks": checks}


@app.on_event("startup")
async def _startup_checks():
    # Log basic expectations and environment status. Do not import heavy libs at
    # module import time; attempt to import them in a try/except and warn.
    logger.info("Starting API; DEV_MODE=%s", dev_mode_enabled())
    if not dev_mode_enabled():
        if not os.getenv("QDRANT_URL"):
            logger.warning("QDRANT_URL is not set. Vector search will be unavailable.")
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY is not set. LLM streaming will be unavailable.")

    # Quick try-import checks to surface potential install issues early
    for pkg in ("sentence_transformers", "qdrant_client", "groq"):
        try:
            __import__(pkg)
            logger.info("%s import OK", pkg)
        except Exception as e:
            logger.warning("Unable to import %s: %s", pkg, str(e))


@app.post("/api/chat")
async def chat(
        request: Request,
        user: User = Depends(get_current_user),
        format: str = Query("stream", description="Response format: 'stream' for SSE or 'json' for JSON")
):
    # Handle both JSON and form data
    content_type = request.headers.get("content-type", "").lower()
    query = None
    
    try:
        if "application/json" in content_type:
            # JSON request
            body = await request.json()
            query = body.get("query", "")
        elif "application/x-www-form-urlencoded" in content_type:
            # URL-encoded form data
            body_bytes = await request.body()
            body_str = body_bytes.decode("utf-8")
            
            # Try to parse as URL-encoded form first
            from urllib.parse import unquote, parse_qs
            try:
                # Decode URL encoding
                decoded = unquote(body_str)
                # Try parsing as form data
                parsed = parse_qs(decoded)
                query = parsed.get("query", [None])[0]
                
                # If not found, try parsing the entire decoded string as JSON
                if not query:
                    try:
                        # Remove trailing '=' if present
                        decoded_clean = decoded.rstrip('=')
                        body_json = json.loads(decoded_clean)
                        query = body_json.get("query", "")
                    except:
                        pass
            except Exception as e:
                # If form parsing fails, try parsing as JSON directly
                try:
                    decoded = unquote(body_str)
                    decoded_clean = decoded.rstrip('=')
                    body_json = json.loads(decoded_clean)
                    query = body_json.get("query", "")
                except:
                    pass
        else:
            # Try to parse as JSON anyway
            try:
                body = await request.json()
                query = body.get("query", "")
            except:
                # Last resort: try reading raw body and parsing
                try:
                    body_bytes = await request.body()
                    body_str = body_bytes.decode("utf-8")
                    from urllib.parse import unquote
                    decoded = unquote(body_str).rstrip('=')
                    body_json = json.loads(decoded)
                    query = body_json.get("query", "")
                except:
                    pass
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing request: {str(e)}")
    
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' field in request body. Expected JSON: {\"query\": \"your question\"}")
    # If DEV_MODE is enabled, return a canned response and skip heavy external services
    if dev_mode_enabled():
        async def _canned_gen():
            yield "data: This is a dev environment fallback response.\n\n"
        return StreamingResponse(_canned_gen(), media_type="text/event-stream")

    try:
        # 1. Embed query
        try:
            query_vector = await get_embedding(query)
        except RuntimeError as e:
            logger.error("Embedding initialization failed: %s", e)
            raise HTTPException(status_code=503, detail=str(e))

        # 2. Retrieve context
        try:
            results = await search_vectors(query_vector, top_k=5, threshold=0.7)
        except RuntimeError as e:
            logger.error("Vector search failed: %s", e)
            raise HTTPException(status_code=503, detail=str(e))
        if not results:
            raise HTTPException(400, "No relevant docs found")

        context = "\n\n".join([r.payload["text"] for r in results])

        # 3. Build prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # 4. Return response based on format
        if format.lower() == "json":
            # Return JSON response
            try:
                response_text = await get_llm_response(prompt)
                return JSONResponse(content={
                    "response": response_text,
                    "query": query,
                    "sources": [{"source": r.payload.get("source", "unknown"), "score": r.score} for r in results]
                })
            except RuntimeError as e:
                logger.error("LLM initialization failed: %s", e)
                raise HTTPException(status_code=503, detail=str(e))
        else:
            # Return streaming response (default)
            try:
                stream = stream_llm_response(prompt)
            except RuntimeError as e:
                logger.error("LLM initialization failed: %s", e)
                raise HTTPException(status_code=503, detail=str(e))
            return StreamingResponse(stream, media_type="text/event-stream")
    except HTTPException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled exception in /api/chat: %s", e)
        raise HTTPException(500, str(e))