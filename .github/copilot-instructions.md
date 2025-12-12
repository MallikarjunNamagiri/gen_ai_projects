# Copilot / AI Agent Instructions — rag-support-chatbot

🔧 Purpose: Provide concise, actionable instructions for AI coding agents to be immediately productive working on this RAG (Retrieval-Augmented Generation) support chatbot.

- Project snapshot: Backend API (FastAPI) serves a single `/api/chat` endpoint that:
  - Encodes a user query to an embedding (Sentence Transformers model).
  - Searches a Qdrant vector collection named `support_docs`.
  - Builds a prompt that includes retrieved docs and streams an LLM response (Groq SDK) back to the client as SSE (Server-Sent Events).
  - Uses a Pydantic `User` model and a dev-friendly `get_current_user` auth dependency that accepts missing tokens for dev.

---

## Quick start (Developer workflows)
1. Create and activate a Python venv (Windows PowerShell):

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Required environment variables (create a `.env`) - the code expects:
  - `QDRANT_URL`, `QDRANT_KEY` — Qdrant vector DB
  - `GROQ_API_KEY` — Groq LLM client
  - (Optional for frontend auth): Supabase envs used by frontend client if added later (not required for backend dev)

3. To ingest documents into Qdrant (rebuilds collection):

```powershell
python scripts/ingest.py
```

Notes: `scripts/ingest.py` uses the SentenceTransformers `all-MiniLM-L6-v2` model and creates the `support_docs` collection with vector size 384 and cosine distance. If embedding size changes, update `VectorParams(size=...)` in `ingest.py` accordingly.

4. Run the API server (from repository root):

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

5. Testing the chat endpoint (simple curl):

```powershell
curl -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" -d '{"query":"How do I reset my password?"}'
```

Streaming example (SSE aware client or `curl -N`) will aggregate incremental `data: ...` chunks.

---

## Architecture & Key Components
- `api/main.py` — FastAPI server and chat endpoint. Handles multiple content-types (JSON and URL-encoded forms). Implements full request parsing and SSE streaming.
- `api/models.py` — Pydantic models: `ChatRequest` and `User` (simple shape used across auth and requests).
- `api/middleware/auth.py` — `get_current_user` dependency. By default, it returns mock users for dev if no token present. Replace this with real Supabase (or another JWT provider) in production.
- `api/services/embeddings.py` — SentenceTransformers model wraps `model.encode` into `get_embedding`. Note: It is declared async but uses synchronous `model.encode`; consider moving to a proper async wrapper or offloading to threads if needed.
- `api/services/vector_store.py` — Qdrant client uses `collection_name='support_docs'` and `client.search_points(...)`. The result's `.points` are expected to have `payload['text']` and `payload['source']`.
- `api/services/llm.py` — Groq client code; `stream_llm_response` yields SSE compatible `data: {text}\n\n` pieces.
- `scripts/ingest.py` — Chunking + embedding pipeline for data ingestion. It creates the collection and upserts `PointStruct`s with `payload` containing `text` and `source`.
- `frontend/src/components/Chat.tsx` — Basic React component reading SSE stream; uses Supabase for token retrieval and assembles server-sent partial chunks into `botMsg.content`.

---

## Important patterns, constraints, and caveats
- Streaming pattern (backend): `StreamingResponse(stream_llm_response(prompt), media_type='text/event-stream')`. The LLM generator yields `data: ...` SSE lines — the front end expects raw data pieces (it removes `data:` before appending). Keep the SSE format when changing LLM streaming behavior.

- Auth middleware is intentionally permissive in dev: `get_current_user` returns a dev user if no credentials are supplied. Production work must replace the placeholder with actual JWT/Supabase verification and then reject missing credentials.

- Dimensions and vector configs must match: `embeddings.py` uses `all-MiniLM-L6-v2` (vector dim 384). Ensure `scripts/ingest.py` VectorParams size matches the embedding dimension.

- Sync vs Async: Many exported functions are async but call synchronous libraries (e.g., Sentence Transformers, Qdrant client). This works, but if you expect heavy loads or long inference times, consider using thread executors or switch to async-capable clients to avoid blocking the event loop.

- `api/main.py` robustly handles content-type parsing (JSON and urlencoded), so if adding new clients or forms, use the existing decoder logic or add a single branch.

- `search_vectors` returns `results.points`; ensure the calling code uses `.payload` with `text` — ingest payloads must include `text` and `source`.

---

## Where to make changes (file references & examples)
- Want to change the LLM model or provider: edit `api/services/llm.py`. Keep streaming format consistent (yield `data: ...\n\n`). Example: replace model name or switch to a different SDK but maintain output shape.
- Update embeddings model: `api/services/embeddings.py` (and `scripts/ingest.py` for vector size). Example: switching to a larger model requires setting `VectorParams.size` and may require more memory/time.
- Add auth/production validation: `api/middleware/auth.py` — replace dev placeholder with a supabase client or JWT validator.
- Update collection or schema: `scripts/ingest.py` — chunking, payload fields, and collection creation. Example change: if you want to add `title`, set `payload={'text': text, 'source': source, 'title': title}` and include in prompt-building.

---

## Debugging & troubleshooting (common issues & hints)
- Qdrant connection fails: confirm `QDRANT_URL` and `QDRANT_KEY` env vars, make sure collection exists, and that you're using the expected vector size.
- The embeddings call is slow or blocks: consider offloading to thread pool via `asyncio.to_thread()`, or running separate worker processes for encoding.
- Streaming behavior broken on client: Ensure `api/services/llm.py` yields SSE chunks and `frontend` decodes `response.body.getReader()` properly; the front end expects `data: ` prefix and strips it.
- Model errors from Groq: wrap network and API calls in try/except and return meaningful error codes from the server.

---

## Minimal example changes (small PR pointers)
- Dev-auth enforce change: In `api/middleware/auth.py`, add a feature flag e.g., `OS.getenv('DEV_AUTH', 'True')` to automatically reject requests in production.
- Add a `--collection` CLI argument to `scripts/ingest.py` to make it re-usable across collections.
- Add an integration test (or a `tests/` script) that runs `scripts/ingest.py` against a local Qdrant mock or Docker Qdrant and runs a short query via `curl` to validate SSE response structure.

---

## Contact points & Next steps for contributors
- Look for: `api/main.py`, `api/services/*`, `scripts/ingest.py`, `api/middleware/auth.py`, and `frontend/src/components/Chat.tsx` for the main changes.
- When in doubt: run `scripts/ingest.py` to reindex docs and `uvicorn api.main:app --reload` to debug locally.

---

⚠️ Note: This guidance is built from the current codebase and only reflects discoverable patterns; it intentionally avoids aspirational or broad policy guidance. If you want tests, CI workflows, or a local dockerized Qdrant + Groq mock environment, please ask and I will add concrete configuration files and commands.

If anything above is unclear or missing details you want included in the instructions (e.g., GitHub Actions CI configuration, Swagger endpoints, or more frontend details), tell me what to expand and I’ll iterate. 👍
