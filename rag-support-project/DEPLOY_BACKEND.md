Deploying the API (backend)

This document explains simple, practical options to deploy the `api/` backend separately from Vercel (so the frontend can stay on Vercel and avoid heavy build-time dependencies).

Short options overview
- Quick: Deploy the Python app (no Docker) on a PaaS that supports Python (Render, Railway) using a start command.
- Reliable: Build and deploy a Docker image on Render / Railway / Fly / AWS ECS.
- Alternative: Replace local heavy ML deps with external embedding APIs (OpenAI/Hugging Face/Groq) and keep the Python API lightweight.

Required environment variables (from project code):
- `QDRANT_URL` and `QDRANT_KEY` — Qdrant vector DB connection
- `GROQ_API_KEY` — Groq LLM key (if used)
- `OPENAI_API_KEY` — OpenAI API key (required for external embeddings if you use OpenAI)
- Optional/frontend-related: `VITE_API_URL` or `REACT_APP_API_URL` in the frontend deployment
- Any other envs in `.env` or referenced in `api/` code (check `api/main.py` and `api/middleware/auth.py`)

Local testing
1. Create and activate a Python venv:

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run ingest if you need vectors locally:

```powershell
python scripts/ingest.py
```

3. Run the API locally:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Dockerfile (example)

Use this Dockerfile to build an image suitable for Render/Railway. It installs dependencies inside a Linux image and runs the uvicorn server.

```dockerfile
# Example Dockerfile
FROM python:3.11-slim
WORKDIR /app

# Avoid cache and reduce layers
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port expected by many PaaS providers
ENV PORT 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
```

Render (Docker) deployment steps
1. Push repo to GitHub.
2. In Render dashboard, create a new "Web Service" using Docker.
   - Connect the repo and branch.
   - Set the build command (if necessary) or let Render use Dockerfile.
   - Set the start command (if not using Dockerfile CMD): `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - Set environment variables (QDRANT_URL, QDRANT_KEY, GROQ_API_KEY, etc.).
   - Optional health check: `curl --fail http://localhost:$PORT/health` (adjust if your app exposes a health endpoint).
3. Deploy and check logs for any missing envs or startup errors.

Render (Static Python service - no Docker) steps (short)
1. Create a new "Web Service" and select Python environment.
2. Set `Start Command` to: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
3. Add env vars and deploy.

Railway (Docker) steps
1. Connect the repo and choose Docker deploy or choose a Python service and configure the start command.
2. Add environment variables in project settings.
3. Deploy and view logs.

Notes & tips
- Build-time disk: Docker builds on Render/Railway are run in their own environment that usually provides more space than Vercel's build container — good for heavy packages (transformers/torch).
- If you want to keep the backend on Vercel: remove heavy packages (`torch`, `transformers`, `sentence-transformers`, `scipy`, etc.) and use an external embeddings API as a drop-in replacement. I can help make that change.
- Vercel-specific: to deploy the Python backend on Vercel while keeping dependencies minimal, use the provided `requirements.vercel.txt` and set the `Build Command` in Vercel to:

    ```bash
    pip install -r requirements.vercel.txt
    ```

    Also add the `OPENAI_API_KEY` (and any other required envs) in the Vercel project settings.
- Secure secrets: use the platform's secrets manager (Render/Railway dashboard) — never commit `.env` to source control.
- Frontend: update `VITE_API_URL` (or the env var you use) in Vercel to point to the deployed backend URL.

Verification
- After deployment, call a simple endpoint (e.g., `GET /api/health` or `curl -X POST $BACKEND_URL/api/chat -H "Content-Type: application/json" -d '{"query":"Hello"}'`) to verify the backend responds and the SSE works.

Need help with any of these steps?
- I can create a `Dockerfile` in `api/` and add a short `render.yaml` or `railway` guidance file.
- I can also create a minimal GitHub Action workflow to build a Docker image and push it to a container registry on each push.

---
If you'd like, I can now add an `api/Dockerfile` and a short `render.yaml` or provide automated deploy steps for your preferred host — tell me which (Render or Railway) and I'll add it next.