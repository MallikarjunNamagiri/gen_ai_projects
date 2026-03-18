# CLT RAG Assistant

Customer Lifetime (CLT) analytics Q&A powered by a fully **local** RAG pipeline.
No cloud APIs, no API keys — everything runs on your machine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI  :8501                  │
│         (question input · answer display · metrics)     │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP (REST)
┌───────────────────────▼─────────────────────────────────┐
│                  FastAPI Backend  :8000                  │
│  POST /query   GET /health   GET /cache/stats           │
│  DELETE /cache  GET /index/stats                        │
└──────┬──────────────────┬────────────────┬──────────────┘
       │                  │                │
  ┌────▼────┐       ┌─────▼─────┐   ┌─────▼──────┐
  │  FAISS  │       │  Ollama   │   │   Qdrant   │
  │ (memory)│       │ :11434    │   │ (on-disk)  │
  │ vectors │       │ llama3.2  │   │ sem. cache │
  └────┬────┘       └───────────┘   └────────────┘
       │
  ┌────▼─────────────────────┐
  │ SentenceTransformers     │
  │ all-MiniLM-L6-v2 (local)│
  └──────────────────────────┘
```

| Component | Tool | Runs |
|---|---|---|
| LLM inference | Ollama (`llama3.2`) | localhost:11434 |
| Vector search | FAISS (IndexFlatIP) | in-memory |
| Embeddings | SentenceTransformers | local CPU |
| Semantic cache | Qdrant (embedded) | `./qdrant_storage/` |
| Backend API | FastAPI + Uvicorn | localhost:8000 |
| Frontend | Streamlit | localhost:8501 |

---

## Project Structure

```
clt_rag/
│
├── backend/
│   ├── __init__.py
│   ├── config.py            ← all settings (edit here)
│   ├── main.py              ← FastAPI app + routes + startup
│   ├── models.py            ← Pydantic request/response schemas
│   │
│   ├── pipeline/
│   │   ├── __init__.py      ← shared dataclasses
│   │   ├── loader.py        ← CSV loader
│   │   ├── chunker.py       ← CSV → natural-language chunks
│   │   ├── embedder.py      ← SentenceTransformers (singleton)
│   │   ├── vector_store.py  ← FAISS index (singleton)
│   │   ├── prompt.py        ← prompt builder
│   │   ├── generator.py     ← Ollama LLM client
│   │   ├── rag.py           ← end-to-end pipeline orchestration
│   │   └── evaluator.py     ← faithfulness / relevancy metrics
│   │
│   └── cache/
│       ├── __init__.py
│       └── semantic_cache.py  ← Qdrant on-disk semantic cache
│
├── frontend/
│   └── app.py               ← Streamlit UI
│
├── data/
│   └── sample_clt_data.csv  ← place your CSV here
│
├── requirements.txt
├── .env.example
├── .gitignore
├── start.sh                 ← one-command launch (Linux/macOS)
└── start.bat                ← one-command launch (Windows)
```

---

## One-Time Setup

### 1. Install Ollama

**Linux / macOS**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows** — download installer from https://ollama.com/download

### 2. Pull a model

```bash
ollama pull llama3.2      # 2 GB — recommended default
# or
ollama pull llama3.1      # 4 GB — better reasoning
# or
ollama pull mistral       # 4 GB — strong alternative
# or
ollama pull phi3          # 2 GB — fastest / lightest
```

Update `OLLAMA_MODEL` in `backend/config.py` if you choose a different model.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Place your CSV

Copy your CLT CSV to:
```
data/sample_clt_data.csv
```

Or set a custom path in `backend/config.py`:
```python
CSV_PATH = "./data/your_file.csv"
```

---

## Running the App

### Option A — One command (recommended)

**Linux / macOS**
```bash
chmod +x start.sh
./start.sh
```

**Windows**
```
Double-click start.bat
```

### Option B — Manual (two terminals)

**Terminal 1 — Backend**
```bash
ollama serve                                          # if not already running
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend**
```bash
streamlit run frontend/app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

---

## API Reference

Interactive docs available at **http://localhost:8000/docs** when the backend is running.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check all subsystems |
| `GET` | `/index/stats` | FAISS vector count + model |
| `GET` | `/cache/stats` | Cache hit rate + lookup count |
| `DELETE` | `/cache` | Wipe semantic cache |
| `POST` | `/query` | Run a RAG query |

**POST /query — request body**
```json
{
  "query": "What is the churn rate at tenure 10?",
  "top_k": 5,
  "retrieval_threshold": 0.3
}
```

**POST /query — response**
```json
{
  "query": "...",
  "answer": "...",
  "sources": ["sample_clt_data.csv | Mobile|5G_Plan|..."],
  "served_from_cache": false,
  "top_retrieval_score": 0.74,
  "input_tokens": 812,
  "output_tokens": 95,
  "evaluation": {
    "faithfulness_score": 0.81,
    "answer_relevancy_score": 0.76,
    "retrieval_precision": 0.60,
    "top_retrieval_score": 0.74,
    "passed": true
  }
}
```

---

## Configuration

All settings live in **`backend/config.py`**:

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Model to use for generation |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server address |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Local HuggingFace embedding model |
| `TOP_K` | `5` | Number of retrieved chunks |
| `RETRIEVAL_THRESHOLD` | `0.3` | Min cosine similarity to pass to LLM |
| `TENURE_BAND_SIZE` | `10` | Rows per chunk (smaller = more precise) |
| `CACHE_SIMILARITY_THR` | `0.92` | Semantic cache match threshold |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry expiry (1 hour) |
| `CSV_PATH` | `./data/sample_clt_data.csv` | Input data path |

---

## How the RAG Pipeline Works

```
User question
     │
     ▼
[1] Embed query            ← all-MiniLM-L6-v2 (local)
     │
     ▼
[2] Semantic cache lookup  ← Qdrant on-disk (TTL + cosine similarity)
     │
     ├── HIT → return cached answer immediately
     │
     └── MISS ↓
          │
          ▼
[3] FAISS retrieval        ← top-K most similar tenure-band chunks
          │
          ▼
[4] Threshold filter       ← drop chunks below 0.3 cosine similarity
          │
          ▼
[5] Build prompt           ← system (CLT grounding) + user (context + question)
          │
          ▼
[6] Ollama LLM inference   ← llama3.2 running locally
          │
          ▼
[7] Evaluate response      ← faithfulness + relevancy + retrieval precision
          │
          ▼
[8] Store in cache         ← Qdrant (persists across restarts)
          │
          ▼
     Return answer to UI
```

### Why tenure-first chunk format?
Each CLT chunk starts with `"Tenures 0 to 9 | ..."` so the embedding model
treats the tenure range as the primary semantic signal. Previously, a long
identical cohort header dominated every chunk's embedding, causing the
retriever to return wrong tenure bands for tenure-specific queries.

---

## Troubleshooting

**`ConnectError: [WinError 10049]`**
Windows resolves `localhost` to IPv6 (`::1`) but Ollama listens on IPv4.
Ensure `OLLAMA_HOST = "http://127.0.0.1:11434"` in `config.py`.

**`Model 'llama3.2' not found`**
```bash
ollama pull llama3.2
```

**`CSV not found`**
Update `CSV_PATH` in `backend/config.py` to point to your file.

**`UnexpectedResponse: 400 Bad Request` (Qdrant)**
The timestamp payload index is missing. This is created automatically on
startup via `create_payload_index()` in `semantic_cache.py`.

**Answers are all the same / wrong tenures retrieved**
Check `TENURE_BAND_SIZE` (should be 10) and `RETRIEVAL_THRESHOLD` (should be 0.3).
Both are set in `backend/config.py`.
