# Simple RAG Pipeline – Generic Analytics

Retrieval-Augmented Generation system over internal analytics documents.  
**Day 1**: Retrieval-only proof-of-concept  
**Day 2**: Full generation (Ollama), structured JSON output, interactive loop, batch evaluation

---

## Architecture

```
.txt documents → chunk → embed (MiniLM-L6-v2) → ChromaDB
                                                      ↓
                   question → embed → retrieve top-3 chunks
                                                      ↓
                              Ollama (llama3.2) → StructuredAnswer JSON
```

---

## Setup

### 1. Install Python dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Install & start Ollama

```bash
# Download from https://ollama.com  then:
ollama pull llama3.2             # ~2 GB, one-time download
ollama serve                     # keep this running in a separate terminal
```

---

## Usage

### Interactive Q&A loop

```bash
python src/interactive.py
```

During a session:
- Type any analytics question and press Enter
- `sources`  → show raw retrieved chunks from the last answer
- `history`  → list questions asked this session
- `export [file.json]` → save last answer to a file
- `exit`     → quit

### Batch evaluation

```bash
python src/eval.py --verbose
```

Runs 16 predefined questions across all 8 analytics domains and saves a scored JSON report to `eval_results/`.

### One-off pipeline test

```bash
python src/rag_pipeline.py [--model llama3.2] [--reset]
```

`--reset` forces a full rebuild of the ChromaDB vector store.

---

## Structured output format

Every generated answer is a JSON object:

```json
{
  "question": "What drove SMB churn?",
  "answer": "SMB churn reached 8.0%...",
  "reasoning": "The customer_churn_analysis document states...",
  "source_chunks": ["raw chunk text 1", "raw chunk text 2"],
  "source_files": ["customer_churn_analysis.txt"],
  "confidence": "high",
  "timestamp": "2026-08-02T13:00:00+00:00"
}
```

---

## Files

| File | Purpose |
|---|---|
| `src/rag_pipeline.py` | `RAGSystem` class — core pipeline |
| `src/interactive.py` | Terminal Q&A loop |
| `src/eval.py` | Batch evaluation script |
| `data/*.txt` | 8 analytics domain documents |
| `data/eval_questions.json` | 16-question golden evaluation set |
| `chroma_db/` | Persisted vector store (auto-generated) |
| `eval_results/` | Timestamped evaluation reports (auto-generated) |
| `interview_prep.md` | Technical interview study guide |
