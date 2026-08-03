# RAG System – Technical Interview Prep Guide

> **Project**: Simple RAG Analytics Pipeline  
> **Stack**: Python · LangChain · ChromaDB · MiniLM-L6-v2 · Ollama  
> **Purpose**: Be able to explain, defend, and extend every design choice confidently.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Design Decision Q&A](#2-design-decision-qa) ← *most interview-relevant*
3. [Metrics & Evaluation](#3-metrics--evaluation)
4. [Failure Modes & Edge Cases](#4-failure-modes--edge-cases)
5. [Production Hardening](#5-production-hardening)
6. [Scaling to 1M+ Documents](#6-scaling-to-1m-documents)
7. [Code Walkthroughs](#7-code-walkthroughs)
8. [Quick-Fire Questions](#8-quick-fire-questions)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION (one-time)                    │
│                                                             │
│  .txt files  →  DirectoryLoader  →  RecursiveTextSplitter   │
│  (8 docs)       (LangChain)         chunk=500, overlap=80   │
│                      ↓                                      │
│           HuggingFaceEmbeddings (MiniLM-L6-v2)             │
│                      ↓                                      │
│              ChromaDB (persist_directory)                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   QUERY TIME (per question)                  │
│                                                             │
│  User Question  →  embed(question)  →  Chroma ANN search    │
│                         top-k=3 chunks                      │
│                              ↓                              │
│             Prompt assembly: system + context + question     │
│                              ↓                              │
│                   Ollama (llama3.2 local)                    │
│                              ↓                              │
│          StructuredAnswer JSON  →  terminal / eval report   │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Role | Why This Choice |
|---|---|---|
| `DirectoryLoader` | Reads all `.txt` files | Simple glob-based; easy to extend to PDF/Docx |
| `RecursiveCharacterTextSplitter` | Chunks documents | Respects sentence/paragraph boundaries |
| `HuggingFaceEmbeddings` | Converts text → vectors | 100% local, no API cost, 384-dim |
| `ChromaDB` | Stores & searches vectors | Embedded (no separate server needed) |
| `ChatOllama` | Generates answers | Local, privacy-preserving, no API key |
| `StructuredAnswer` | Output schema | Enables programmatic eval and UI rendering |

---

## 2. Design Decision Q&A

### 📦 Chunking Strategy

---

**Q: Why chunk at 500 characters with 80 overlap?**

> **A**: 500 chars ≈ 3–4 sentences — enough context for the embedding model to capture meaning, small enough that a single chunk doesn't mix multiple topics. The 80-character overlap (16%) ensures that answers spanning a sentence boundary are still retrievable. If I used 0 overlap, a sentence split exactly at a chunk boundary would cause the answer to appear in neither chunk.
>
> **Trade-off**: Larger chunks → more context per retrieval but diluted embeddings (the vector averages over more topics). Smaller chunks → more precise embeddings but risk splitting a fact mid-sentence.
>
> **When I'd change it**: For legal documents (dense, long paragraphs) I'd go 1000/200. For bullet-point reports I'd go 300/50.

---

**Q: Why `RecursiveCharacterTextSplitter` instead of `CharacterTextSplitter` or semantic chunking?**

> **A**: `RecursiveCharacterTextSplitter` tries to split on `\n\n` first (paragraphs), then `\n` (lines), then `. ` (sentences), then characters — falling back only if needed. This means it preserves natural text units. `CharacterTextSplitter` only splits on one delimiter and often cuts mid-sentence. Semantic chunking (grouping by embedding similarity) is better but requires an extra embedding pass and is ~10× slower — not justified for 8 small documents.

---

### 🔢 Embedding Model

---

**Q: Why `all-MiniLM-L6-v2` instead of OpenAI `text-embedding-ada-002` or `text-embedding-3-small`?**

> **A**: Three reasons:
> 1. **Cost**: MiniLM is completely free and runs on CPU. ada-002 costs ~$0.10/1M tokens — fine for production, overkill for a prototype.
> 2. **Privacy**: Local embeddings mean document text never leaves the machine — critical for internal analytics data.
> 3. **Quality**: On the MTEB benchmark, MiniLM-L6-v2 scores ~56.3 vs ada-002 ~60.5. The ~4-point gap rarely matters when documents are in English and well-structured.
>
> **When I'd upgrade**: If retrieval recall was consistently below 70% on evaluation, I'd switch to `text-embedding-3-small` (better multilingual support, higher ceiling) or `bge-large-en-v1.5` (state-of-the-art free model).

---

**Q: What does the embedding model actually produce?**

> **A**: It encodes text into a 384-dimensional dense vector where semantic similarity maps to cosine distance. "Customer churn increased" and "clients are leaving" will have vectors close together even with zero lexical overlap. At query time, the question is embedded into the same space, and ChromaDB finds the k nearest chunk vectors using approximate nearest-neighbor (ANN) search.

---

### 🗄️ Vector Store

---

**Q: Why ChromaDB instead of FAISS, Pinecone, or Weaviate?**

> **A**: ChromaDB hits the sweet spot for a local prototype:
> - **vs FAISS**: FAISS is a raw ANN library — no metadata filtering, no persistence without extra code, no built-in document store. Chroma wraps all of that.
> - **vs Pinecone**: Pinecone is a managed cloud service. Great for production but requires an API key and internet access — wrong for a privacy-first local system.
> - **vs Weaviate**: Weaviate is a full knowledge graph / vector database requiring Docker or a managed service. Overkill for 8 documents.
>
> ChromaDB runs in-process as an embedded SQLite-backed store. It persists automatically and requires zero infrastructure.

---

**Q: How does ChromaDB perform similarity search?**

> **A**: By default, Chroma uses HNSW (Hierarchical Navigable Small World) graphs — an ANN algorithm with O(log N) query time. For fewer than ~100k documents, exact search would be fast enough, but Chroma uses HNSW regardless for consistency. The similarity metric is cosine similarity by default.

---

### 🤖 LLM & Generation

---

**Q: Why Ollama and llama3.2 specifically?**

> **A**: 
> - **Ollama**: Provides a clean REST API that `langchain-ollama` wraps directly. Models are downloaded with a single command (`ollama pull`), and serve a standard OpenAI-compatible interface. Zero cost, runs on CPU (slower) or GPU.
> - **llama3.2 (3B)**: The 3B variant is the smallest Llama 3.2 model — fast on CPU, fits in 4GB RAM, instruction-tuned so it follows the "respond in JSON" directive reliably. For richer reasoning, I'd use `llama3.1:8b` or `mistral:7b`.

---

**Q: Why structure the prompt as `system + human` messages instead of a single string?**

> **A**: Chat models (including llama3.2) are trained on conversation data with explicit role tokens: `<|system|>`, `<|user|>`, `<|assistant|>`. Using `ChatPromptTemplate.from_messages` maps naturally to these tokens — the model responds "in character" as an analytics assistant. A single-string prompt works but yields lower-quality instruction-following because it merges context that the model was trained to treat separately.

---

**Q: How do you prevent the LLM from hallucinating information not in the documents?**

> **A**: Three layers:
> 1. **Prompt grounding**: "Use ONLY the following context" instructs the model to cite from retrieved text only.
> 2. **Confidence field**: The model is asked to output `"low"` if the context doesn't contain the answer — a soft self-check.
> 3. **Source tracking**: `source_files` lets a downstream user or evaluator verify the answer against the original document.
>
> For production I'd add: (4) **faithfulness scoring** via RAGAS — compute cosine similarity between the answer and the retrieved chunks to detect when the model drifts.

---

### 📐 Retrieval Design

---

**Q: Why top-k = 3? How did you choose k?**

> **A**: With 8 documents of ~1200 chars each and 500-char chunks, we have ~25–30 chunks total. k=3 retrieves ~10% of the corpus — enough to capture the relevant section without flooding the context window with noise. 
>
> The right k depends on: (a) document density, (b) query specificity, (c) LLM context window size. For 1000-document corpora I'd evaluate k=5 and k=10 using recall@k on a golden set.

---

**Q: What is the difference between dense retrieval (what we use) and sparse retrieval (BM25)?**

> **A**:
> | | Dense (our system) | Sparse (BM25) |
> |---|---|---|
> | Matching | Semantic similarity | Keyword overlap |
> | Handles synonyms | ✅ Yes | ❌ No |
> | Handles typos | Partial | ❌ No |
> | Exact term match | Weaker | ✅ Strong |
> | Speed | O(log N) ANN | O(log N) inverted index |
>
> Dense retrieval wins for paraphrased queries ("clients leaving" → "customer churn"). BM25 wins for exact-term queries ("5.2%" → find "5.2%"). **Hybrid search** (dense + BM25 re-ranked) is the production answer.

---

### 🏗️ Architecture & Code Design

---

**Q: Why refactor from a script into a `RAGSystem` class?**

> **A**: The original script ran everything at module-import time — loading documents, building embeddings, constructing the retriever. That made it impossible to: (a) test individual components, (b) change model/config without editing the file, (c) import from other scripts (interactive.py, eval.py). A class encapsulates state (vectorstore, LLM, retriever) and exposes a clean API (`retrieve()`, `generate()`).

---

**Q: Why skip re-embedding if ChromaDB already exists?**

> **A**: Embedding 30 chunks with MiniLM on CPU takes ~5–10 seconds. That's irrelevant for documents that haven't changed. The check `(persist_dir / "chroma.sqlite3").exists()` detects an existing DB and loads it directly. The `--reset` flag provides an escape hatch for when documents are updated. This follows the principle of "idempotent setup" — running setup twice should be safe and fast.

---

**Q: How does `_parse_llm_response` handle malformed JSON?**

> **A**: The LLM is instructed to return pure JSON, but it occasionally wraps it in markdown fences (` ```json `) or adds surrounding text. The parser: (1) strips markdown fences with a regex, (2) extracts the first `{...}` block with `re.search(r'\{.*\}', text, re.DOTALL)`, (3) falls back to returning the raw text in the `answer` field with `confidence: low`. This makes the system degrade gracefully rather than crash.

---

## 3. Metrics & Evaluation

### What we measure in `eval.py`

| Metric | Definition | How Computed |
|---|---|---|
| **Retrieval Precision** | % questions where retrieved chunks contain expected keywords | keyword match in `source_chunks` |
| **Answer Precision** | % questions where generated answer contains expected keywords | keyword match in `answer + reasoning` |
| **Source File Hit** | % questions where correct source file was retrieved | exact filename match |
| **Avg Confidence** | Normalised mean of confidence scores (0=low, 0.5=medium, 1=high) | avg(score)/2 |

### RAGAS (production evaluation framework)

> **Q: Have you heard of RAGAS? How would you use it here?**

RAGAS provides LLM-judge metrics that don't require a golden answer set:

| RAGAS Metric | Measures | Formula |
|---|---|---|
| **Faithfulness** | Is the answer grounded in context? | NLI(answer, context) |
| **Answer Relevancy** | Does the answer address the question? | cosine(question, answer_embedding) |
| **Context Recall** | Did we retrieve enough relevant context? | requires reference answer |
| **Context Precision** | Are retrieved chunks focused (no noise)? | proportion of relevant chunks in top-k |

Integration would be: `pip install ragas` → wrap each `StructuredAnswer` into a RAGAS `Dataset` → call `evaluate()`.

---

## 4. Failure Modes & Edge Cases

| Failure | Root Cause | Mitigation |
|---|---|---|
| **Hallucination** | LLM generates facts not in context | Faithfulness scoring, source citations |
| **Retrieval miss** | Question phrasing has no semantic overlap with chunks | Hybrid search (dense + BM25), query rewriting |
| **Wrong source retrieved** | Multiple documents cover similar topics | Metadata filtering by domain, re-ranking |
| **JSON parse failure** | LLM wraps output in markdown | Robust `_parse_llm_response()` with regex fallback |
| **Ollama not running** | No local server | `try/except RuntimeError` with clear user message |
| **Stale embeddings** | Documents updated but Chroma not rebuilt | `--reset` flag; or hash-based incremental update |
| **Context window overflow** | k×chunk_size > LLM context window | Limit k, reduce chunk_size, use long-context model |

---

## 5. Production Hardening

### Re-ranking
Add a cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) as a second pass after dense retrieval:
```
Dense retrieval: top-20 chunks
        ↓
Cross-encoder re-rank: score each (question, chunk) pair
        ↓
Take top-3 after re-ranking
```
Cross-encoders are more accurate than bi-encoders but too slow to run over the full corpus, hence the two-stage design.

### Hybrid Search (Dense + BM25)
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(chunks, k=3)
dense = vectorstore.as_retriever(search_kwargs={"k": 3})
hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.3, 0.7])
```

### Query Rewriting
Before retrieval, ask the LLM to rewrite the user question into a retrieval-optimised form:
```
"What drove churn?" → "customer churn causes Q1 drivers SMB enterprise"
```

### Guardrails
- Input: filter PII, reject questions unrelated to analytics domain
- Output: strip any LLM-generated code or links from answers

---

## 6. Scaling to 1M+ Documents

> **Q: How would you scale this system to 1 million documents?**

| Concern | Current (8 docs) | At Scale (1M docs) |
|---|---|---|
| **Embedding** | Run once on startup, ~1s | Async batch pipeline (Celery/Airflow), GPU inference |
| **Vector store** | ChromaDB embedded | Pinecone, Weaviate, or pgvector on PostgreSQL |
| **Retrieval latency** | <100ms (small index) | HNSW still ~10ms at 1M vectors; sharding if needed |
| **LLM** | Local Ollama | Managed API (Gemini/GPT) or dedicated GPU inference |
| **Ingestion** | File glob | Streaming pipeline from S3/GCS with change detection |
| **Observability** | Print statements | LangSmith / OpenTelemetry traces, latency dashboards |
| **Auth & multi-tenancy** | None | Chroma metadata filtering by tenant_id |

**The most important change**: Move from embedded ChromaDB to a persistent server-mode Chroma (or Pinecone) with a separate ingestion pipeline that runs incrementally (only re-embeds changed documents using content hashing).

---

## 7. Code Walkthroughs

### `RAGSystem.__init__` — what happens on startup

```python
RAGSystem(model="llama3.2", reset=False)
```
1. `HuggingFaceEmbeddings` loads the MiniLM model weights into memory (~90MB)
2. `_load_or_build_vectorstore()` checks if `chroma_db/chroma.sqlite3` exists
   - **If yes**: opens the existing collection (instant)
   - **If no**: loads .txt files → splits into chunks → embeds each → stores in Chroma
3. `vectorstore.as_retriever(search_kwargs={"k": 3})` creates a retriever wrapper
4. `ChatOllama(model="llama3.2")` opens a connection to the local Ollama server

### `RAGSystem.generate()` — request lifecycle

```python
result = rag.generate("What drove SMB churn?")
```
1. `self.retrieve(question)` → embeds the question → ANN search → returns 3 `Document` objects
2. `format_docs()` concatenates chunk text with `\n\n` separators
3. `source_files` is extracted from `Document.metadata["source"]` paths
4. `_chain.invoke({"context": ..., "question": ...})` sends the assembled prompt to Ollama
5. `_parse_llm_response()` extracts the JSON from the LLM's raw string output
6. A `StructuredAnswer` dataclass is constructed and returned

### The LCEL chain

```python
self._chain = _PROMPT | self._llm
```
This is LangChain Expression Language (LCEL). The `|` operator chains:
- `_PROMPT`: formats the dict `{"context": ..., "question": ...}` into a list of chat messages
- `self._llm`: passes those messages to Ollama, returns an `AIMessage`

The result is a composable Runnable that can be extended: `_PROMPT | _PROMPT_2 | self._llm | StrOutputParser()`.

---

## 8. Quick-Fire Questions

| Question | One-Line Answer |
|---|---|
| What is RAG? | Retrieval-Augmented Generation: ground LLM answers in retrieved documents to reduce hallucination |
| What is a vector embedding? | A dense numeric representation of text where semantic similarity = geometric proximity |
| What is cosine similarity? | Dot product of unit vectors; 1 = identical direction, 0 = orthogonal, -1 = opposite |
| What is HNSW? | Hierarchical Navigable Small World — a graph-based ANN algorithm, O(log N) query |
| What is the context window? | Max tokens an LLM processes in one forward pass (llama3.2: 128k tokens) |
| What is a retriever in LangChain? | Any object with `.invoke(query) → List[Document]` |
| What is LCEL? | LangChain Expression Language — composes Runnables with `\|` operator |
| What is RAGAS? | Evaluation framework for RAG systems with LLM-as-judge metrics |
| Dense vs sparse retrieval? | Dense = semantic (embedding), Sparse = keyword (BM25/TF-IDF) |
| What is re-ranking? | Second-pass cross-encoder scoring of (query, chunk) pairs to improve precision |
| What is faithfulness in RAGAS? | Whether every statement in the answer is supported by the retrieved context |
| Why overlap in chunking? | Prevents facts that span a chunk boundary from being unsearchable |
| What is a cross-encoder? | A model that takes (query, document) jointly — slower but more accurate than bi-encoders |
| What does `persist_directory` do? | Tells ChromaDB to save to disk (SQLite) so embeddings survive process restarts |
| What is `temperature=0` for? | Makes LLM output deterministic — important for reproducible evaluation |

---

*Built: Day 2 of the Gen AI Projects series.*  
*Stack: Python 3.11 · LangChain 0.3 · ChromaDB 0.5 · Ollama · MiniLM-L6-v2 · llama3.2*
