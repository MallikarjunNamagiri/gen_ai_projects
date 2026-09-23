# ✦ Document AI: Enterprise Hybrid RAG System

An enterprise-ready, modular Retrieval-Augmented Generation (RAG) system built with hybrid search (dense semantic + sparse lexical), cross-encoder re-ranking, confidence guardrails, FAISS semantic response caching, and conversational memory reformulation.

---

## 📑 Table of Contents

1. [High-Level Architecture](#-high-level-architecture)
2. [Core Intuition & The "AI Brain"](#-core-intuition--the-ai-brain)
   - [1. Hybrid Retrieval (Dense MMR + Sparse BM25)](#1-hybrid-retrieval-dense-mmr--sparse-bm25)
   - [2. Cross-Encoder Re-ranking & Logit Threshold Guardrails](#2-cross-encoder-re-ranking--logit-threshold-guardrails)
   - [3. Semantic Query Caching](#3-semantic-query-caching)
   - [4. Intent Routing: Global Summary vs. Local Precision](#4-intent-routing-global-summary-vs-local-precision)
   - [5. Conversational Query Reformulation](#5-conversational-query-reformulation)
3. [Project Directory Layout](#-project-directory-layout)
4. [Data Flow Pipeline](#-data-flow-pipeline)
5. [Getting Started & Local Setup](#-getting-started--local-setup)
6. [GenAI Interview Prep: Deep-Dive Q&A](#-genai-interview-prep-deep-dive-qa)

---

## 🏛 High-Level Architecture

```
                                  [ User Query ]
                                         │
                                         ▼
                      [ Conversational Contextualizer ]
                       (LLM Reformulates Multi-turn)
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │     FAISS Semantic Query Cache        │
                     │ (Threshold: Cosine Similarity ≥ 0.88) │
                     └───────────────────┬───────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                     [Hit]                               [Miss]
                       │                                   │
                       ▼                                   ▼
             Return Cached Payload               [ Intent Router ]
                                            (Global Summary vs. Local)
                                             ┌─────────────┴─────────────┐
                                      [Global Query]               [Local Query]
                                             │                           │
                                             ▼                           ▼
                                    Stratified Chunk Sampling      [ Hybrid Retrieval ]
                                    (Outline / TOC synthesis)      ├── Dense FAISS (MMR)
                                             │                     └── Sparse BM25
                                             │                           │
                                             │                           ▼
                                             │                  Candidate Chunks (k=20)
                                             │                           │
                                             │                           ▼
                                             │                [ Cross-Encoder Re-ranker ]
                                             │                (MS-MARCO MiniLM L-6-v2)
                                             │                           │
                                             │                           ▼
                                             │                [ Confidence Guardrail ]
                                             │               (Logit Score ≥ -2.5 Cutoff)
                                             │                           │
                                             │                 ┌─────────┴─────────┐
                                             │             [Pass]                [Fail]
                                             │               │                     │
                                             └───────┬───────┘                     ▼
                                                     ▼                    Return Fallback:
                                            [ Groq LLM Stream ]          "Out of Scope /
                                          (openai/gpt-oss-20b)          Not Found"
                                                     │
                                                     ▼
                                          [ Follow-up Parser ]
                                                     │
                                                     ▼
                                           Write-Through to Cache
                                                     │
                                                     ▼
                                            Stream to Streamlit UI
```

---

## 🧠 Core Intuition & The "AI Brain"

### 1. Hybrid Retrieval (Dense MMR + Sparse BM25)

- **The Problem with Dense Only:** Bi-encoders (`all-MiniLM-L6-v2`) encode semantic concepts well, but struggle on exact matches (e.g., product codes, UUIDs, rare acronyms, specific dates).
- **The Problem with Sparse Only:** Lexical models (`BM25`) rely on exact term frequency–inverse document frequency statistics ($TF\text{-}IDF$) and fail when users use synonyms.
- **The Solution:** We combine dense and sparse candidate pools via an `EnsembleRetriever` with weighted reciprocal scoring:
  - **Dense Pool ($k=10$):** Uses Maximal Marginal Relevance (MMR) with $\lambda = 0.7$ to balance relevance with chunk diversity and avoid redundancy.
  - **Sparse Pool ($k=10$):** BM25 captures exact lexical hits.
  - Candidate pool size fed forward: $N = 20$.

### 2. Cross-Encoder Re-ranking & Logit Threshold Guardrails

- **Bi-Encoder vs. Cross-Encoder:**
  - Bi-encoders encode queries and chunks independently ($u = f(q)$, $v = f(d)$) and measure vector cosine similarity $\cos(u, v)$. This enables $O(1)$ vector indexing, but loses token-level cross-attention.
  - Cross-encoders (`cross-encoder/ms-marco-MiniLM-L-6-v2`) pass both query and document through full cross-attention simultaneously:
    $$\text{Score} = \text{Model}([\text{CLS}] \circ q \circ [\text{SEP}] \circ d \circ [\text{EOS}])$$
- **The Guardrail:** Cross-encoder outputs raw logit scores. For unrelated passages, logit scores typically fall below $-3.0$. By enforcing a strict cutoff:
  $$\text{Relevance Guardrail} \implies \text{Logit Score} \ge -2.5$$
  If no chunk clears $-2.5$, retrieval short-circuits. This prevents hallucination by never sending unrelated context to the generator LLM.

### 3. Semantic Query Caching

- Exact hash match caching fails when users ask the same question in slightly different words (e.g., _"What is the warranty period?"_ vs. _"How long does the warranty last?"_).
- **Our Cache Engine:**
  1. Stores incoming questions inside an independent FAISS vector index alongside a key-value store (`query_payloads.json`).
  2. Embeds incoming queries and measures similarity against cached entries for that document hash.
  3. If similarity exceeds $0.88$, the system bypasses retrieval and LLM generation entirely, serving the verified answer with sub-10ms latency.

### 4. Intent Routing: Global Summary vs. Local Precision

- Standard vector search fails on holistic queries like _"Summarize this document"_ or _"What are the key topics?"_, because no single embedding matches an entire document's theme.
- The router checks regex intent:
  - **Global Intent:** Bypasses vector top-$k$ search. Extracts a stratified outline across early chunks (TOC/intro), middle chunks (body topics), and final chunks (conclusions).
  - **Local Intent:** Standard hybrid search + cross-encoder re-ranking.

### 5. Conversational Query Reformulation

- Chat history introduces anaphoric references (e.g., Turn 1: _"Tell me about Section 3"_, Turn 2: _"What are its main risks?"_).
- Directly querying a vector store with _"What are its main risks?"_ yields poor results.
- A fast LLM chain analyzes the rolling chat history and rewrites the user input into a self-contained, standalone query before it hits retrieval.

---

## 📁 Project Directory Layout

```text
document-ai/
├── data/                         # Uploaded and persistent source documents
├── faiss_index/                  # Persisted FAISS vector index for documents
├── faiss_query_cache/            # Semantic cache FAISS index and JSON payload store
├── src/
│   ├── __init__.py
│   ├── config.py                 # Central configurations, models, and path constants
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py             # Groq LLM, Embeddings, and Cross-Encoder loaders
│   │   ├── cache.py              # Semantic query cache lookup and storage
│   │   └── guardrails.py         # Cross-encoder re-ranking & logit cutoffs
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion.py          # Document loading, SHA-256 validation, and chunking
│   │   ├── rag.py                # Hybrid Ensemble Retriever, prompt templates, follow-ups
│   │   └── router.py             # Query reformulation and global intent routing
│   └── ui/
│       ├── __init__.py
│       ├── styles.py             # Custom CSS injection and DOM scripts
│       └── components.py         # Streamlit UI renderers (chat bubbles, badges, sources)
├── app.py                        # Main application orchestrator
├── requirements.txt              # Production dependency lock
└── README.md
```

---

## 🔄 Data Flow Pipeline

```
[Uploaded Document]
       │
       ▼
[SHA-256 Hashing] ──────► [Check if Already Active]
       │
       ▼
[Recursive Character Splitter] (chunk_size=1000, chunk_overlap=150)
       │
       ├─────────────────────────────────────┐
       ▼                                     ▼
[all-MiniLM-L6-v2 Embeddings]         [BM25 Inverted Index]
       │                                     │
       ▼                                     │
[FAISS Dense Index]                          │
       │                                     │
       └──────────────────┬──────────────────┘
                          ▼
            Ready for Hybrid Execution
```

---

## 🚀 Getting Started & Local Setup

### 1. Prerequisites

- Python 3.10+
- A Groq Cloud API Key ([console.groq.com](https://console.groq.com))

### 2. Installation

```bash
# Clone the repository
git clone <repo-url>
cd document-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-gpt-oss-20b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
SEMANTIC_SIMILARITY_THRESHOLD=0.88
GUARDRAIL_CONFIDENCE_THRESHOLD=-2.5
AUTO_REFRESH_ENABLED=false
AUTO_REFRESH_SECONDS=60
```

### 4. Run Application

```bash
streamlit run app.py
```

---
