"""
RAG pipeline for generic analytics documents.
Day 2 upgrade: RAGSystem class with Ollama generation + structured JSON output.
Day 2 (follow-up): Rich metadata injected at ingestion & chunking time.

Components:
  - Local embeddings: sentence-transformers/all-MiniLM-L6-v2 (free, no API key)
  - Vector store: ChromaDB (persistent, skips re-embedding if DB exists)
  - LLM: Ollama / llama3.2 (local, free)
  - Output: structured JSON with answer, reasoning, source_chunks,
             chunk_metadata, confidence

Chunk metadata fields (stored in Chroma, surfaced in every answer):
  - file_name      : basename of the source document
  - domain         : business domain inferred from filename
  - chunk_index    : 0-based position of this chunk within its source document
  - total_chunks   : total number of chunks produced from the source document
  - char_count     : character length of the chunk text
  - ingested_at    : ISO-8601 UTC timestamp of the ingestion run

Usage:
  from src.rag_pipeline import RAGSystem
  rag = RAGSystem()
  result = rag.generate("What drove customer churn in Q1?")
  print(result)
"""

import json
import re
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
# Maps filename stem → business domain label.
# Extend this dict when new document types are added.
_DOMAIN_MAP: dict[str, str] = {
    "customer_churn_analysis":  "churn",
    "employee_productivity":     "productivity",
    "financial_kpis":            "financial",
    "inventory_turnover":        "inventory",
    "marketing_campaign_roi":    "marketing",
    "product_usage_metrics":     "product",
    "sales_q1_summary":          "sales",
    "support_ticket_trends":     "support",
}


def _enrich_chunk_metadata(
    chunks: list,
    ingested_at: str,
) -> list:
    """
    Inject rich metadata into every chunk after splitting.

    Added fields
    ------------
    file_name    : str   – basename of the source file
    domain       : str   – business domain (from _DOMAIN_MAP, else "unknown")
    chunk_index  : int   – 0-based position within its source document
    total_chunks : int   – total chunks produced from the same source file
    char_count   : int   – character length of the chunk text
    ingested_at  : str   – ISO-8601 UTC timestamp of the ingestion run
    """
    # Group chunks by their source path to compute per-document totals.
    from collections import defaultdict
    source_groups: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        src = chunk.metadata.get("source", "")
        source_groups[src].append(chunk)

    enriched: list = []
    for src, group in source_groups.items():
        file_name = Path(src).name
        stem      = Path(src).stem
        domain    = _DOMAIN_MAP.get(stem, "unknown")
        total     = len(group)
        for idx, chunk in enumerate(group):
            chunk.metadata.update({
                "file_name":    file_name,
                "domain":       domain,
                "chunk_index":  idx,
                "total_chunks": total,
                "char_count":   len(chunk.page_content),
                "ingested_at":  ingested_at,
            })
            enriched.append(chunk)

    return enriched

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
@dataclass
class StructuredAnswer:
    """The enriched answer returned by RAGSystem.generate()."""
    question: str
    answer: str
    reasoning: str
    source_chunks: List[str]
    source_files: List[str]
    chunk_metadata: List[dict]  # per-chunk metadata (domain, chunk_index, …)
    confidence: str             # "high" | "medium" | "low"
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a precise analytics assistant. You will be given retrieved \
context from internal analytics documents and a question.

Respond ONLY with a valid JSON object — no markdown fences, no extra text — that has \
exactly these keys:
  "answer"    : A concise, direct answer to the question (1-3 sentences).
  "reasoning" : Step-by-step explanation of how you derived the answer from the context.
  "confidence": One of "high", "medium", or "low".
               Use "high"   if ≥2 context chunks directly support the answer.
               Use "medium" if 1 chunk partially supports it.
               Use "low"    if the context does not clearly contain the answer.

If the answer is genuinely not in the context, set confidence to "low" and answer with:
"I don't have enough information in the provided documents."
"""

_HUMAN_TEMPLATE = """Context:
{context}

Question: {question}

JSON:"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", _HUMAN_TEMPLATE),
])


# ---------------------------------------------------------------------------
# RAGSystem
# ---------------------------------------------------------------------------
class RAGSystem:
    """
    Full RAG pipeline: embed → store → retrieve → generate → structured output.

    Parameters
    ----------
    model : str
        Name of the Ollama model to use for generation (default: llama3.2).
    data_dir : Path | None
        Directory containing .txt analytics documents.
    persist_dir : Path | None
        ChromaDB persistence directory.
    reset : bool
        If True, delete and rebuild the vector store from scratch.
    temperature : float
        LLM temperature (0 = deterministic).
    """

    def __init__(
        self,
        model: str = "llama3.2",
        data_dir: Optional[Path] = None,
        persist_dir: Optional[Path] = None,
        reset: bool = False,
        temperature: float = 0.0,
    ):
        self.model = model
        self.data_dir = data_dir or DATA_DIR
        self.persist_dir = persist_dir or CHROMA_DIR
        self.temperature = temperature

        self._embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._vectorstore = self._load_or_build_vectorstore(reset=reset)
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 3})

        try:
            self._llm = ChatOllama(model=model, temperature=temperature)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Could not connect to Ollama (model={model}). "
                "Is Ollama running? Run: ollama serve"
            ) from exc

        self._chain = _PROMPT | self._llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str, k: int = 3) -> list:
        """Return the top-k relevant Document objects for *question*."""
        return self._retriever.invoke(question)

    def generate(self, question: str) -> StructuredAnswer:
        """
        Full RAG: retrieve context → generate structured answer.

        Returns a StructuredAnswer dataclass with all metadata fields populated.
        """
        docs = self.retrieve(question)
        context = "\n\n".join(d.page_content for d in docs)
        source_files = sorted({
            d.metadata.get("file_name")
            or Path(d.metadata.get("source", "unknown")).name
            for d in docs
        })
        # Carry per-chunk metadata through to the answer for traceability.
        chunk_metadata = [
            {
                "file_name":    d.metadata.get("file_name", "unknown"),
                "domain":       d.metadata.get("domain", "unknown"),
                "chunk_index":  d.metadata.get("chunk_index"),
                "total_chunks": d.metadata.get("total_chunks"),
                "char_count":   d.metadata.get("char_count"),
                "ingested_at":  d.metadata.get("ingested_at"),
            }
            for d in docs
        ]

        raw = self._chain.invoke({"context": context, "question": question})
        payload = self._parse_llm_response(raw.content)

        return StructuredAnswer(
            question=question,
            answer=payload.get("answer", ""),
            reasoning=payload.get("reasoning", ""),
            source_chunks=[d.page_content for d in docs],
            source_files=source_files,
            chunk_metadata=chunk_metadata,
            confidence=payload.get("confidence", "low"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_or_build_vectorstore(self, reset: bool) -> Chroma:
        """Load existing Chroma DB or build from documents if absent/reset."""
        if reset and self.persist_dir.exists():
            print(f"[reset] Deleting existing vector store at {self.persist_dir}")
            shutil.rmtree(self.persist_dir)

        db_exists = (self.persist_dir / "chroma.sqlite3").exists()

        if db_exists and not reset:
            print(f"[vectorstore] Loading existing DB from {self.persist_dir}")
            return Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self._embeddings,
            )

        print("[vectorstore] Building vector store from documents …")
        loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
        )
        docs = loader.load()
        print(f"  Loaded {len(docs)} documents")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        print(f"  Created {len(chunks)} chunks")

        # --- Enrich every chunk with structured metadata ---
        ingested_at = datetime.now(timezone.utc).isoformat()
        chunks = _enrich_chunk_metadata(chunks, ingested_at=ingested_at)
        domains_found = sorted({c.metadata["domain"] for c in chunks})
        print(f"  Domains indexed: {domains_found}")
        print(f"  Sample metadata: {chunks[0].metadata}")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=str(self.persist_dir),
        )
        print(f"  Stored in {self.persist_dir}")
        return vectorstore

    @staticmethod
    def _parse_llm_response(raw: str) -> dict:
        """
        Robustly parse the LLM JSON response.
        Handles markdown fences and minor formatting issues.
        """
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?", "", raw).strip()
        # Find the outermost JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Fallback — return minimal structure
        return {
            "answer": raw.strip(),
            "reasoning": "Could not parse structured response.",
            "confidence": "low",
        }


# ---------------------------------------------------------------------------
# CLI entry-point (python src/rag_pipeline.py [--reset] [--model MODEL])
# ---------------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(description="Test the RAG pipeline directly.")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument(
        "--reset", action="store_true", help="Delete and rebuild the vector store"
    )
    args = parser.parse_args()

    rag = RAGSystem(model=args.model, reset=args.reset)

    test_questions = [
        "What were the main drivers of customer churn?",
        "Which product features have the highest adoption?",
        "What is the current inventory turnover situation?",
    ]

    for q in test_questions:
        result = rag.generate(q)
        print(result.to_json())
        print()


if __name__ == "__main__":
    _cli()