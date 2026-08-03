"""
Batch evaluation script for the RAG Analytics pipeline.

Runs all questions in data/eval_questions.json against the RAGSystem,
scores each answer, and saves a timestamped report.

Scoring heuristics (keyword-based, no external judge LLM required):
  - retrieval_hit   : 1 if any expected keyword appears in retrieved chunks
  - answer_hit      : 1 if any expected keyword appears in the generated answer
  - confidence_score: high=2, medium=1, low=0

Usage:
    python src/eval.py [--model llama3.2] [--reset] [--out results/]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import RAGSystem

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
EVAL_QUESTIONS_PATH = ROOT_DIR / "data" / "eval_questions.json"
DEFAULT_OUT_DIR = ROOT_DIR / "eval_results"

CONFIDENCE_SCORE = {"high": 2, "medium": 1, "low": 0}

ANSI_RESET  = "\033[0m"
ANSI_GREEN  = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED    = "\033[91m"
ANSI_CYAN   = "\033[96m"
ANSI_DIM    = "\033[2m"
ANSI_BOLD   = "\033[1m"


def _c(text, code): return f"{code}{text}{ANSI_RESET}"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _keyword_hit(text: str, keywords: list[str]) -> bool:
    """True if any keyword appears (case-insensitive) in text."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _score_result(result, q_entry: dict) -> dict:
    """Compute per-question scores from a StructuredAnswer + golden entry."""
    expected = q_entry["expected_keywords"]
    chunks_text = " ".join(result.source_chunks)
    retrieved_files = result.source_files

    retrieval_hit = _keyword_hit(chunks_text, expected)
    answer_hit    = _keyword_hit(result.answer + " " + result.reasoning, expected)
    source_hit    = (
        Path(q_entry["source_file"]).name in retrieved_files
    )
    conf_score    = CONFIDENCE_SCORE.get(result.confidence, 0)

    return {
        "id":              q_entry["id"],
        "domain":          q_entry["domain"],
        "question":        q_entry["question"],
        "expected_keywords": expected,
        "answer":          result.answer,
        "reasoning":       result.reasoning,
        "source_files":    result.source_files,
        "confidence":      result.confidence,
        "timestamp":       result.timestamp,
        # --- scores ---
        "retrieval_hit":  retrieval_hit,   # did we retrieve the right text?
        "answer_hit":     answer_hit,      # does the answer contain expected info?
        "source_hit":     source_hit,      # did we retrieve from the right file?
        "confidence_score": conf_score,    # 0-2
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def _aggregate(scored: list[dict]) -> dict:
    n = len(scored)
    if n == 0:
        return {}

    retrieval_precision = sum(s["retrieval_hit"] for s in scored) / n
    answer_precision    = sum(s["answer_hit"]    for s in scored) / n
    source_precision    = sum(s["source_hit"]    for s in scored) / n
    avg_confidence      = sum(s["confidence_score"] for s in scored) / (n * 2)  # normalise 0-1

    by_domain: dict[str, dict] = {}
    for s in scored:
        d = s["domain"]
        by_domain.setdefault(d, {"total": 0, "retrieval_hits": 0, "answer_hits": 0})
        by_domain[d]["total"]          += 1
        by_domain[d]["retrieval_hits"] += int(s["retrieval_hit"])
        by_domain[d]["answer_hits"]    += int(s["answer_hit"])

    domain_summary = {
        domain: {
            "retrieval_accuracy": v["retrieval_hits"] / v["total"],
            "answer_accuracy":    v["answer_hits"]    / v["total"],
        }
        for domain, v in by_domain.items()
    }

    return {
        "total_questions":     n,
        "retrieval_precision": round(retrieval_precision, 3),
        "answer_precision":    round(answer_precision,    3),
        "source_precision":    round(source_precision,    3),
        "avg_confidence_norm": round(avg_confidence,      3),
        "by_domain":           domain_summary,
    }


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------
def _print_summary(metrics: dict, elapsed: float):
    print()
    print(_c("=" * 62, ANSI_CYAN))
    print(_c("  EVALUATION SUMMARY", ANSI_BOLD + ANSI_CYAN))
    print(_c("=" * 62, ANSI_CYAN))
    print(f"  Questions evaluated : {metrics['total_questions']}")
    print(f"  Elapsed time        : {elapsed:.1f}s")
    print()

    def pct(v): return f"{v * 100:.1f}%"

    def colored_pct(v):
        if v >= 0.75: c = ANSI_GREEN
        elif v >= 0.5: c = ANSI_YELLOW
        else: c = ANSI_RED
        return _c(pct(v), c)

    print(f"  Retrieval precision : {colored_pct(metrics['retrieval_precision'])}")
    print(f"  Answer precision    : {colored_pct(metrics['answer_precision'])}")
    print(f"  Source file hit     : {colored_pct(metrics['source_precision'])}")
    print(f"  Avg confidence      : {colored_pct(metrics['avg_confidence_norm'])}")
    print()
    print(_c("  Per-domain breakdown:", ANSI_BOLD))
    for domain, dm in metrics["by_domain"].items():
        ret = dm["retrieval_accuracy"]
        ans = dm["answer_accuracy"]
        print(f"    {domain:<14} retrieval={colored_pct(ret)}  answer={colored_pct(ans)}")
    print(_c("=" * 62, ANSI_CYAN))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(model: str, reset: bool, out_dir: Path, verbose: bool):
    # Load questions
    with open(EVAL_QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    print(_c(f"\n  Loaded {len(questions)} evaluation questions.", ANSI_DIM))

    # Boot RAG system
    print(_c("  Initialising RAG system …", ANSI_DIM))
    try:
        rag = RAGSystem(model=model, reset=reset)
    except RuntimeError as exc:
        print(_c(f"\n  ✗ {exc}", ANSI_RED))
        sys.exit(1)

    print(_c(f"  Running evaluation with model: {model}\n", ANSI_DIM))
    start = datetime.now(timezone.utc)
    scored: list[dict] = []

    for entry in questions:
        qid    = entry["id"]
        domain = entry["domain"]
        q      = entry["question"]

        print(f"  [{qid:02d}/{len(questions)}] {_c(domain, ANSI_CYAN)}: {q[:70]}")

        try:
            result = rag.generate(q)
            score  = _score_result(result, entry)
        except Exception as exc:  # pylint: disable=broad-except
            print(_c(f"       ✗ Error: {exc}", ANSI_RED))
            score = {
                "id": qid, "domain": domain, "question": q,
                "expected_keywords": entry["expected_keywords"],
                "answer": "", "reasoning": "", "source_files": [],
                "confidence": "low", "timestamp": datetime.now(timezone.utc).isoformat(),
                "retrieval_hit": False, "answer_hit": False,
                "source_hit": False, "confidence_score": 0,
                "error": str(exc),
            }

        scored.append(score)

        if verbose:
            hit_r = _c("✓", ANSI_GREEN) if score["retrieval_hit"] else _c("✗", ANSI_RED)
            hit_a = _c("✓", ANSI_GREEN) if score["answer_hit"]    else _c("✗", ANSI_RED)
            conf  = score["confidence"]
            print(f"       retrieval={hit_r}  answer={hit_a}  confidence={conf}")

    end     = datetime.now(timezone.utc)
    elapsed = (end - start).total_seconds()

    metrics = _aggregate(scored)
    _print_summary(metrics, elapsed)

    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = start.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_results_{ts_slug}.json"

    report = {
        "metadata": {
            "model":      model,
            "run_at":     start.isoformat(),
            "elapsed_s":  round(elapsed, 2),
        },
        "metrics": metrics,
        "results": scored,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(_c(f"  ✓ Full report saved to: {out_path}", ANSI_GREEN))
    print()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation for RAG Analytics.")
    parser.add_argument("--model",   default="llama3.2",        help="Ollama model to use")
    parser.add_argument("--reset",   action="store_true",        help="Rebuild vector store")
    parser.add_argument("--out",     default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--verbose", action="store_true",        help="Print per-question scores")
    args = parser.parse_args()

    run(
        model=args.model,
        reset=args.reset,
        out_dir=Path(args.out),
        verbose=args.verbose,
    )
