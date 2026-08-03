"""
Interactive Q&A terminal loop for the RAG Analytics system.

Usage:
    python src/interactive.py [--model llama3.2] [--reset]

Commands during a session:
    <any question>  →  generate a structured answer
    sources         →  show raw retrieved chunks from the last question
    history         →  list all questions asked this session
    clear           →  clear the screen
    exit / quit     →  end the session
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import RAGSystem, StructuredAnswer

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
DIM     = "\033[2m"
MAGENTA = "\033[95m"

CONFIDENCE_COLOR = {
    "high":   GREEN,
    "medium": YELLOW,
    "low":    RED,
}


def _color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def _print_banner():
    print(_color("=" * 62, CYAN))
    print(_color("  🔍  RAG Analytics Assistant", BOLD + CYAN))
    print(_color(f"  Powered by Ollama · ChromaDB · MiniLM-L6-v2", DIM))
    print(_color("=" * 62, CYAN))
    print()
    print("Commands:")
    print(f"  {_color('sources', YELLOW)}   – show raw chunks from the last answer")
    print(f"  {_color('history', YELLOW)}   – list questions asked this session")
    print(f"  {_color('clear', YELLOW)}     – clear the screen")
    print(f"  {_color('exit/quit', YELLOW)} – end the session")
    print()


def _print_answer(result: StructuredAnswer):
    """Pretty-print the structured answer to the terminal."""
    conf_color = CONFIDENCE_COLOR.get(result.confidence, DIM)
    conf_badge = _color(f" {result.confidence.upper()} ", conf_color + BOLD)

    print()
    print(_color("┌─ Answer " + "─" * 52, CYAN))

    # Answer text
    print(f"\n{BOLD}{result.answer}{RESET}\n")

    # Reasoning
    print(_color("  Reasoning:", DIM))
    for line in result.reasoning.split(". "):
        line = line.strip()
        if line:
            print(f"  {DIM}• {line}.{RESET}")

    print()
    # Source files + confidence on same line
    files = ", ".join(result.source_files) if result.source_files else "unknown"
    print(f"  📄 Sources: {_color(files, MAGENTA)}")
    print(f"  🎯 Confidence: {conf_badge}")
    print(f"  🕒 {_color(result.timestamp, DIM)}")
    print(_color("└" + "─" * 61, CYAN))
    print()


def _print_sources(result: StructuredAnswer):
    """Print the raw retrieved chunks for a result."""
    print()
    print(_color(f"  Retrieved {len(result.source_chunks)} chunk(s):", YELLOW + BOLD))
    for i, chunk in enumerate(result.source_chunks, 1):
        print(_color(f"\n  ── Chunk {i} ──", DIM))
        for line in chunk.strip().split("\n"):
            print(f"  {DIM}{line}{RESET}")
    print()


def _export_json(result: StructuredAnswer, path: Path):
    """Save a result to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(result.to_json())
    print(_color(f"  ✓ Saved to {path}", GREEN))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(model: str, reset: bool):
    print()
    print(_color("  Initialising RAG system …", DIM))

    try:
        rag = RAGSystem(model=model, reset=reset)
    except RuntimeError as exc:
        print(_color(f"\n  ✗ {exc}", RED))
        print(_color("  Start Ollama with: ollama serve", YELLOW))
        sys.exit(1)

    _print_banner()
    print(_color(f"  Model: {model}  |  Type a question to begin.", DIM))
    print()

    history: list[str] = []
    last_result: StructuredAnswer | None = None

    while True:
        try:
            raw = input(_color("> ", CYAN + BOLD)).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n" + _color("  Goodbye!", CYAN))
            break

        if not raw:
            continue

        cmd = raw.lower()

        # ---- built-in commands ----
        if cmd in ("exit", "quit"):
            print(_color("  Goodbye!", CYAN))
            break

        if cmd == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            _print_banner()
            continue

        if cmd == "history":
            if not history:
                print(_color("  No questions asked yet.", DIM))
            else:
                print()
                for i, q in enumerate(history, 1):
                    print(f"  {_color(str(i), CYAN)}. {q}")
                print()
            continue

        if cmd == "sources":
            if last_result is None:
                print(_color("  Ask a question first.", DIM))
            else:
                _print_sources(last_result)
            continue

        if cmd.startswith("export"):
            if last_result is None:
                print(_color("  Ask a question first.", DIM))
            else:
                parts = raw.split(maxsplit=1)
                out_path = Path(parts[1]) if len(parts) > 1 else Path("last_answer.json")
                _export_json(last_result, out_path)
            continue

        # ---- regular question ----
        history.append(raw)
        print(_color("  ⏳ Retrieving & generating …", DIM))

        try:
            result = rag.generate(raw)
            last_result = result
            _print_answer(result)
        except Exception as exc:  # pylint: disable=broad-except
            print(_color(f"\n  ✗ Error: {exc}", RED))
            print()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive RAG Analytics Q&A loop.")
    parser.add_argument(
        "--model", default="llama3.2", help="Ollama model to use (default: llama3.2)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the ChromaDB vector store before starting",
    )
    args = parser.parse_args()
    run(model=args.model, reset=args.reset)
