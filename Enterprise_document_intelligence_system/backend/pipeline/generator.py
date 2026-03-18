# ============================================================
# backend/pipeline/generator.py
# Local LLM inference via Ollama.
# ============================================================

import sys
import requests
from ollama import Client as OllamaClient

from backend.pipeline import PromptPackage
from backend.config import OLLAMA_HOST, OLLAMA_MODEL

# Singleton client — initialized once, reused for all requests.
# Using OllamaClient(host=...) explicitly avoids the Windows
# localhost→IPv6 resolution bug that breaks module-level ollama.chat().
_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient(host=OLLAMA_HOST)
    return _client


def check_ollama() -> bool:
    """
    Returns True if Ollama is reachable and the configured model
    is available. Prints a clear fix message if either fails.
    """
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [
            m["name"].split(":")[0]
            for m in resp.json().get("models", [])
        ]
        if OLLAMA_MODEL not in available:
            print(
                f"[Ollama] Model '{OLLAMA_MODEL}' not pulled.\n"
                f"  → Run: ollama pull {OLLAMA_MODEL}\n"
                f"  → Available: {available or 'none'}"
            )
            return False
        print(f"[Ollama] Ready | model='{OLLAMA_MODEL}'")
        return True
    except requests.exceptions.ConnectionError:
        print(
            f"[Ollama] Cannot connect to {OLLAMA_HOST}\n"
            f"  → Run: ollama serve"
        )
        return False


def generate_answer(
    prompt_package: PromptPackage,
    model: str = OLLAMA_MODEL,
    max_tokens: int = 1024
) -> dict:
    """
    Sends the assembled prompt to the local Ollama server.
    Returns answer text + token usage + source metadata.
    """
    client   = get_ollama_client()
    response = client.chat(
        model=model,
        options={"num_predict": max_tokens, "temperature": 0},
        messages=[
            {"role": "system", "content": prompt_package.system_prompt},
            {"role": "user",   "content": prompt_package.user_prompt}
        ]
    )

    return {
        "answer":        response["message"]["content"],
        "model":         model,
        "input_tokens":  response.get("prompt_eval_count", 0),
        "output_tokens": response.get("eval_count", 0),
        "chunks_used":   len(prompt_package.context_chunks),
        "sources":       list({c.source for c in prompt_package.context_chunks})
    }
