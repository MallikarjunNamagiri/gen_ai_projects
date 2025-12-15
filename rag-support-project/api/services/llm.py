# api/services/llm.py
from dotenv import load_dotenv
import os
from typing import AsyncGenerator

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    try:
        from groq import Groq
    except Exception as e:
        raise RuntimeError("Failed to import groq SDK: " + str(e))

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Please configure your .env or env vars.")

    try:
        _client = Groq(api_key=api_key)
    except Exception as e:
        raise RuntimeError("Failed to initialize Groq client: " + str(e))

    return _client


async def stream_llm_response(prompt) -> AsyncGenerator[str, None]:
    client = _get_client()
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=500
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield f"data: {chunk.choices[0].delta.content}\n\n"


async def get_llm_response(prompt) -> str:
    """Get a complete LLM response (non-streaming) for JSON responses."""
    client = _get_client()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=500
    )
    return completion.choices[0].message.content