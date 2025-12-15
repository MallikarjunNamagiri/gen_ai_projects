# api/services/llm.py
from dotenv import load_dotenv
import os
from typing import AsyncGenerator, Callable, Any
import asyncio
import logging

load_dotenv()

logger = logging.getLogger(__name__)
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


async def _call_with_rate_limit_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Call a synchronous LLM client function with exponential backoff on rate limits."""
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            # Heuristic detection of rate limit errors (works for OpenAI-like / Groq-style APIs)
            status_code = getattr(e, "status_code", None)
            response = getattr(e, "response", None)
            if status_code is None and response is not None:
                status_code = getattr(response, "status_code", None)

            message = str(e).lower()
            is_rate_limited = (
                status_code == 429
                or "rate limit" in message
                or "rate-limit" in message
            )

            if not is_rate_limited or attempt >= max_retries:
                logger.error(
                    "LLM call failed on attempt %d/%d: %s",
                    attempt,
                    max_retries,
                    e,
                )
                raise

            logger.warning(
                "LLM rate limit encountered; retrying in %.1fs (attempt %d/%d)",
                delay,
                attempt,
                max_retries,
            )
            await asyncio.sleep(delay)
            delay *= 2


async def stream_llm_response(prompt) -> AsyncGenerator[str, None]:
    client = _get_client()

    def _create_stream():
        return client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=500,
        )

    stream = await _call_with_rate_limit_retry(_create_stream)

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield f"data: {chunk.choices[0].delta.content}\n\n"


async def get_llm_response(prompt) -> str:
    """Get a complete LLM response (non-streaming) for JSON responses."""
    client = _get_client()

    def _create_completion():
        return client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=500,
        )

    completion = await _call_with_rate_limit_retry(_create_completion)
    return completion.choices[0].message.content