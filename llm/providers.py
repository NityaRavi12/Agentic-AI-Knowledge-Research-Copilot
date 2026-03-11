"""
Single provider wrapper for embeddings and chat.
Uses OpenAI when OPENAI_API_KEY is set; otherwise raises a clear error.
"""

import os
from typing import List

# Lazy init to avoid import error when openai not installed
_openai_client = None


def _get_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package required. Install with: pip install openai"
            )
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _require_key():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_API_KEY").strip():
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in .env or environment to use chat and embeddings, "
            "or implement a local embedding/LLM fallback in llm/providers.py."
        )


def chat(prompt: str, *, temperature: float = 0.1) -> str:
    """
    Send a single user prompt to the LLM and return the assistant reply.
    Uses low temperature by default for factual answers.
    """
    _require_key()
    client = _get_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text strings. Returns a list of embedding vectors.
    """
    _require_key()
    if not texts:
        return []
    client = _get_client()
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    response = client.embeddings.create(input=texts, model=model)
    # Preserve order by index
    by_index = {item.index: item.embedding for item in response.data}
    return [by_index[i] for i in range(len(texts))]
