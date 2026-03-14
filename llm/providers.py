"""
Provider wrapper for embeddings and chat.

Chat   → Groq API (free tier, fast)  — set GROQ_API_KEY in .env
Embed  → sentence-transformers (local, free, no API key needed)

Falls back with clear error messages if keys or packages are missing.
"""

from __future__ import annotations

import os
from typing import List

# Lazy-loaded clients so missing packages don't crash on import
_groq_client = None
_embed_model = None


# ---------------------------------------------------------------------------
# Chat via Groq
# ---------------------------------------------------------------------------

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "Groq package not installed. Run: pip install groq"
            )
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and add it to your .env file."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def chat(prompt: str, *, temperature: float = 0.1) -> str:
    """
    Send a single user prompt to the LLM and return the assistant reply.
    Uses Groq (free tier). Model is configurable via GROQ_MODEL in .env.
    Default: llama-3.3-70b-versatile — excellent quality, free.
    """
    client = _get_groq_client()
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Embeddings via sentence-transformers (local, free)
# ---------------------------------------------------------------------------

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        print(f"[providers] Loading embedding model '{model_name}' (first run downloads ~90MB)...")
        _embed_model = SentenceTransformer(model_name)
        print(f"[providers] Embedding model ready.")
    return _embed_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text strings locally using sentence-transformers.
    Returns a list of embedding vectors (384-dim for all-MiniLM-L6-v2).
    No API key or internet connection needed after first download.
    """
    if not texts:
        return []
    model = _get_embed_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()
