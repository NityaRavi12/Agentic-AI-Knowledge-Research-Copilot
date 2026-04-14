"""
Provider wrapper for embeddings and chat.

Chat   -> Groq API (free tier, fast) — set GROQ_API_KEY in .env
Embed  -> sentence-transformers (local, free, no API key needed)

Falls back with clear error messages if keys or packages are missing.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, List

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
        except ImportError as exc:
            raise ImportError(
                "Groq package not installed. Run: pip install groq"
            ) from exc

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and add it to your .env file."
            )

        _groq_client = Groq(api_key=api_key)

    return _groq_client


def _extract_json_object(text: str) -> str:
    """
    Extract the first valid-looking JSON object or array from a model response.
    Handles cases where the model wraps JSON in markdown fences or adds extra text.
    """
    cleaned = text.strip()

    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    cleaned = cleaned.strip()

    # If the whole response is already JSON, return it
    if (cleaned.startswith("{") and cleaned.endswith("}")) or (
        cleaned.startswith("[") and cleaned.endswith("]")
    ):
        return cleaned

    # Try to find first JSON object
    obj_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if obj_match:
        return obj_match.group(0).strip()

    # Try to find first JSON array
    arr_match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if arr_match:
        return arr_match.group(0).strip()

    raise ValueError("No JSON object or array found in model response.")


def chat(prompt: str, *, temperature: float = 0.1) -> str:
    """
    Send a single user prompt to the LLM and return the assistant reply.
    Uses Groq. Model is configurable via GROQ_MODEL in .env.
    Default: llama-3.3-70b-versatile
    """
    client = _get_groq_client()
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    return response.choices[0].message.content or ""


def chat_json(prompt: str, *, temperature: float = 0.0) -> dict[str, Any]:
    """
    Send a prompt to the LLM and require a JSON object response.

    This is used for evaluation steps where structured output is needed,
    such as retrieval grading, answer grounding checks, and revision feedback.
    """
    json_prompt = f"""
You must respond with exactly one valid JSON object.
Do not include markdown fences.
Do not include explanations before or after the JSON.
Ensure the JSON is syntactically valid.

{prompt}
""".strip()

    raw = chat(json_prompt, temperature=temperature)

    try:
        json_text = _extract_json_object(raw)
        parsed = json.loads(json_text)
    except Exception as exc:
        raise ValueError(
            "Model did not return valid JSON.\n"
            f"Raw response:\n{raw}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "Expected a JSON object from chat_json(), "
            f"but got {type(parsed).__name__}."
        )

    return parsed


# ---------------------------------------------------------------------------
# Embeddings via sentence-transformers (local, free)
# ---------------------------------------------------------------------------

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from exc

        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        print(
            f"[providers] Loading embedding model '{model_name}' "
            "(first run downloads ~90MB)..."
        )
        _embed_model = SentenceTransformer(model_name)
        print("[providers] Embedding model ready.")

    return _embed_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text strings locally using sentence-transformers.
    Returns a list of embedding vectors.
    """
    if not texts:
        return []

    model = _get_embed_model()
    vectors = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vectors.tolist()