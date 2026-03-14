"""
Qdrant-based retriever. Returns evidence chunks with scores for Layer 1.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Project root for config
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EvidenceChunk:
    """A single retrieved chunk with score and citation metadata."""
    chunk_id: str
    text: str
    score: float
    technique_id: str
    title: str
    url: str
    section: str = "description"


def _load_settings() -> dict[str, Any]:
    """Load configs/settings.yaml or return defaults."""
    path = ROOT / "configs" / "settings.yaml"
    if path.exists():
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {
        "qdrant": {"collection_name": "attack_chunks", "top_k": 12},
        "models": {"embed_dim": 1536},
    }


def search(query: str, top_k: int | None = None) -> list[EvidenceChunk]:
    """
    Embed query, search Qdrant, return list of EvidenceChunk (score included).
    Uses collection_name and top_k from config if top_k not provided.
    """
    settings = _load_settings()
    qdrant_cfg = settings.get("qdrant", {})
    collection_name = qdrant_cfg.get("collection_name", "attack_chunks")
    default_top_k = qdrant_cfg.get("top_k", 12)
    k = top_k if top_k is not None else default_top_k

    from llm.providers import embed_texts
    from retrieval.qdrant_client import get_client

    query_vector = embed_texts([query])[0]
    client = get_client()
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,
    ).points

    chunks = []
    for hit in results:
        payload = hit.payload or {}
        # Qdrant returns score for cosine/dot; for cosine, score is in [0, 1] or similar
        score = float(hit.score) if hit.score is not None else 0.0
        chunks.append(EvidenceChunk(
            chunk_id=payload.get("chunk_id", ""),
            text=payload.get("text", ""),
            score=score,
            technique_id=payload.get("technique_id", ""),
            title=payload.get("title", ""),
            url=payload.get("url", ""),
            section=payload.get("section", "description"),
        ))
    return chunks
