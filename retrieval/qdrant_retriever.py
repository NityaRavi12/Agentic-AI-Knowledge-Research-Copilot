"""
Qdrant-based retriever. Returns evidence chunks with scores for Layer 1.

Improved behavior:
- detects MITRE ATT&CK technique IDs like T1059 or T1059.001
- performs exact technique_id lookup first
- falls back to multiple semantic query variants
- deduplicates results by chunk_id
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EvidenceChunk:
    chunk_id: str
    text: str
    score: float
    technique_id: str
    title: str
    url: str
    section: str = "description"


def _load_settings() -> dict[str, Any]:
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
        "models": {"embed_dim": 384},
    }


def _extract_attack_id(query: str) -> str | None:
    match = re.search(r"\b(T\d{4}(?:\.\d{3})?)\b", query.upper())
    return match.group(1) if match else None


def _build_query_variants(query: str) -> list[str]:
    attack_id = _extract_attack_id(query)
    variants = [query.strip()]

    if attack_id:
        variants.extend(
            [
                attack_id,
                f"MITRE ATT&CK {attack_id}",
                f"{attack_id} technique",
                f"{attack_id} ATT&CK technique",
            ]
        )

    seen = set()
    deduped = []
    for q in variants:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            deduped.append(q)

    return deduped


def _payload_to_chunk(payload: dict[str, Any], score: float) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=payload.get("chunk_id", ""),
        text=payload.get("text", ""),
        score=score,
        technique_id=payload.get("technique_id", ""),
        title=payload.get("title", ""),
        url=payload.get("url", ""),
        section=payload.get("section", "description"),
    )


def _exact_lookup_by_technique_id(
    client: Any,
    collection_name: str,
    attack_id: str,
    limit: int,
) -> list[EvidenceChunk]:
    """
    Use scroll to find exact technique_id matches in local Qdrant.
    This is reliable for ATT&CK IDs and avoids embedding mismatch.
    """
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=5000,
        with_payload=True,
    )

    exact_matches: list[EvidenceChunk] = []

    for point in points:
        payload = point.payload or {}
        technique_id = str(payload.get("technique_id", "")).upper()
        if technique_id == attack_id:
            exact_matches.append(_payload_to_chunk(payload, score=10.0))

    exact_matches.sort(key=lambda c: (c.section != "description", c.title))
    return exact_matches[:limit]


def search(query: str, top_k: int | None = None) -> list[EvidenceChunk]:
    settings = _load_settings()
    qdrant_cfg = settings.get("qdrant", {})
    collection_name = qdrant_cfg.get("collection_name", "attack_chunks")
    default_top_k = qdrant_cfg.get("top_k", 12)
    k = top_k if top_k is not None else default_top_k

    from llm.providers import embed_texts
    from retrieval.qdrant_client import get_client

    client = get_client()
    attack_id = _extract_attack_id(query)

    # 1. Exact lookup first for ATT&CK IDs
    if attack_id:
        exact_hits = _exact_lookup_by_technique_id(
            client=client,
            collection_name=collection_name,
            attack_id=attack_id,
            limit=k,
        )
        if exact_hits:
            return exact_hits

    # 2. Fallback to semantic retrieval
    query_variants = _build_query_variants(query)
    internal_limit = max(k, 8)

    all_hits: list[EvidenceChunk] = []
    seen_chunk_ids: set[str] = set()

    for variant in query_variants:
        query_vector = embed_texts([variant])[0]
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=internal_limit,
        ).points

        for hit in results:
            payload = hit.payload or {}
            score = float(hit.score) if hit.score is not None else 0.0
            chunk = _payload_to_chunk(payload, score)

            unique_id = chunk.chunk_id or f"{chunk.technique_id}|{chunk.title}|{chunk.section}|{chunk.text[:80]}"
            if unique_id not in seen_chunk_ids:
                seen_chunk_ids.add(unique_id)
                all_hits.append(chunk)

    all_hits.sort(key=lambda c: c.score, reverse=True)
    return all_hits[:k]