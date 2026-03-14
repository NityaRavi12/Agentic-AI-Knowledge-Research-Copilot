"""
Qdrant client and collection setup.

Uses local persistent file storage by default (no Docker needed).
Set QDRANT_URL to a real remote URL in .env to use a remote server instead.
"""

import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

_ROOT = Path(__file__).resolve().parent.parent


def _local_path() -> str:
    p = _ROOT / "data" / "qdrant_storage"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def get_client() -> QdrantClient:
    qdrant_url = os.environ.get("QDRANT_URL", "").strip()
    if qdrant_url and qdrant_url != "http://localhost:6333":
        api_key = os.environ.get("QDRANT_API_KEY") or None
        return QdrantClient(url=qdrant_url, api_key=api_key)
    return QdrantClient(path=_local_path())


def get_qdrant_client(url=None, api_key=None):
    if url is None:
        url = os.environ.get("QDRANT_URL", "").strip()
    if not url or url == "http://localhost:6333":
        return QdrantClient(path=_local_path())
    if api_key is None:
        api_key = os.environ.get("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client, collection_name, vector_size, distance="cosine"):
    distance_map = {"cosine": Distance.COSINE, "euclid": Distance.EUCLID, "dot": Distance.DOT}
    dist = distance_map.get((distance or "cosine").lower(), Distance.COSINE)
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=dist),
        )
        