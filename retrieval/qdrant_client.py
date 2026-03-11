"""
Qdrant client and collection setup.
"""

import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def get_client() -> QdrantClient:
    """Return a Qdrant client using QDRANT_URL and optional QDRANT_API_KEY."""
    return get_qdrant_client(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
    )


def get_qdrant_client(url: str | None = None, api_key: str | None = None) -> QdrantClient:
    """Return a Qdrant client. Uses env QDRANT_URL / QDRANT_API_KEY when url/api_key are None."""
    if url is None:
        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    if api_key is None:
        api_key = os.environ.get("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: str = "cosine",
) -> None:
    """
    Create the collection if it does not exist. If it exists, ensure vector config
    matches (no-op; for mismatched config, caller must delete and re-create).
    """
    distance_map = {
        "cosine": Distance.COSINE,
        "euclid": Distance.EUCLID,
        "dot": Distance.DOT,
    }
    dist = distance_map.get((distance or "cosine").lower(), Distance.COSINE)
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=dist),
        )
