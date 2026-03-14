"""
Chunk documents and index into Qdrant. Writes chunks.jsonl and upserts vectors.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Project root
ROOT = Path(__file__).resolve().parent.parent


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
        "qdrant": {"collection_name": "attack_chunks", "distance": "cosine"},
        "chunking": {"chunk_size": 1200, "overlap": 150},
        "models": {"embed_dim": 1536},
    }


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks. Returns list of chunk strings."""
    if not text or chunk_size <= 0:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def docs_to_chunks(docs: list[dict]) -> list[dict]:
    """
    Convert doc records to chunk records. Each chunk has chunk_id, technique_id,
    title, section, url, text. chunk_id format: technique_id::section::NNN.
    """
    settings = _load_settings()
    chunk_size = settings.get("chunking", {}).get("chunk_size", 1200)
    overlap = settings.get("chunking", {}).get("overlap", 150)
    chunk_records = []
    for doc in docs:
        text = doc.get("text") or ""
        text_chunks = chunk_text(text, chunk_size, overlap)
        technique_id = doc.get("technique_id", "")
        title = doc.get("title", "")
        section = doc.get("section", "description")
        url = doc.get("url", "")
        for i, chunk_text_str in enumerate(text_chunks):
            segment = f"{i + 1:03d}"
            chunk_id = f"{technique_id}::{section}::{segment}"
            chunk_records.append({
                "chunk_id": chunk_id,
                "technique_id": technique_id,
                "title": title,
                "section": section,
                "url": url,
                "text": chunk_text_str,
            })
    return chunk_records


def index_chunks_qdrant(chunks: list[dict]) -> None:
    """
    Create Qdrant collection (embed_dim, distance), embed chunk texts,
    upsert points with payload (text + metadata). Uses llm.providers.embed_texts.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    settings = _load_settings()
    qdrant_cfg = settings.get("qdrant", {})
    collection_name = qdrant_cfg.get("collection_name", "attack_chunks")
    distance_name = (qdrant_cfg.get("distance") or "cosine").upper()
    distance = Distance.COSINE if distance_name == "COSINE" else Distance.EUCLID
    embed_dim = settings.get("models", {}).get("embed_dim", 1536)

    # Use local persistent storage — no Docker or server needed.
    # Data is saved to data/qdrant_storage/ and survives between runs.
    # To use a remote Qdrant server instead, set QDRANT_URL in .env.
    qdrant_url = __import__("os").environ.get("QDRANT_URL", "").strip()
    if qdrant_url and qdrant_url != "http://localhost:6333":
        api_key = __import__("os").environ.get("QDRANT_API_KEY") or None
        client = QdrantClient(url=qdrant_url, api_key=api_key or None)
        print(f"[qdrant] Using remote Qdrant at {qdrant_url}")
    else:
        import pathlib
        local_path = str(pathlib.Path(__file__).resolve().parent.parent / "data" / "qdrant_storage")
        pathlib.Path(local_path).mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=local_path)
        print(f"[qdrant] Using local persistent storage at {local_path}")

    # Create collection; recreate if exists to keep schema in sync (optional: check first)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embed_dim, distance=distance),
    )

    if not chunks:
        return

    from llm.providers import embed_texts
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    points = [
        PointStruct(
            id=i,
            vector=vec,
            payload={
                "chunk_id": c["chunk_id"],
                "technique_id": c["technique_id"],
                "title": c["title"],
                "section": c["section"],
                "url": c["url"],
                "text": c["text"],
            },
        )
        for i, (c, vec) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=collection_name, points=points)


def main() -> None:
    """Read docs.jsonl, build chunks, write chunks.jsonl, index into Qdrant."""
    settings = _load_settings()
    docs_path = ROOT / "data" / "processed" / "docs.jsonl"
    chunks_path = ROOT / "data" / "processed" / "chunks.jsonl"
    if len(sys.argv) >= 2:
        docs_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        chunks_path = Path(sys.argv[2])

    if not docs_path.exists():
        print(f"Docs not found: {docs_path}. Run extract_docs first.")
        sys.exit(1)

    docs = []
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    chunks = docs_to_chunks(docs)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for rec in chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks to {chunks_path}")

    index_chunks_qdrant(chunks)
    print("Indexed chunks into Qdrant.")


if __name__ == "__main__":
    main()