#!/usr/bin/env python3
"""
Sanity check: confirm Qdrant collection is populated and payload has required keys.
Run: python scripts/check_qdrant.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_PAYLOAD_KEYS = {"chunk_id", "technique_id", "title", "section", "url", "text"}


def main() -> int:
    from retrieval.qdrant_client import get_client

    try:
        client = get_client()
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return 1

    try:
        from app.config import get_settings
        settings = get_settings()
        collection_name = settings.get("qdrant", {}).get("collection_name", "attack_chunks")
    except Exception:
        import yaml
        collection_name = "attack_chunks"
        config_path = ROOT / "configs" / "settings.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                collection_name = cfg.get("qdrant", {}).get("collection_name", collection_name)

    try:
        info = client.get_collection(collection_name)
    except Exception as e:
        print(f"Collection {collection_name!r} not found or error: {e}")
        return 1

    count = info.points_count
    print(f"Collection: {collection_name}")
    print(f"Points count: {count}")

    if count == 0:
        print("WARNING: Collection is empty. Run ingestion/chunk_and_index.py first.")
        return 0

    # Sample first few points and verify payload keys
    result, _ = client.scroll(collection_name=collection_name, limit=5, with_payload=True, with_vectors=False)
    missing_keys_per_point = []
    for pt in result:
        payload = pt.payload or {}
        missing = REQUIRED_PAYLOAD_KEYS - set(payload.keys())
        if missing:
            missing_keys_per_point.append((pt.id, missing))

    if missing_keys_per_point:
        print(f"FAIL: Some points missing payload keys: {missing_keys_per_point[:3]}")
        return 1
    print("Payload check: all sampled points have chunk_id, technique_id, title, section, url, text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
