#!/usr/bin/env python3
"""
CLI to run and verify retrieval: takes a query, prints top results with score, chunk_id, snippet.
Usage: python scripts/search.py "your query"
"""

import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.qdrant_retriever import search

SNIPPET_LEN = 200


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/search.py \"<query>\"", file=sys.stderr)
        sys.exit(1)
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: python scripts/search.py \"<query>\"", file=sys.stderr)
        sys.exit(1)

    chunks = search(query)
    print(f"Query: {query}\n")
    print(f"Top {len(chunks)} result(s):\n")
    for i, c in enumerate(chunks, 1):
        raw = c.text.replace("\n", " ").strip()
        snippet = (raw[:SNIPPET_LEN] + "...") if len(raw) > SNIPPET_LEN else raw
        print(f"  [{i}] score={c.score:.4f}  chunk_id={c.chunk_id}")
        print(f"      technique={c.technique_id}  title={c.title}")
        print(f"      snippet: {snippet}")
        print()


if __name__ == "__main__":
    main()
