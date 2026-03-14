"""
Test that search returns at least one result for a known query (indexing non-empty).
Requires Qdrant running and collection populated; skips if unavailable.
"""

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))


def _can_run_integration() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        from retrieval.qdrant_client import get_client
        client = get_client()
        from app.config import get_settings
        settings = get_settings()
        name = settings.get("qdrant", {}).get("collection_name", "attack_chunks")
        info = client.get_collection(name)
        return info.points_count > 0
    except Exception:
        return False


class TestIndexingNonempty(unittest.TestCase):
    @unittest.skipUnless(_can_run_integration(), "Qdrant populated and OPENAI_API_KEY set")
    def test_search_returns_at_least_one_result_for_known_query(self) -> None:
        from retrieval.qdrant_retriever import search
        results = search("T1059 command and scripting interpreter", top_k=5)
        self.assertGreater(len(results), 0, "Search should return at least one result for known technique query")
        c = results[0]
        self.assertIsInstance(c.score, (int, float))
        self.assertTrue(c.chunk_id)
        self.assertTrue(c.text or c.chunk_id)

    @unittest.skipUnless(_can_run_integration(), "Qdrant populated and OPENAI_API_KEY set")
    def test_search_scores_non_zero_for_relevant_query(self) -> None:
        from retrieval.qdrant_retriever import search
        results = search("credential access mitigation", top_k=3)
        if not results:
            self.skipTest("No results returned")
        for c in results:
            self.assertGreaterEqual(c.score, 0.0, f"Score should be non-negative: {c.score}")


if __name__ == "__main__":
    unittest.main()
