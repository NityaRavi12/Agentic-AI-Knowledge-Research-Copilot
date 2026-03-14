"""
Test that STIX extraction produced docs.jsonl with at least one doc.
Prevents breaking ingestion later.
"""

import json
import unittest
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = ROOT / "data" / "processed" / "docs.jsonl"


class TestStixLoads(unittest.TestCase):
    def test_docs_jsonl_exists(self) -> None:
        self.assertTrue(DOCS_PATH.exists(), f"Expected {DOCS_PATH} to exist. Run ingestion/extract_docs.py first.")

    def test_docs_jsonl_has_docs(self) -> None:
        if not DOCS_PATH.exists():
            self.skipTest("docs.jsonl not found")
        count = 0
        required = {"doc_id", "technique_id", "title", "section", "url", "text"}
        with open(DOCS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                self.assertGreater(len(doc), 0, "Each line must be a non-empty JSON object")
                missing = required - set(doc.keys())
                self.assertEqual(missing, set(), f"Doc missing keys: {missing}")
                count += 1
        self.assertGreater(count, 0, "docs.jsonl must contain at least one doc")


if __name__ == "__main__":
    unittest.main()
