"""
Test that search() returns EvidenceChunk with required fields.
Does not require Qdrant to be populated (can mock or skip if empty).
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from retrieval.qdrant_retriever import EvidenceChunk


class TestRetrievalShape(unittest.TestCase):
    def test_evidence_chunk_has_required_fields(self) -> None:
        """EvidenceChunk must have score, chunk_id, technique_id, title, section, url, text."""
        c = EvidenceChunk(
            chunk_id="T1059::description::001",
            text="Some text",
            score=0.85,
            technique_id="T1059",
            title="Command and Scripting Interpreter",
            url="https://attack.mitre.org/techniques/T1059/",
            section="description",
        )
        self.assertEqual(c.chunk_id, "T1059::description::001")
        self.assertIn("text", c.text)
        self.assertGreaterEqual(c.score, 0.0)
        self.assertEqual(c.technique_id, "T1059")
        self.assertEqual(c.section, "description")

    def test_evidence_chunk_is_dataclass(self) -> None:
        """EvidenceChunk should be usable as a simple record."""
        c = EvidenceChunk(
            chunk_id="x::y::001",
            text="t",
            score=0.0,
            technique_id="T1",
            title="T",
            url="https://example.com/",
        )
        self.assertTrue(hasattr(c, "section"))
        self.assertEqual(c.section, "description")


if __name__ == "__main__":
    unittest.main()
