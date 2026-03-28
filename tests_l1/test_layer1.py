"""
tests_l1/test_layer1.py
Unit tests for Layer 1. All mocked — no Groq or Qdrant needed.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_mock_chunk(score=0.8):
    from retrieval.qdrant_retriever import EvidenceChunk
    return EvidenceChunk(
        chunk_id="T1059::description::001",
        text="Adversaries may abuse PowerShell to execute commands.",
        score=score,
        technique_id="T1059",
        title="Command and Scripting Interpreter",
        url="https://attack.mitre.org/techniques/T1059/",
    )


def test_run_layer1_returns_layer1result():
    """run_layer1() must return a Layer1Result object."""
    from graph.layer1_graph import Layer1Result
    mock_chunks = [make_mock_chunk(), make_mock_chunk()]
    with patch("graph.layer1_graph.search", return_value=mock_chunks), \
         patch("graph.layer1_graph.chat", return_value="PowerShell [T1059] is used by attackers to run scripts."):
        from graph.layer1_graph import run_layer1
        result = run_layer1("How is PowerShell abused?")
        assert isinstance(result, Layer1Result)


def test_run_layer1_has_answer():
    """Layer1Result must have a non-empty answer."""
    mock_chunks = [make_mock_chunk(), make_mock_chunk()]
    with patch("graph.layer1_graph.search", return_value=mock_chunks), \
         patch("graph.layer1_graph.chat", return_value="PowerShell [T1059] is used by attackers."):
        from graph.layer1_graph import run_layer1
        result = run_layer1("How is PowerShell abused?")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0


def test_run_layer1_has_citations():
    """Layer1Result.citations must be a list."""
    mock_chunks = [make_mock_chunk(), make_mock_chunk()]
    with patch("graph.layer1_graph.search", return_value=mock_chunks), \
         patch("graph.layer1_graph.chat", return_value="Attackers use [T1059] and [T1546]."):
        from graph.layer1_graph import run_layer1
        result = run_layer1("How is PowerShell abused?")
        assert isinstance(result.citations, list)


def test_run_layer1_confidence_in_range():
    """Confidence must be between 0.0 and 1.0."""
    mock_chunks = [make_mock_chunk(), make_mock_chunk()]
    with patch("graph.layer1_graph.search", return_value=mock_chunks), \
         patch("graph.layer1_graph.chat", return_value="Attackers use [T1059] extensively for execution."):
        from graph.layer1_graph import run_layer1
        result = run_layer1("How is PowerShell abused?")
        assert 0.0 <= result.confidence <= 1.0
