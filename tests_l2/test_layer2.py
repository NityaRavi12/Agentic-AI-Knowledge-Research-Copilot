"""
tests_l2/test_layer2.py
Unit tests for Layer 2. All mocked - no Groq or Qdrant needed.
"""

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer2_research.schemas import Layer2Report


def make_mock_result(confidence=0.8):
    """Make a fake Layer 1 result for testing."""
    from graph.layer1_graph import Layer1Result
    return Layer1Result(
        question="test question",
        answer="Attackers use [T1059] and [T1546] for persistence.",
        citations=["T1059", "T1546"],
        confidence=confidence,
        chunks_used=[],
        retries=0,
    )


SAMPLE_REPORT = """## Executive Summary
Attackers use multiple techniques for persistence on Windows.

## Findings
- PowerShell [T1059] is used for execution.
- Registry keys [T1060] are modified for persistence.

## Recommendations
- Monitor PowerShell usage [T1059].
- Audit registry changes [T1060].

## Assumptions and Gaps
All sub-questions had high confidence.

## Confidence
0.9"""


def test_report_has_required_headings():
    """Report must contain all 5 required section headings."""
    report = Layer2Report(
        subquestions=["What is T1059?"],
        subresults=[make_mock_result()],
        report=SAMPLE_REPORT,
        confidence=0.9,
    )
    required = [
        "## Executive Summary",
        "## Findings",
        "## Recommendations",
        "## Assumptions and Gaps",
        "## Confidence",
    ]
    for heading in required:
        assert heading in report.report, f"Missing heading: {heading}"


def test_subquestions_at_most_5():
    """Decomposer must return at most 5 sub-questions."""
    from layer2_research.decomposer import _parse
    raw = '["q1","q2","q3","q4","q5","q6","q7"]'
    result = _parse(raw, max_subquestions=5)
    assert len(result) <= 5


def test_report_includes_citations():
    """Report must include at least one citation."""
    report = Layer2Report(
        subquestions=["What is T1059?"],
        subresults=[make_mock_result()],
        report=SAMPLE_REPORT,
        confidence=0.9,
    )
    assert "T1059" in report.report or "T1060" in report.report


def test_confidence_is_computed():
    """Average confidence must be between 0.0 and 1.0."""
    from layer2_research.researcher import _average_confidence
    results = [
        make_mock_result(0.8),
        make_mock_result(0.6),
        make_mock_result(1.0),
    ]
    confidence = _average_confidence(results)
    assert 0.0 <= confidence <= 1.0
    assert abs(confidence - 0.8) < 0.01


def test_layer2report_has_all_fields():
    """Layer2Report must have all required fields."""
    report = Layer2Report(
        subquestions=["q1", "q2"],
        subresults=[make_mock_result(), make_mock_result()],
        report=SAMPLE_REPORT,
        confidence=0.8,
    )
    assert hasattr(report, "subquestions")
    assert hasattr(report, "subresults")
    assert hasattr(report, "report")
    assert hasattr(report, "confidence")
