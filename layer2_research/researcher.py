"""
layer2_research/pipeline.py
Orchestrates the full Layer 2 pipeline:
decompose -> run Layer 1 for each -> synthesize -> Layer2Report
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer2_research.decomposer import decompose
from layer2_research.synthesizer import synthesize
from layer2_research.schemas import Layer2Report
from graph.layer1_graph import run_layer1


def run_layer2(question: str, max_subquestions: int = 5) -> Layer2Report:
    """Run the full Layer 2 research pipeline."""

    print(f"\n[Layer 2] Breaking question into sub-questions...")
    subquestions = decompose(question, max_subquestions=max_subquestions)
    print(f"[Layer 2] Got {len(subquestions)} sub-questions")

    subresults = []
    for i, sq in enumerate(subquestions, start=1):
        print(f"[Layer 2] Running Layer 1 on sub-question {i}/{len(subquestions)}: {sq}")
        result = run_layer1(sq)
        subresults.append(result)

    print(f"[Layer 2] Synthesizing final report...")
    report_text = synthesize(question, subquestions, subresults)

    confidence = _average_confidence(subresults)

    return Layer2Report(
        subquestions=subquestions,
        subresults=subresults,
        report=report_text,
        confidence=confidence,
    )


def _average_confidence(subresults: list) -> float:
    """Compute average confidence across all sub-results."""
    if not subresults:
        return 0.0
    total = 0.0
    for sr in subresults:
        if hasattr(sr, "confidence"):
            total += float(sr.confidence)
        elif isinstance(sr, dict):
            total += float(sr.get("confidence", 0.0))
    return round(total / len(subresults), 2)
