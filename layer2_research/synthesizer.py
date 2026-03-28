"""
layer2_research/synthesizer.py
Takes all Layer 1 sub-answers and writes a full research report.
"""

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.providers import chat


def synthesize(
    original_question: str,
    subquestions: list[str],
    subresults: list[Any],
) -> str:
    """Write a structured research report from all sub-answers."""

    # Build context from all sub-results
    context_parts = []
    for i, (sq, sr) in enumerate(zip(subquestions, subresults), start=1):
        answer = _get_answer(sr)
        citations = _get_citations(sr)
        confidence = _get_confidence(sr)
        context_parts.append(
            f"Sub-question {i}: {sq}\n"
            f"Answer: {answer}\n"
            f"Citations: {citations}\n"
            f"Confidence: {confidence:.2f}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a senior cybersecurity analyst writing a research report.

Original question: {original_question}

Sub-question results:
{context}

Write a structured report with EXACTLY these five sections
using these exact headings:

## Executive Summary
3-5 sentences summarising the key findings.

## Findings
Bullet points. Each bullet MUST include citations like [T1059].

## Recommendations
2-4 actionable bullet points for defenders with citations.

## Assumptions and Gaps
Note any sub-questions with low confidence or missing information.

## Confidence
A single decimal between 0.0 and 1.0 for overall quality.

Use only the provided evidence. Be specific."""

    return chat(prompt, temperature=0.2)


def _get_answer(result: Any) -> str:
    if hasattr(result, "answer"):
        return result.answer or "(no answer)"
    if isinstance(result, dict):
        return result.get("answer", "(no answer)")
    return str(result)


def _get_citations(result: Any) -> str:
    if hasattr(result, "citations"):
        citations = result.citations
    elif isinstance(result, dict):
        citations = result.get("citations", [])
    else:
        return "(none)"
    if not citations:
        return "(none)"
    if isinstance(citations, list):
        return ", ".join(str(c) for c in citations)
    return str(citations)


def _get_confidence(result: Any) -> float:
    if hasattr(result, "confidence"):
        val = result.confidence
    elif isinstance(result, dict):
        val = result.get("confidence", 0.5)
    else:
        return 0.5
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.5