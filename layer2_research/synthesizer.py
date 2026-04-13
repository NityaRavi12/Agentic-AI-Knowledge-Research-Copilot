from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from llm.providers import chat

def _get_answer(result: Any) -> str:
    if hasattr(result, "answer") and result.answer:
        return result.answer
    if isinstance(result, dict):
        return result.get("answer", "(no answer)")
    return "(no answer)"

def _get_citations(result: Any) -> str:
    if hasattr(result, "citations"):
        return ", ".join(str(c) for c in result.citations) if result.citations else "(none)"
    if isinstance(result, dict):
        return ", ".join(str(c) for c in result.get("citations", []))
    return "(none)"

def _get_confidence(result: Any) -> float:
    try:
        if hasattr(result, "confidence"):
            return float(result.confidence)
        if isinstance(result, dict):
            return float(result.get("confidence", 0.5))
    except Exception:
        pass
    return 0.5

def synthesize(original_question: str, subquestions: list[str], subresults: list[Any]) -> str:
    context_parts = []
    for i, (sq, sr) in enumerate(zip(subquestions, subresults), start=1):
        context_parts.append(
            f"Sub-question {i}: {sq}\n"
            f"Answer: {_get_answer(sr)}\n"
            f"Citations: {_get_citations(sr)}\n"
            f"Confidence: {_get_confidence(sr):.2f}"
        )
    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""You are a senior cybersecurity analyst writing a research report.
Original question: {original_question}

Sub-question results:
{context}

Write a structured report with EXACTLY these five sections:

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
