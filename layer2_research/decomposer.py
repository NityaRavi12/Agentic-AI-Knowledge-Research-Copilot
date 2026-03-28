"""
layer2_research/decomposer.py
Breaks a complex question into up to 5 simpler sub-questions.
"""

import os
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.providers import chat


def decompose(question: str, max_subquestions: int = 5) -> list[str]:
    """Break a complex question into smaller sub-questions."""

    prompt = f"""You are a cybersecurity research assistant.
Break the following complex question into at most {max_subquestions} smaller,
specific, factual sub-questions that can each be answered by searching
the MITRE ATT&CK knowledge base.

Rules:
- Return ONLY a JSON array of strings like ["q1", "q2", "q3"]
- No open-ended or opinion questions
- Each sub-question must be specific and factual
- No explanation, no extra text — just the JSON array

Question: {question}"""

    raw = chat(prompt, temperature=0.2)

    return _parse(raw, max_subquestions)


def _parse(raw: str, max_subquestions: int) -> list[str]:
    """Safely parse the LLM response into a list of strings."""

    # Remove markdown code fences if present
    clean = raw.strip()
    clean = clean.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            result = [str(q).strip() for q in parsed if str(q).strip()]
            return result[:max_subquestions]
    except json.JSONDecodeError:
        pass

    # Fallback: split by newlines
    lines = [
        line.strip().lstrip("-*•0123456789.)").strip()
        for line in raw.splitlines()
        if line.strip()
    ]
    return [l for l in lines if l][:max_subquestions]