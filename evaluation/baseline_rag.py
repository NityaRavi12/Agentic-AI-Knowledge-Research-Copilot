"""
evaluation/baseline_rag.py

Naive baseline RAG pipeline for comparison against the agentic system.
No planning, no evaluation, no reflection - just retrieve and generate.
This is the "dumb" RAG that the agentic system is supposed to beat.
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from retrieval.qdrant_retriever import search
from llm.providers import chat


@dataclass
class BaselineResult:
    question: str
    answer: str
    citations: list[str]
    chunks_used: list


def run_baseline(question: str, top_k: int = 12) -> BaselineResult:
    """
    Naive RAG: retrieve top-k chunks, generate answer directly.
    No planning, no filtering, no reflection, no retry.
    """
    # Step 1: Retrieve directly using the raw question (no planning)
    chunks = search(question, top_k=top_k)

    # Step 2: Build evidence block from top 8 chunks
    evidence = "\n\n".join(
        f"[{c.technique_id}] {c.title}:\n{c.text[:400]}"
        for c in chunks[:8]
    )

    # Step 3: Generate answer with no special instructions
    prompt = f"""You are a cybersecurity assistant. Answer the following question
using the evidence provided. Cite techniques inline using bracket notation like [T1059].

Evidence:
{evidence}

Question: {question}

Answer:"""

    answer = chat(prompt, temperature=0.1)

    # Extract citations
    import re
    citations = []
    for match in re.findall(r'\[T\d+(?:\.\d+)?\]', answer):
        cid = match.strip("[]")
        if cid not in citations:
            citations.append(cid)

    return BaselineResult(
        question=question,
        answer=answer,
        citations=citations,
        chunks_used=chunks[:8],
    )


if __name__ == "__main__":
    result = run_baseline("How do adversaries use PowerShell for execution?")
    print(f"Answer: {result.answer[:300]}")
    print(f"Citations: {result.citations}")
