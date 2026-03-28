"""
app/graph.py
Layer 1 — Agentic RAG pipeline using LangGraph.
search → evaluate → generate → reflect → (retry once if needed)
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from retrieval.qdrant_retriever import search, EvidenceChunk
from llm.providers import chat
from app.config import get_settings


# ── Output schema ──────────────────────────────────────────────────────────

@dataclass
class Layer1Result:
    question: str
    answer: str
    citations: list[str]
    confidence: float
    chunks_used: list[EvidenceChunk]
    retries: int = 0


# ── LangGraph state ────────────────────────────────────────────────────────

class Layer1State(TypedDict):
    question: str
    chunks: list[EvidenceChunk]
    answer: str
    citations: list[str]
    confidence: float
    retries: int
    settings: dict[str, Any]


# ── Node 1: Retrieve ───────────────────────────────────────────────────────

def node_retrieve(state: Layer1State) -> Layer1State:
    settings = state["settings"]
    top_k = settings.get("qdrant", {}).get("top_k", 12)
    chunks = search(state["question"], top_k=top_k)
    return {**state, "chunks": chunks}


# ── Node 2: Evaluate ───────────────────────────────────────────────────────

def node_evaluate(state: Layer1State) -> Layer1State:
    settings = state["settings"]
    min_score = settings.get("layer1", {}).get("min_evidence_score", 0.40)
    min_chunks = settings.get("layer1", {}).get("min_evidence_chunks", 2)
    good = [c for c in state["chunks"] if c.score >= min_score]
    if len(good) < min_chunks:
        return {**state, "confidence": 0.0}
    return {**state, "confidence": 0.5}


# ── Node 3: Generate ───────────────────────────────────────────────────────

def node_generate(state: Layer1State) -> Layer1State:
    settings = state["settings"]
    temperature = settings.get("layer1", {}).get("temperature", 0.1)
    chunks = state["chunks"][:8]

    evidence = "\n\n".join(
        f"[{c.technique_id}] {c.title}:\n{c.text[:400]}"
        for c in chunks
    )

    prompt = f"""You are a cybersecurity analyst with access to the MITRE ATT&CK knowledge base.
Answer the question below using ONLY the evidence provided.
Cite each technique you use inline like this: [T1059]

Evidence:
{evidence}

Question: {state["question"]}

Write a clear, concise answer with inline citations."""

    answer = chat(prompt, temperature=temperature)

    citations = []
    import re
    for match in re.findall(r'\[T\d+(?:\.\d+)?\]', answer):
        cid = match.strip("[]")
        if cid not in citations:
            citations.append(cid)

    return {**state, "answer": answer, "citations": citations}


# ── Node 4: Reflect ────────────────────────────────────────────────────────

def node_reflect(state: Layer1State) -> Layer1State:
    answer = state["answer"]
    citations = state["citations"]

    score = 0.0
    if len(answer) > 100:
        score += 0.4
    if len(citations) >= 1:
        score += 0.4
    if len(citations) >= 2:
        score += 0.2

    return {**state, "confidence": round(score, 2)}


# ── Conditional edge: should we retry? ────────────────────────────────────

def should_retry(state: Layer1State) -> str:
    settings = state["settings"]
    max_retries = settings.get("layer1", {}).get("max_retries", 1)
    threshold = 0.5

    if state["confidence"] < threshold and state["retries"] < max_retries:
        return "retrieve"
    return "end"


# ── Retry bump node ────────────────────────────────────────────────────────

def node_increment_retry(state: Layer1State) -> Layer1State:
    return {**state, "retries": state["retries"] + 1}


# ── Build the graph ────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(Layer1State)

    builder.add_node("retrieve",       node_retrieve)
    builder.add_node("evaluate",       node_evaluate)
    builder.add_node("generate",       node_generate)
    builder.add_node("reflect",        node_reflect)
    builder.add_node("increment_retry",node_increment_retry)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve",  "evaluate")
    builder.add_edge("evaluate",  "generate")
    builder.add_edge("generate",  "reflect")
    builder.add_conditional_edges("reflect", should_retry, {
        "retrieve": "increment_retry",
        "end":      END,
    })
    builder.add_edge("increment_retry", "retrieve")

    return builder.compile()


# ── Public function ────────────────────────────────────────────────────────

def run_layer1(question: str) -> Layer1Result:
    settings = get_settings()
    graph = build_graph()

    initial: Layer1State = {
        "question":  question,
        "chunks":    [],
        "answer":    "",
        "citations": [],
        "confidence": 0.0,
        "retries":   0,
        "settings":  settings,
    }

    final = graph.invoke(initial)

    return Layer1Result(
        question=final["question"],
        answer=final["answer"],
        citations=final["citations"],
        confidence=final["confidence"],
        chunks_used=final["chunks"],
        retries=final["retries"],
    )