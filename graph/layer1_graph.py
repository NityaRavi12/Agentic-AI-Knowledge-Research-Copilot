"""
Layer 1 — Agentic RAG pipeline using LangGraph.

Flow:
retrieve -> evaluate_retrieval -> generate -> evaluate_answer
         -> revise_answer (if needed) -> finalize

Retries retrieval once if evidence is weak.
Revises the answer once if grounding/completeness is weak.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.qdrant_retriever import search, EvidenceChunk
from app.config import get_settings
from llm.providers import chat
from layer1_qa.evaluators import (
    evaluate_retrieval,
    evaluate_answer,
    revise_answer,
)


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
    retrieval_query: str
    chunks: list[EvidenceChunk]
    approved_chunks: list[EvidenceChunk]
    answer: str
    citations: list[str]
    confidence: float
    retries: int
    revision_count: int
    retrieval_eval: dict[str, Any]
    answer_eval: dict[str, Any]
    settings: dict[str, Any]


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_citations(answer: str) -> list[str]:
    citations: list[str] = []
    for match in re.findall(r"\[T\d+(?:\.\d+)?\]", answer):
        cid = match.strip("[]")
        if cid not in citations:
            citations.append(cid)
    return citations


def _build_evidence_block(chunks: list[EvidenceChunk], limit: int = 8) -> str:
    selected = chunks[:limit]
    parts: list[str] = []

    for idx, chunk in enumerate(selected, start=1):
        technique_id = getattr(chunk, "technique_id", "N/A")
        title = getattr(chunk, "title", "Untitled")
        text = getattr(chunk, "text", "")
        parts.append(
            f"[Evidence {idx}] [{technique_id}] {title}\n"
            f"{text[:700]}"
        )

    return "\n\n".join(parts)


def _compute_confidence(
    retrieval_eval: dict[str, Any],
    answer_eval: dict[str, Any],
) -> float:
    relevance = float(retrieval_eval.get("relevance_score", 0.0))
    sufficiency = float(retrieval_eval.get("sufficiency_score", 0.0))
    groundedness = float(answer_eval.get("groundedness_score", 0.0))
    completeness = float(answer_eval.get("completeness_score", 0.0))

    score = (
        0.25 * relevance
        + 0.25 * sufficiency
        + 0.30 * groundedness
        + 0.20 * completeness
    )
    return round(max(0.0, min(1.0, score)), 2)


# ── Node 1: Retrieve ───────────────────────────────────────────────────────

def node_retrieve(state: Layer1State) -> Layer1State:
    settings = state["settings"]
    top_k = settings.get("qdrant", {}).get("top_k", 12)
    query = state.get("retrieval_query") or state["question"]

    chunks = search(query, top_k=top_k)

    return {
        **state,
        "chunks": chunks,
    }


# ── Node 2: Evaluate retrieval ─────────────────────────────────────────────

def node_evaluate_retrieval(state: Layer1State) -> Layer1State:
    settings = state["settings"]
    layer1 = settings.get("layer1", {})

    min_score = layer1.get("min_evidence_score", 0.40)
    temperature = layer1.get("retrieval_eval_temperature", 0.0)

    result = evaluate_retrieval(
        state["question"],
        state["chunks"],
        min_score=min_score,
        temperature=temperature,
    )

    return {
        **state,
        "approved_chunks": result.get("approved_chunks", []),
        "retrieval_eval": result,
    }


# ── Node 3: Rewrite query / bump retry ─────────────────────────────────────

def node_prepare_retry(state: Layer1State) -> Layer1State:
    retrieval_eval = state.get("retrieval_eval", {})
    rewritten_query = retrieval_eval.get("rewritten_query", "").strip()

    if not rewritten_query:
        rewritten_query = state["question"]

    return {
        **state,
        "retries": state["retries"] + 1,
        "retrieval_query": rewritten_query,
    }


# ── Node 4: Generate ───────────────────────────────────────────────────────

def node_generate(state: Layer1State) -> Layer1State:
    settings = state["settings"]
    layer1 = settings.get("layer1", {})
    temperature = layer1.get("temperature", 0.1)

    chunks = state.get("approved_chunks") or state.get("chunks") or []
    evidence = _build_evidence_block(chunks, limit=8)

    prompt = f"""You are a cybersecurity analyst with access to MITRE ATT&CK evidence.

Answer the question using ONLY the evidence provided below.
Do not invent facts.
If the evidence is incomplete, say so clearly.
Cite techniques inline like this: [T1059]

Question:
{state["question"]}

Evidence:
{evidence}

Write a clear, concise answer with inline citations where supported.
"""

    answer = chat(prompt, temperature=temperature).strip()
    citations = _extract_citations(answer)

    return {
        **state,
        "answer": answer,
        "citations": citations,
    }


# ── Node 5: Evaluate answer ────────────────────────────────────────────────

def node_evaluate_answer(state: Layer1State) -> Layer1State:
    settings = state["settings"]
    layer1 = settings.get("layer1", {})
    temperature = layer1.get("answer_eval_temperature", 0.0)

    chunks = state.get("approved_chunks") or state.get("chunks") or []

    result = evaluate_answer(
        state["question"],
        state["answer"],
        chunks,
        temperature=temperature,
    )

    confidence = _compute_confidence(
        state.get("retrieval_eval", {}),
        result,
    )

    return {
        **state,
        "answer_eval": result,
        "confidence": confidence,
    }


# ── Node 6: Revise answer ──────────────────────────────────────────────────

def node_revise_answer(state: Layer1State) -> Layer1State:
    settings = state["settings"]
    layer1 = settings.get("layer1", {})
    temperature = layer1.get("temperature", 0.1)

    chunks = state.get("approved_chunks") or state.get("chunks") or []
    feedback = state.get("answer_eval", {})

    revised = revise_answer(
        state["question"],
        state["answer"],
        chunks,
        feedback,
        temperature=temperature,
    ).strip()

    citations = _extract_citations(revised)

    return {
        **state,
        "answer": revised,
        "citations": citations,
        "revision_count": state["revision_count"] + 1,
    }


# ── Node 7: Finalize ───────────────────────────────────────────────────────

def node_finalize(state: Layer1State) -> Layer1State:
    return state


# ── Routing ────────────────────────────────────────────────────────────────

def route_after_retrieval_eval(state: Layer1State) -> str:
    settings = state["settings"]
    layer1 = settings.get("layer1", {})
    max_retries = layer1.get("max_retries", 1)

    retrieval_eval = state.get("retrieval_eval", {})
    needs_retry = bool(retrieval_eval.get("needs_retry", False))

    if needs_retry and state["retries"] < max_retries:
        return "retry"

    return "generate"


def route_after_answer_eval(state: Layer1State) -> str:
    settings = state["settings"]
    layer1 = settings.get("layer1", {})
    max_revision_rounds = layer1.get("max_revision_rounds", 1)

    answer_eval = state.get("answer_eval", {})
    revision_needed = bool(answer_eval.get("revision_needed", False))

    if revision_needed and state["revision_count"] < max_revision_rounds:
        return "revise"

    return "finalize"


# ── Build the graph ────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(Layer1State)

    builder.add_node("retrieve", node_retrieve)
    builder.add_node("evaluate_retrieval", node_evaluate_retrieval)
    builder.add_node("prepare_retry", node_prepare_retry)
    builder.add_node("generate", node_generate)
    builder.add_node("evaluate_answer", node_evaluate_answer)
    builder.add_node("revise_answer", node_revise_answer)
    builder.add_node("finalize", node_finalize)

    builder.set_entry_point("retrieve")

    builder.add_edge("retrieve", "evaluate_retrieval")

    builder.add_conditional_edges(
        "evaluate_retrieval",
        route_after_retrieval_eval,
        {
            "retry": "prepare_retry",
            "generate": "generate",
        },
    )

    builder.add_edge("prepare_retry", "retrieve")
    builder.add_edge("generate", "evaluate_answer")

    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_answer_eval,
        {
            "revise": "revise_answer",
            "finalize": "finalize",
        },
    )

    builder.add_edge("revise_answer", "evaluate_answer")
    builder.add_edge("finalize", END)

    return builder.compile()


# ── Public function ────────────────────────────────────────────────────────

def run_layer1(question: str) -> Layer1Result:
    settings = get_settings()
    graph = build_graph()

    initial: Layer1State = {
        "question": question,
        "retrieval_query": question,
        "chunks": [],
        "approved_chunks": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "retries": 0,
        "revision_count": 0,
        "retrieval_eval": {},
        "answer_eval": {},
        "settings": settings,
    }

    final = graph.invoke(initial)

    chunks_used = final.get("approved_chunks") or final.get("chunks") or []

    return Layer1Result(
        question=final["question"],
        answer=final["answer"],
        citations=final["citations"],
        confidence=final["confidence"],
        chunks_used=chunks_used,
        retries=final["retries"],
    )