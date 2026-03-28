"""
graph/layer2_graph.py
LangGraph workflow for Layer 2.
Nodes: decompose -> run_layer1_for_each -> synthesize
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from layer2_research.decomposer import decompose as _decompose
from layer2_research.synthesizer import synthesize as _synthesize
from layer2_research.schemas import Layer2Report
from graph.layer1_graph import run_layer1


class Layer2State(TypedDict):
    question: str
    subquestions: list[str]
    subresults: list[Any]
    report: str
    confidence: float


def node_decompose(state: Layer2State) -> Layer2State:
    subquestions = _decompose(state["question"], max_subquestions=5)
    return {**state, "subquestions": subquestions, "subresults": []}


def node_run_layer1_for_each(state: Layer2State) -> Layer2State:
    subresults = []
    for sq in state["subquestions"]:
        result = run_layer1(sq)
        subresults.append(result)
    return {**state, "subresults": subresults}


def node_synthesize(state: Layer2State) -> Layer2State:
    report_text = _synthesize(
        original_question=state["question"],
        subquestions=state["subquestions"],
        subresults=state["subresults"],
    )
    total = sum(
        float(sr.confidence) if hasattr(sr, "confidence")
        else float(sr.get("confidence", 0.0))
        for sr in state["subresults"]
    )
    confidence = round(total / len(state["subresults"]), 2) if state["subresults"] else 0.0
    return {**state, "report": report_text, "confidence": confidence}


def build_layer2_graph():
    builder = StateGraph(Layer2State)
    builder.add_node("decompose",           node_decompose)
    builder.add_node("run_layer1_for_each", node_run_layer1_for_each)
    builder.add_node("synthesize",          node_synthesize)
    builder.set_entry_point("decompose")
    builder.add_edge("decompose",           "run_layer1_for_each")
    builder.add_edge("run_layer1_for_each", "synthesize")
    builder.add_edge("synthesize",          END)
    return builder.compile()


def run_layer2_graph(question: str) -> Layer2Report:
    graph = build_layer2_graph()
    initial: Layer2State = {
        "question":     question,
        "subquestions": [],
        "subresults":   [],
        "report":       "",
        "confidence":   0.0,
    }
    final = graph.invoke(initial)
    return Layer2Report(
        subquestions=final["subquestions"],
        subresults=final["subresults"],
        report=final["report"],
        confidence=final["confidence"],
    )
