"""
evaluation/run_layer1.py

Runs all Layer 1 questions from eval_questions.json through the agentic
pipeline and saves results to evaluation/results_layer1.json.

Also runs the same questions through the baseline RAG for comparison.
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

EVAL_DIR = ROOT / "evaluation"
QUESTIONS_FILE = EVAL_DIR / "eval_questions.json"
OUTPUT_FILE = EVAL_DIR / "results_layer1.json"


def run():
    from graph.layer1_graph import run_layer1
    from evaluation.baseline_rag import run_baseline

    with open(QUESTIONS_FILE) as f:
        all_questions = json.load(f)

    # Only Layer 1 questions
    questions = [q for q in all_questions if q["layer"] == 1]

    print(f"Running {len(questions)} questions through Layer 1 agentic pipeline...\n")

    results = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question'][:70]}...")

        # --- Agentic pipeline ---
        t0 = time.time()
        try:
            agentic = run_layer1(q["question"])
            agentic_result = {
                "answer": agentic.answer,
                "citations": agentic.citations,
                "confidence": agentic.confidence,
                "retries": agentic.retries,
                "chunks_used": [
                    {"technique_id": c.technique_id, "title": c.title, "score": c.score, "text": c.text}
                    for c in agentic.chunks_used
                ],
                "error": None,
            }
        except Exception as e:
            agentic_result = {"answer": "", "citations": [], "confidence": 0.0, "retries": 0, "chunks_used": [], "error": str(e)}
        agentic_time = round(time.time() - t0, 2)

        # --- Baseline pipeline ---
        t0 = time.time()
        try:
            baseline = run_baseline(q["question"])
            baseline_result = {
                "answer": baseline.answer,
                "citations": baseline.citations,
                "chunks_used": [
                    {"technique_id": c.technique_id, "title": c.title, "score": c.score, "text": c.text}
                    for c in baseline.chunks_used
                ],
                "error": None,
            }
        except Exception as e:
            baseline_result = {"answer": "", "citations": [], "chunks_used": [], "error": str(e)}
        baseline_time = round(time.time() - t0, 2)

        results.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "expected_citations": q["expected_citations"],
            "agentic": agentic_result,
            "agentic_time_s": agentic_time,
            "baseline": baseline_result,
            "baseline_time_s": baseline_time,
        })

        print(f"    Agentic  → confidence={agentic_result['confidence']:.0%}, citations={agentic_result['citations']}, time={agentic_time}s")
        print(f"    Baseline → citations={baseline_result['citations']}, time={baseline_time}s\n")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
