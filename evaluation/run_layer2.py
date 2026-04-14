"""
evaluation/run_layer2.py

Runs all Layer 2 questions from eval_questions.json through the full
research pipeline and saves results to evaluation/results_layer2.json.
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
OUTPUT_FILE = EVAL_DIR / "results_layer2.json"


def run():
    from layer2_research.researcher import run_layer2

    with open(QUESTIONS_FILE) as f:
        all_questions = json.load(f)

    # Only Layer 2 questions
    questions = [q for q in all_questions if q["layer"] == 2]

    print(f"Running {len(questions)} questions through Layer 2 research pipeline...\n")
    print("Note: Each question spawns multiple Layer 1 calls. This will take a few minutes.\n")

    results = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question'][:70]}...")

        t0 = time.time()
        try:
            report = run_layer2(q["question"], max_subquestions=4)
            result = {
                "report": report.report,
                "subquestions": report.subquestions,
                "confidence": report.confidence,
                "subresults": [
                    {
                        "question": sr.question,
                        "answer": sr.answer,
                        "citations": sr.citations,
                        "confidence": sr.confidence,
                    }
                    for sr in report.subresults
                ],
                "error": None,
            }
        except Exception as e:
            result = {"report": "", "subquestions": [], "confidence": 0.0, "subresults": [], "error": str(e)}

        elapsed = round(time.time() - t0, 2)

        results.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "expected_citations": q["expected_citations"],
            "result": result,
            "time_s": elapsed,
        })

        print(f"    Sub-questions: {len(result['subquestions'])}")
        print(f"    Confidence: {result['confidence']:.0%}")
        print(f"    Time: {elapsed}s\n")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
