"""
evaluation/compare.py

Reads metrics_ragas.json and prints a clean comparison table of
agentic pipeline vs baseline RAG across all RAGAS metrics.

Run this after metrics.py has completed.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "evaluation"
METRICS_FILE = EVAL_DIR / "metrics_ragas.json"
LAYER1_FILE = EVAL_DIR / "results_layer1.json"
LAYER2_FILE = EVAL_DIR / "results_layer2.json"


def print_table(rows: list[tuple], headers: list[str]):
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def run():
    if not METRICS_FILE.exists():
        print("metrics_ragas.json not found. Run metrics.py first.")
        return

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    agentic = metrics.get("agentic", {})
    baseline = metrics.get("baseline", {})

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS: Agentic RAG vs Baseline RAG")
    print("=" * 60 + "\n")

    # --- RAGAS metrics comparison ---
    ragas_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("RAGAS Metrics (Layer 1):")
    rows = []
    for key in ragas_keys:
        a_val = agentic.get(key, "N/A")
        b_val = baseline.get(key, "N/A")
        if isinstance(a_val, float) and isinstance(b_val, float):
            delta = round(a_val - b_val, 4)
            delta_str = f"+{delta}" if delta >= 0 else str(delta)
        else:
            delta_str = "N/A"
        rows.append((key, str(a_val), str(b_val), delta_str))

    print_table(rows, ["Metric", "Agentic", "Baseline", "Delta"])

    # --- Citation metrics ---
    print("\nCitation Metrics (Layer 1):")
    cit_rows = [
        ("citation_precision", str(agentic.get("citation_precision", "N/A")), str(baseline.get("citation_precision", "N/A"))),
        ("citation_recall",    str(agentic.get("citation_recall", "N/A")),    str(baseline.get("citation_recall", "N/A"))),
    ]
    print_table(cit_rows, ["Metric", "Agentic", "Baseline"])

    # --- Confidence and retries ---
    if LAYER1_FILE.exists():
        with open(LAYER1_FILE) as f:
            l1_results = json.load(f)

        agentic_results = [r["agentic"] for r in l1_results if not r["agentic"].get("error")]
        baseline_results = [r["baseline"] for r in l1_results if not r["baseline"].get("error")]

        avg_confidence = round(sum(r["confidence"] for r in agentic_results) / len(agentic_results), 3) if agentic_results else 0
        avg_retries = round(sum(r["retries"] for r in agentic_results) / len(agentic_results), 2) if agentic_results else 0
        retry_rate = round(sum(1 for r in agentic_results if r["retries"] > 0) / len(agentic_results), 3) if agentic_results else 0
        avg_agentic_time = round(sum(r["agentic_time_s"] for r in l1_results) / len(l1_results), 2)
        avg_baseline_time = round(sum(r["baseline_time_s"] for r in l1_results) / len(l1_results), 2)

        print("\nAgentic Pipeline Stats (Layer 1):")
        stats_rows = [
            ("avg_confidence",  str(avg_confidence), "—"),
            ("retry_rate",      str(retry_rate),      "—"),
            ("avg_retries",     str(avg_retries),     "—"),
            ("avg_time_s",      str(avg_agentic_time), str(avg_baseline_time)),
        ]
        print_table(stats_rows, ["Stat", "Agentic", "Baseline"])

        # --- Out-of-scope handling ---
        oob = next((r for r in l1_results if r["id"] == "q7"), None)
        if oob:
            print("\nOut-of-scope question (T9999):")
            answer_preview = oob["agentic"]["answer"][:120].replace("\n", " ")
            print(f"  Agentic answer:  {answer_preview}...")
            print(f"  Correctly returned insufficient evidence: {'INSUFFICIENT' in oob['agentic']['answer'].upper()}")

    # --- Layer 2 summary ---
    if LAYER2_FILE.exists():
        with open(LAYER2_FILE) as f:
            l2_results = json.load(f)

        print("\nLayer 2 Research Pipeline Summary:")
        l2_rows = []
        for r in l2_results:
            res = r["result"]
            n_sub = len(res.get("subquestions", []))
            conf = res.get("confidence", 0)
            t = r.get("time_s", 0)
            has_sections = all(
                sec in res.get("report", "")
                for sec in ["## Executive Summary", "## Findings", "## Recommendations"]
            )
            l2_rows.append((r["id"], str(n_sub), f"{conf:.0%}", f"{t}s", str(has_sections)))

        print_table(l2_rows, ["ID", "Sub-Qs", "Confidence", "Time", "All Sections"])

    print("\n" + "=" * 60)
    print("  END OF EVALUATION REPORT")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()
