"""
evaluation/metrics.py

Computes evaluation metrics for agentic pipeline vs baseline RAG.
Uses LOCAL embeddings only (all-MiniLM-L6-v2) — zero API calls, zero token limits.

Metrics computed:
  - faithfulness_proxy    : cosine similarity between answer and retrieved chunks
  - answer_relevancy      : cosine similarity between answer and question
  - context_precision     : cosine similarity between chunks and question
  - context_recall        : cosine similarity between answer and ground truth
  - citation_precision    : fraction of returned citations that are expected
  - citation_recall       : fraction of expected citations that were returned
  - avg_confidence        : mean internal confidence score (agentic only)

Reads:  evaluation/results_layer1.json
Writes: evaluation/metrics_ragas.json
"""

from __future__ import annotations
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "evaluation"
INPUT_FILE  = EVAL_DIR / "results_layer1.json"
OUTPUT_FILE = EVAL_DIR / "metrics_ragas.json"


# ── Embedder (runs fully locally, no API) ─────────────────────────────────

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading all-MiniLM-L6-v2 (local, no API)...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit vectors (already normalized)."""
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


# ── Per-question metric helpers ────────────────────────────────────────────

def faithfulness_proxy(answer: str, chunks: list[dict]) -> float:
    """
    Proxy for faithfulness: max cosine similarity between the answer
    and any retrieved chunk. Higher = answer is more grounded in evidence.
    """
    if not answer.strip() or not chunks:
        return 0.0
    chunk_texts = [c["text"][:500] for c in chunks if c.get("text")]
    if not chunk_texts:
        return 0.0
    all_texts = [answer] + chunk_texts
    vecs = embed(all_texts)
    ans_vec = vecs[0]
    chunk_vecs = vecs[1:]
    return round(float(np.max([cosine(ans_vec, cv) for cv in chunk_vecs])), 4)


def answer_relevancy(question: str, answer: str) -> float:
    """Cosine similarity between question and answer."""
    if not answer.strip():
        return 0.0
    vecs = embed([question, answer])
    return round(cosine(vecs[0], vecs[1]), 4)


def context_precision(question: str, chunks: list[dict]) -> float:
    """
    Mean cosine similarity between the question and each retrieved chunk.
    Measures whether retrieved chunks are relevant to the question.
    """
    if not chunks:
        return 0.0
    chunk_texts = [c["text"][:500] for c in chunks if c.get("text")]
    if not chunk_texts:
        return 0.0
    all_texts = [question] + chunk_texts
    vecs = embed(all_texts)
    q_vec = vecs[0]
    chunk_vecs = vecs[1:]
    scores = [cosine(q_vec, cv) for cv in chunk_vecs]
    return round(float(np.mean(scores)), 4)


def context_recall(answer: str, ground_truth: str) -> float:
    """
    Cosine similarity between the answer and the ground truth.
    Proxy for whether the answer covers what it should.
    """
    if not answer.strip() or not ground_truth.strip():
        return 0.0
    vecs = embed([answer, ground_truth])
    return round(cosine(vecs[0], vecs[1]), 4)


# ── Citation metrics (no embeddings needed) ────────────────────────────────

def compute_citation_precision(results: list, mode: str) -> float:
    scores = []
    for r in results:
        expected = set(r.get("expected_citations", []))
        if not expected:
            continue
        data = r[mode]
        returned = set(data.get("citations", []))
        if not returned:
            scores.append(0.0)
            continue
        hits = sum(
            1 for c in returned
            if any(c.startswith(e) or e.startswith(c) for e in expected)
        )
        scores.append(hits / len(returned))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def compute_citation_recall(results: list, mode: str) -> float:
    scores = []
    for r in results:
        expected = set(r.get("expected_citations", []))
        if not expected:
            continue
        data = r[mode]
        returned = set(data.get("citations", []))
        hits = sum(
            1 for e in expected
            if any(e.startswith(c) or c.startswith(e) for c in returned)
        )
        scores.append(hits / len(expected))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    print("Loading Layer 1 results...")
    with open(INPUT_FILE) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results.\n")

    output = {}

    for mode in ["agentic", "baseline"]:
        print(f"--- Computing metrics for: {mode.upper()} (local embeddings only) ---")

        valid = [
            r for r in results
            if r["ground_truth"] != "INSUFFICIENT EVIDENCE"
            and not r[mode].get("error")
            and r[mode].get("answer")
        ]
        print(f"  Valid questions: {len(valid)}")

        if not valid:
            print("  No valid results. Skipping.\n")
            output[mode] = {}
            continue

        faith_scores, rel_scores, prec_scores, recall_scores = [], [], [], []

        for r in valid:
            data = r[mode]
            ans   = data.get("answer", "")
            chunks = data.get("chunks_used", [])
            q     = r["question"]
            gt    = r["ground_truth"]

            faith_scores.append(faithfulness_proxy(ans, chunks))
            rel_scores.append(answer_relevancy(q, ans))
            prec_scores.append(context_precision(q, chunks))
            recall_scores.append(context_recall(ans, gt))

        avg = lambda s: round(float(np.mean(s)), 4) if s else 0.0

        cit_precision = compute_citation_precision(results, mode)
        cit_recall    = compute_citation_recall(results, mode)
        avg_confidence = (
            round(sum(r[mode].get("confidence", 0) for r in valid) / len(valid), 4)
            if mode == "agentic" else None
        )

        output[mode] = {
            "faithfulness":       avg(faith_scores),
            "answer_relevancy":   avg(rel_scores),
            "context_precision":  avg(prec_scores),
            "context_recall":     avg(recall_scores),
            "citation_precision": cit_precision,
            "citation_recall":    cit_recall,
            "avg_confidence":     avg_confidence,
            "n_evaluated":        len(valid),
            "note": "Computed with local all-MiniLM-L6-v2 embeddings (cosine similarity proxies). No API calls used."
        }

        print(f"  faithfulness (proxy):  {output[mode]['faithfulness']}")
        print(f"  answer_relevancy:      {output[mode]['answer_relevancy']}")
        print(f"  context_precision:     {output[mode]['context_precision']}")
        print(f"  context_recall:        {output[mode]['context_recall']}")
        print(f"  citation_precision:    {cit_precision}")
        print(f"  citation_recall:       {cit_recall}")
        if avg_confidence is not None:
            print(f"  avg_confidence:        {avg_confidence}")
        print()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved metrics to {OUTPUT_FILE}")
    print("\n--- COMPARISON SUMMARY ---")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "citation_precision", "citation_recall"]:
        a = output.get("agentic", {}).get(metric, "N/A")
        b = output.get("baseline", {}).get(metric, "N/A")
        if isinstance(a, float) and isinstance(b, float):
            delta = round(a - b, 4)
            sign  = "+" if delta >= 0 else ""
            print(f"  {metric:<22}  agentic={a}  baseline={b}  delta={sign}{delta}")


if __name__ == "__main__":
    run()
