from __future__ import annotations
import logging
from layer1_qa.schemas import AnswerDraft, EvidenceChunk

logger = logging.getLogger(__name__)
_INSUFFICIENT_MARKER = "INSUFFICIENT EVIDENCE"


def _build_evidence_block(chunks: list[EvidenceChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c.technique_id} — {c.title} (score={c.score:.3f})\n{c.text[:900]}")
    return "\n\n".join(parts)


def answer(question: str, evidence: list[EvidenceChunk]) -> AnswerDraft:
    from llm.providers import chat
    evidence_block = _build_evidence_block(evidence)
    prompt = (
        f"Answer the question using ONLY the evidence below. "
        f"Cite every claim with [technique_id]. If there is not enough evidence, "
        f"write 'INSUFFICIENT EVIDENCE: <reason>'.\n\n"
        f"Question: {question}\n\nEvidence:\n{evidence_block}\n\nAnswer:"
    )
    try:
        answer_text = chat(prompt, temperature=0.1).strip()
    except Exception as exc:
        answer_text = f"{_INSUFFICIENT_MARKER}: LLM call failed ({exc})"

    return AnswerDraft(
        question=question,
        answer_text=answer_text,
        evidence=evidence,
        is_insufficient=answer_text.upper().startswith(_INSUFFICIENT_MARKER),
    )
