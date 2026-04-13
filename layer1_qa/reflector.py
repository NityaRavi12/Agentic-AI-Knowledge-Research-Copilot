from __future__ import annotations
import json
import logging
import re
from layer1_qa.schemas import AnswerDraft, EvidenceChunk, GroundingResult

logger = logging.getLogger(__name__)


def _parse_grounding_json(raw: str) -> GroundingResult:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
        return GroundingResult(grounded=bool(data.get("grounded", False)), issues=list(data.get("issues") or []), raw_response=raw)
    except Exception:
        pass
    grounded_match = re.search(r'"grounded"\s*:\s*(true|false)', raw, re.IGNORECASE)
    grounded = grounded_match.group(1).lower() == "true" if grounded_match else False
    return GroundingResult(grounded=bool(grounded), issues=[], raw_response=raw)


def reflect(draft: AnswerDraft) -> GroundingResult:
    if draft.is_insufficient:
        return GroundingResult(grounded=False, issues=["Answer was marked INSUFFICIENT EVIDENCE."])
    if not draft.evidence:
        return GroundingResult(grounded=False, issues=["No evidence provided."])

    from llm.providers import chat
    evidence_block = "\n\n".join(f"[{c.technique_id}] {c.title}\n{c.text[:600]}" for c in draft.evidence)
    prompt = (
        f"Check that every factual claim in the Answer is supported by the Evidence. "
        f'Respond ONLY with JSON: {{"grounded": true|false, "issues": ["..."]}}\n\n'
        f"Question: {draft.question}\nEvidence:\n{evidence_block}\nAnswer:\n{draft.answer_text}"
    )
    try:
        raw = chat(prompt, temperature=0.0)
    except Exception as exc:
        return GroundingResult(grounded=True, issues=[], raw_response=str(exc))

    return _parse_grounding_json(raw)
