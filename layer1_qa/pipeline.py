from __future__ import annotations
import logging
from layer1_qa.schemas import AnswerResult, AnswerStatus, Citation, EvidenceChunk, EvidenceStatus, Question

logger = logging.getLogger(__name__)

def _build_citations(chunks):
    seen = {}
    for c in chunks:
        if c.technique_id not in seen:
            seen[c.technique_id] = Citation(technique_id=c.technique_id, title=c.title, url=c.url)
    return sorted(seen.values(), key=lambda x: x.technique_id)

def _run_once(question_text, query):
    from retrieval.qdrant_retriever import search as qdrant_search
    from layer1_qa.schemas import EvidenceChunk as EC
    raw_hits = qdrant_search(query)
    chunks = [EC(chunk_id=h.chunk_id, technique_id=h.technique_id, title=h.title, section=h.section, url=h.url, text=h.text, score=h.score) for h in raw_hits]
    approved = [c for c in chunks if c.score >= 0.40]
    for c in approved:
        c.status = EvidenceStatus.RELEVANT
    return approved, query

def run(question):
    from layer1_qa.planner import plan
    from layer1_qa.answerer import answer
    from layer1_qa.reflector import reflect
    try:
        rp = plan(question)
    except Exception as exc:
        return AnswerResult(question=question.text, answer="", status=AnswerStatus.ERROR, error_message=str(exc))
    try:
        approved, query_used = _run_once(question.text, rp.query)
    except Exception as exc:
        return AnswerResult(question=question.text, answer="", status=AnswerStatus.ERROR, error_message=str(exc))
    if len(approved) < 2:
        return AnswerResult(question=question.text, answer="I could not find sufficient evidence in the MITRE ATT&CK knowledge base to answer this question reliably.", status=AnswerStatus.INSUFFICIENT_EVIDENCE)
    draft = answer(question.text, approved)
    if draft.is_insufficient:
        return AnswerResult(question=question.text, answer=draft.answer_text, status=AnswerStatus.INSUFFICIENT_EVIDENCE)
    grounding = reflect(draft)
    retry_count = 0
    if not grounding.grounded:
        retry_count = 1
        try:
            retry_approved, _ = _run_once(question.text, question.text + " MITRE ATT&CK")
        except Exception:
            retry_approved = approved
        if len(retry_approved) >= 2:
            retry_draft = answer(question.text, retry_approved)
            if not retry_draft.is_insufficient:
                grounding = reflect(retry_draft)
                if grounding.grounded:
                    draft = retry_draft
                    approved = retry_approved
    if grounding.grounded:
        return AnswerResult(question=question.text, answer=draft.answer_text, status=AnswerStatus.SUCCESS, citations=_build_citations(draft.evidence), retry_count=retry_count)
    return AnswerResult(question=question.text, answer=draft.answer_text, status=AnswerStatus.UNGROUNDED_AFTER_RETRY, citations=_build_citations(draft.evidence), grounding_issues=grounding.issues, retry_count=retry_count)
