from __future__ import annotations
import logging
from layer1_qa.schemas import EvidenceChunk, EvidenceSet, EvidenceStatus

logger = logging.getLogger(__name__)

def evaluate(question: str, chunks: list[EvidenceChunk], *, min_score: float | None = None) -> EvidenceSet:
    threshold = min_score if min_score is not None else 0.40
    evidence_set = EvidenceSet(question=question, retrieved=list(chunks))
    for chunk in evidence_set.retrieved:
        if chunk.score < threshold:
            chunk.status = EvidenceStatus.IRRELEVANT
        else:
            chunk.status = EvidenceStatus.RELEVANT
    return evidence_set
