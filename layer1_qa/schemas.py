from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Question(BaseModel):
    text: str
    session_id: Optional[str] = None


class RetrievalPlan(BaseModel):
    query: str
    original_question: str


class EvidenceStatus(str, Enum):
    RELEVANT = "RELEVANT"
    IRRELEVANT = "IRRELEVANT"
    UNSCORED = "UNSCORED"


class EvidenceChunk(BaseModel):
    chunk_id: str
    technique_id: str
    title: str
    section: str = "description"
    url: str
    text: str
    score: float = Field(..., ge=0.0)
    status: EvidenceStatus = EvidenceStatus.UNSCORED


class EvidenceSet(BaseModel):
    question: str
    retrieved: list[EvidenceChunk] = Field(default_factory=list)

    @property
    def approved(self) -> list[EvidenceChunk]:
        return [c for c in self.retrieved if c.status == EvidenceStatus.RELEVANT]

    @property
    def rejected(self) -> list[EvidenceChunk]:
        return [c for c in self.retrieved if c.status == EvidenceStatus.IRRELEVANT]


class AnswerDraft(BaseModel):
    question: str
    answer_text: str
    evidence: list[EvidenceChunk]
    is_insufficient: bool = False


class GroundingResult(BaseModel):
    grounded: bool
    issues: list[str] = Field(default_factory=list)
    raw_response: str = ""


class AnswerStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNGROUNDED_AFTER_RETRY = "UNGROUNDED_AFTER_RETRY"
    ERROR = "ERROR"


class Citation(BaseModel):
    technique_id: str
    title: str
    url: str


class AnswerResult(BaseModel):
    question: str
    answer: str
    status: AnswerStatus
    citations: list[Citation] = Field(default_factory=list)
    grounding_issues: list[str] = Field(default_factory=list)
    retry_count: int = 0
    error_message: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == AnswerStatus.SUCCESS
