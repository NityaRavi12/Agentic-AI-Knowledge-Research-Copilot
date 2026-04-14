from __future__ import annotations

import logging
from typing import Any

from llm.providers import chat, chat_json
from layer1_qa.schemas import EvidenceChunk as SchemaEvidenceChunk
from layer1_qa.schemas import EvidenceSet, EvidenceStatus

logger = logging.getLogger(__name__)


def _get_attr(chunk: Any, name: str, default: Any = None) -> Any:
    return getattr(chunk, name, default)


def _normalize_chunk_to_schema(chunk: Any) -> SchemaEvidenceChunk:
    """
    Convert a retriever chunk or dict-like chunk into the pydantic schema version
    used by layer1_qa.schemas.
    """
    if isinstance(chunk, SchemaEvidenceChunk):
        return chunk

    data = {
        "chunk_id": _get_attr(chunk, "chunk_id", ""),
        "technique_id": _get_attr(chunk, "technique_id", "N/A"),
        "title": _get_attr(chunk, "title", "Untitled"),
        "text": _get_attr(chunk, "text", ""),
        "score": float(_get_attr(chunk, "score", 0.0) or 0.0),
        "source": _get_attr(chunk, "source", ""),
        "section": _get_attr(chunk, "section", ""),
        "status": _get_attr(chunk, "status", EvidenceStatus.RETRIEVED),
    }

    return SchemaEvidenceChunk(**data)


def _format_chunks_for_prompt(chunks: list[Any]) -> str:
    lines: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        title = _get_attr(chunk, "title", "Untitled") or "Untitled"
        technique_id = _get_attr(chunk, "technique_id", "N/A") or "N/A"
        source = _get_attr(chunk, "source", "N/A") or "N/A"
        score = float(_get_attr(chunk, "score", 0.0) or 0.0)
        text = _get_attr(chunk, "text", "") or ""

        lines.append(
            f"[Chunk {idx}]\n"
            f"Title: {title}\n"
            f"Technique ID: {technique_id}\n"
            f"Source: {source}\n"
            f"Retriever Score: {score:.4f}\n"
            f"Text: {text}\n"
        )

    return "\n".join(lines)


def evaluate_retrieval(
    question: str,
    chunks: list[Any],
    *,
    min_score: float | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluate whether retrieved chunks are relevant and sufficient for the question.
    """
    threshold = min_score if min_score is not None else 0.40

    if not chunks:
        return {
            "approved_chunks": [],
            "rejected_chunks": [],
            "relevance_score": 0.0,
            "sufficiency_score": 0.0,
            "missing_aspects": ["No evidence was retrieved."],
            "needs_retry": True,
            "rewritten_query": question,
        }

    chunk_block = _format_chunks_for_prompt(chunks)

    prompt = f"""
You are evaluating retrieved cybersecurity evidence for a question-answering system.

Question:
{question}

Retrieved chunks:
{chunk_block}

Tasks:
1. Decide which chunks are relevant to the question.
2. Decide whether the retrieved evidence is sufficient to answer the question fully without guessing.
3. Identify what information is missing, if any.
4. Suggest a rewritten search query only if retrieval should be retried.

Return a JSON object with exactly these keys:
{{
  "relevant_chunk_ids": [1, 2],
  "irrelevant_chunk_ids": [3],
  "relevance_score": 0.0,
  "sufficiency_score": 0.0,
  "missing_aspects": ["..."],
  "needs_retry": false,
  "rewritten_query": "..."
}}

Rules:
- Chunk ids are 1-based and must refer to the chunk numbers shown above.
- Use scores between 0.0 and 1.0.
- Mark a chunk relevant only if it directly helps answer the question.
- Set needs_retry to true if evidence is not sufficient.
- If no better query is needed, set rewritten_query to the original question.
""".strip()

    try:
        result = chat_json(prompt, temperature=temperature)
    except Exception as exc:
        logger.warning("LLM retrieval evaluation failed, falling back to score threshold: %s", exc)

        approved_chunks: list[Any] = []
        rejected_chunks: list[Any] = []

        for chunk in chunks:
            if float(_get_attr(chunk, "score", 0.0) or 0.0) < threshold:
                if hasattr(chunk, "status"):
                    chunk.status = EvidenceStatus.IRRELEVANT
                rejected_chunks.append(chunk)
            else:
                if hasattr(chunk, "status"):
                    chunk.status = EvidenceStatus.RELEVANT
                approved_chunks.append(chunk)

        relevance_score = len(approved_chunks) / len(chunks) if chunks else 0.0
        sufficiency_score = 1.0 if len(approved_chunks) >= 2 else 0.4

        return {
            "approved_chunks": approved_chunks,
            "rejected_chunks": rejected_chunks,
            "relevance_score": relevance_score,
            "sufficiency_score": sufficiency_score,
            "missing_aspects": [] if len(approved_chunks) >= 2 else ["Evidence coverage is thin."],
            "needs_retry": len(approved_chunks) < 2,
            "rewritten_query": question,
        }

    relevant_ids = {
        idx for idx in result.get("relevant_chunk_ids", []) if isinstance(idx, int)
    }
    irrelevant_ids = {
        idx for idx in result.get("irrelevant_chunk_ids", []) if isinstance(idx, int)
    }

    approved_chunks: list[Any] = []
    rejected_chunks: list[Any] = []

    for idx, chunk in enumerate(chunks, start=1):
        score = float(_get_attr(chunk, "score", 0.0) or 0.0)

        if idx in relevant_ids:
            if hasattr(chunk, "status"):
                chunk.status = EvidenceStatus.RELEVANT
            approved_chunks.append(chunk)
        elif idx in irrelevant_ids:
            if hasattr(chunk, "status"):
                chunk.status = EvidenceStatus.IRRELEVANT
            rejected_chunks.append(chunk)
        else:
            if score >= threshold:
                if hasattr(chunk, "status"):
                    chunk.status = EvidenceStatus.RELEVANT
                approved_chunks.append(chunk)
            else:
                if hasattr(chunk, "status"):
                    chunk.status = EvidenceStatus.IRRELEVANT
                rejected_chunks.append(chunk)

    return {
        "approved_chunks": approved_chunks,
        "rejected_chunks": rejected_chunks,
        "relevance_score": float(result.get("relevance_score", 0.0)),
        "sufficiency_score": float(result.get("sufficiency_score", 0.0)),
        "missing_aspects": result.get("missing_aspects", []),
        "needs_retry": bool(result.get("needs_retry", False)),
        "rewritten_query": result.get("rewritten_query", question) or question,
    }


def evaluate_answer(
    question: str,
    answer: str,
    evidence: list[Any],
    *,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if not answer.strip():
        return {
            "grounded": False,
            "complete": False,
            "groundedness_score": 0.0,
            "completeness_score": 0.0,
            "unsupported_claims": ["No answer was generated."],
            "missing_aspects": ["The question was not answered."],
            "revision_needed": True,
            "revision_instructions": "Generate an answer using only the approved evidence.",
        }

    evidence_block = _format_chunks_for_prompt(evidence)

    prompt = f"""
You are verifying whether an answer is properly grounded in cybersecurity evidence.

Question:
{question}

Answer:
{answer}

Evidence:
{evidence_block}

Return a JSON object with exactly these keys:
{{
  "grounded": true,
  "complete": true,
  "groundedness_score": 0.0,
  "completeness_score": 0.0,
  "unsupported_claims": ["..."],
  "missing_aspects": ["..."],
  "revision_needed": false,
  "revision_instructions": "..."
}}

Rules:
- grounded = true only if all major claims are supported by the evidence.
- complete = true only if the answer addresses all major parts of the question.
- Use scores between 0.0 and 1.0.
- If the answer needs improvement, set revision_needed to true.
- revision_instructions should be concrete and short.
""".strip()

    try:
        result = chat_json(prompt, temperature=temperature)
    except Exception as exc:
        logger.warning("LLM answer evaluation failed, using conservative fallback: %s", exc)

        has_evidence = len(evidence) > 0
        groundedness_score = 0.6 if has_evidence else 0.0
        completeness_score = 0.5 if len(answer.split()) >= 40 else 0.3
        revision_needed = (not has_evidence) or len(answer.split()) < 40

        return {
            "grounded": has_evidence,
            "complete": not revision_needed,
            "groundedness_score": groundedness_score,
            "completeness_score": completeness_score,
            "unsupported_claims": [] if has_evidence else ["No evidence available to support the answer."],
            "missing_aspects": [] if not revision_needed else ["The answer may be incomplete or weakly supported."],
            "revision_needed": revision_needed,
            "revision_instructions": "Revise the answer to use only the approved evidence and cover the full question.",
        }

    return {
        "grounded": bool(result.get("grounded", False)),
        "complete": bool(result.get("complete", False)),
        "groundedness_score": float(result.get("groundedness_score", 0.0)),
        "completeness_score": float(result.get("completeness_score", 0.0)),
        "unsupported_claims": result.get("unsupported_claims", []),
        "missing_aspects": result.get("missing_aspects", []),
        "revision_needed": bool(result.get("revision_needed", False)),
        "revision_instructions": result.get(
            "revision_instructions",
            "Revise the answer using only the evidence.",
        ),
    }


def revise_answer(
    question: str,
    answer: str,
    evidence: list[Any],
    feedback: dict[str, Any],
    *,
    temperature: float = 0.1,
) -> str:
    evidence_block = _format_chunks_for_prompt(evidence)

    prompt = f"""
Revise the answer using only the provided evidence.

Question:
{question}

Current answer:
{answer}

Evidence:
{evidence_block}

Evaluator feedback:
{feedback}

Rules:
- Keep only claims supported by the evidence.
- Remove unsupported or exaggerated claims.
- Cover missing aspects if the evidence supports them.
- Do not invent facts.
- Keep the answer clear and concise.
- Include citations only if your generation format already supports them.
""".strip()

    return chat(prompt, temperature=temperature).strip()


def evaluate(
    question: str,
    chunks: list[Any],
    *,
    min_score: float | None = None,
) -> EvidenceSet:
    """
    Backward-compatible wrapper for older code paths that still expect EvidenceSet.
    """
    threshold = min_score if min_score is not None else 0.40
    normalized_chunks = [_normalize_chunk_to_schema(chunk) for chunk in chunks]
    evidence_set = EvidenceSet(question=question, retrieved=normalized_chunks)

    for chunk in evidence_set.retrieved:
        chunk.status = (
            EvidenceStatus.RELEVANT
            if float(chunk.score or 0.0) >= threshold
            else EvidenceStatus.IRRELEVANT
        )

    return evidence_set