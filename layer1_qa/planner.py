from __future__ import annotations
import logging
from layer1_qa.schemas import Question, RetrievalPlan

logger = logging.getLogger(__name__)

_FALLBACK_PROMPT = (
    "Convert the following user question into a short, precise search query "
    "for a MITRE ATT&CK vector database. Output ONLY the query string.\n\n"
    "User question: {question}"
)


def plan(question: Question) -> RetrievalPlan:
    from llm.providers import chat
    prompt = _FALLBACK_PROMPT.format(question=question.text)
    try:
        query = chat(prompt, temperature=0.0).strip()
        if len(query) > 300:
            query = query[:300]
        if not query:
            raise ValueError("Empty query")
    except Exception as exc:
        logger.warning("Planner failed (%s); using raw question.", exc)
        query = question.text
    return RetrievalPlan(query=query, original_question=question.text)
