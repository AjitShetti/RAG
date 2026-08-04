"""Grading node for Agentic RAG workflow."""

import logging
from typing import Any

from ...llm.client import LLMClient
from ..state import AgentState

logger = logging.getLogger(__name__)

GRADE_SYSTEM_PROMPT = (
    "You are a relevance grader evaluating whether retrieved scientific paper chunks contain "
    "enough relevant context and information to answer the user query.\n"
    "Respond ONLY with 'YES' if the retrieved chunks contain relevant and sufficient information to answer the query, "
    "or 'NO' if they lack sufficient relevant context."
)


def grade_node(
    state: AgentState,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Evaluate relevance of retrieved_chunks against current_query via LLM."""
    llm = llm_client or LLMClient()
    query = state.get("current_query") or state.get("original_query") or ""
    chunks = state.get("retrieved_chunks") or []

    if not chunks:
        logger.info("No chunks retrieved; grading result is weak")
        return {
            "grading_result": "weak",
            "reasoning_steps": [
                {
                    "node": "grade",
                    "decision": "weak",
                    "detail": "Retrieved chunks lack sufficient relevant context",
                }
            ],
        }

    context_preview = "\n\n".join(
        [
            f"--- Chunk {i + 1} (Score: {getattr(c, 'score', 0.0):.4f}) ---\n{getattr(c, 'text', '')[:400]}"
            for i, c in enumerate(chunks[:5])
        ]
    )

    messages = [
        {"role": "system", "content": GRADE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"User Query: {query}\n\nRetrieved Context:\n{context_preview}",
        },
    ]

    try:
        response = llm.generate(messages=messages, temperature=0.0)
        verdict = response.content.strip().upper()
    except Exception as e:
        logger.warning("Relevance grading LLM call failed: %s; marking as weak", e)
        verdict = "NO"

    if "YES" in verdict and "NO" not in verdict:
        logger.info("Grading node assessed context as strong for query: %r", query)
        return {
            "grading_result": "strong",
            "reasoning_steps": [
                {
                    "node": "grade",
                    "decision": "strong",
                    "detail": "Chunks contain sufficient information",
                }
            ],
        }

    logger.info("Grading node assessed context as weak for query: %r", query)
    return {
        "grading_result": "weak",
        "reasoning_steps": [
            {
                "node": "grade",
                "decision": "weak",
                "detail": "Retrieved chunks lack sufficient relevant context",
            }
        ],
    }
