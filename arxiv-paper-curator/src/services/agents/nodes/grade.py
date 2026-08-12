"""Grading node for Agentic RAG workflow."""

import logging
from typing import Any

from ...llm.client import LLMClient
from ..state import AgentState

logger = logging.getLogger(__name__)

GRADE_SYSTEM_PROMPT = (
    "You are an expert scientific relevance grader evaluating whether retrieved paper chunks "
    "contain relevant context to answer the user's query.\n"
    "Respond 'YES' if the chunks contain direct information, explanations, methods, definitions, "
    "or key facts addressing the user's question.\n"
    "Respond 'NO' only if the retrieved chunks are off-topic, completely unrelated, or lack "
    "useful information to address the query.\n"
    "Your response MUST contain 'YES' or 'NO'."
)


def grade_node(
    state: AgentState,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Evaluate relevance of retrieved_chunks against current_query via LLM."""
    llm = llm_client or LLMClient(model="llama-3.1-8b-instant")
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
            f"--- Chunk {i + 1} [{c.section_name}] (Score: {getattr(c, 'score', 0.0):.4f}) ---\n{getattr(c, 'text', '')[:1200]}"
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
