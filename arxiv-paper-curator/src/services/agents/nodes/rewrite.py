"""Query rewrite node for Agentic RAG workflow."""

import logging
from typing import Any

from ...llm.client import LLMClient
from ..state import AgentState

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = (
    "You are an expert search query optimizer for arXiv computer science and AI research papers.\n"
    "The user query previously failed to retrieve sufficient relevant paper chunks. "
    "Rewrite the query to be clearer, broader, or use standard scientific and academic terminology "
    "better suited for hybrid search retrieval.\n"
    "Respond ONLY with the rewritten search query text, with no explanation or surrounding quotes."
)


def rewrite_node(
    state: AgentState,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Rewrite current_query into a broader/more search-friendly query and increment rewrite_count."""
    llm = llm_client or LLMClient()
    old = state.get("current_query") or state.get("original_query") or ""
    count = state.get("rewrite_count", 0)

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Original query to optimize: {old}"},
    ]

    try:
        response = llm.generate(messages=messages, temperature=0.2)
        new_query = response.content.strip().strip('"').strip("'")
        if not new_query:
            new_query = old
    except Exception as e:
        logger.warning("Query rewrite LLM call failed: %s; keeping previous query", e)
        new_query = old

    logger.info("Rewrote query %r -> %r (attempt %d)", old, new_query, count + 1)

    return {
        "current_query": new_query,
        "rewrite_count": count + 1,
        "reasoning_steps": [
            {
                "node": "rewrite",
                "decision": "rewritten",
                "detail": f"Rewrote query '{old}' to '{new_query}' (attempt {count + 1})",
            }
        ],
    }
