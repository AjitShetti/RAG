"""Query rewrite node for Agentic RAG workflow."""

import logging
from typing import Any

from ...llm.client import LLMClient
from ..state import AgentState

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = (
    "You are an expert search query optimizer for arXiv computer science and AI research papers.\n"
    "The initial query failed to retrieve relevant paper chunks.\n"
    "Optimize the query by extracting key technical terms, acronyms, and core scientific concepts.\n"
    "Keep the query concise (5 to 10 key search terms). Do NOT write long explanatory sentences or titles.\n"
    "Respond ONLY with the concise optimized search terms, with no quotes or explanation."
)


def rewrite_node(
    state: AgentState,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Rewrite original_query into a concise, keyword-tightened query and increment rewrite_count."""
    llm = llm_client or LLMClient()
    original = state.get("original_query") or state.get("current_query") or ""
    current = state.get("current_query") or original
    count = state.get("rewrite_count", 0)

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Original query: {original}\nPrevious attempt: {current}\nGenerate concise optimized search keywords:",
        },
    ]

    try:
        response = llm.generate(messages=messages, temperature=0.2)
        new_query = response.content.strip().strip('"').strip("'")
        if not new_query:
            new_query = current
    except Exception as e:
        logger.warning("Query rewrite LLM call failed: %s; keeping previous query", e)
        new_query = current

    logger.info("Rewrote query %r -> %r (attempt %d)", current, new_query, count + 1)

    return {
        "current_query": new_query,
        "rewrite_count": count + 1,
        "reasoning_steps": [
            {
                "node": "rewrite",
                "decision": "rewritten",
                "detail": f"Rewrote query '{current}' to '{new_query}' (attempt {count + 1})",
            }
        ],
    }
