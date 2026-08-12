"""Guardrail node for Agentic RAG workflow."""

import logging
from typing import Any

from ...llm.client import LLMClient
from ..state import AgentState

logger = logging.getLogger(__name__)

GUARDRAIL_SYSTEM_PROMPT = (
    "You are an input guardrail classifier for an arXiv research assistant specializing in "
    "Computer Science, Artificial Intelligence, Machine Learning, Data Science, and related technical fields.\n"
    "Determine whether the given user query is in-domain (CS/AI/ML/Data Science/software engineering research) "
    "or out-of-domain (recipes, pop culture, non-technical chit-chat, unrelated topics).\n"
    "Respond ONLY with 'YES' if in-domain, or 'NO' if out-of-domain."
)

OUT_OF_DOMAIN_ANSWER = (
    "This question is out of domain. Please ask questions related to Computer Science, "
    "Artificial Intelligence, Machine Learning, or Data Science research."
)


def guardrail_node(
    state: AgentState,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Validate whether the input query is in-domain for arXiv CS/AI research."""
    llm = llm_client or LLMClient(model="llama-3.1-8b-instant")
    query = state.get("current_query") or state.get("original_query") or ""

    messages = [
        {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}"},
    ]

    try:
        response = llm.generate(messages=messages, temperature=0.0)
        verdict = response.content.strip().upper()
    except Exception as e:
        logger.warning("Guardrail LLM call failed: %s; allowing query through", e)
        verdict = "YES"

    if "NO" in verdict and "YES" not in verdict:
        logger.info("Guardrail rejected out-of-domain query: %r", query)
        return {
            "rejected": True,
            "final_answer": OUT_OF_DOMAIN_ANSWER,
            "reasoning_steps": [
                {
                    "node": "guardrail",
                    "decision": "out_of_domain",
                    "detail": "Rejected query as out-of-domain",
                }
            ],
        }

    logger.info("Guardrail accepted query: %r", query)
    return {
        "rejected": False,
        "reasoning_steps": [
            {
                "node": "guardrail",
                "decision": "in_domain",
                "detail": "Query validated as CS/AI research topic",
            }
        ],
    }
