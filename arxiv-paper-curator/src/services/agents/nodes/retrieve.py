"""Retrieval node for Agentic RAG workflow."""

import logging
from typing import Any

from ....schemas.rag import AskRequest
from ...rag.pipeline import RAGPipeline
from ..state import AgentState

logger = logging.getLogger(__name__)


def retrieve_node(
    state: AgentState,
    rag_pipeline: RAGPipeline | None = None,
) -> dict[str, Any]:
    """Retrieve candidate paper chunks for the current query using RAGPipeline."""
    pipeline = rag_pipeline or RAGPipeline()
    query = state.get("current_query") or state.get("original_query") or ""

    request = AskRequest(query=query)
    chunks = pipeline.retrieve(request)

    logger.info("Retrieved %d candidate chunks for query: %r", len(chunks), query)

    return {
        "retrieved_chunks": chunks,
        "reasoning_steps": [
            {
                "node": "retrieve",
                "decision": "retrieved",
                "detail": f"Retrieved {len(chunks)} candidate chunks for query: '{query}'",
            }
        ],
    }
