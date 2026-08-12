"""Generation node for Agentic RAG workflow."""

import logging
from typing import Any

from ...rag.pipeline import NO_CONTEXT_ANSWER, RAGPipeline
from ...rag.prompt_builder import build_context_block, build_messages
from ..state import AgentState

logger = logging.getLogger(__name__)


def generate_node(
    state: AgentState,
    rag_pipeline: RAGPipeline | None = None,
) -> dict[str, Any]:
    """Generate final answer using retrieved_chunks and RAG prompt assembly."""
    pipeline = rag_pipeline or RAGPipeline()
    query = state.get("original_query") or state.get("current_query") or ""
    chunks = state.get("retrieved_chunks") or []
    grading_result = state.get("grading_result", "strong")

    if not chunks:
        logger.info("No candidate chunks available for generation")
        return {
            "final_answer": NO_CONTEXT_ANSWER,
            "sources": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_steps": [
                {
                    "node": "generate",
                    "decision": "answered",
                    "detail": "Generated answer using 0 context chunks",
                }
            ],
        }

    context_block, used_chunks = build_context_block(
        chunks=chunks,
        max_tokens=pipeline.max_context_tokens,
    )

    if not used_chunks:
        logger.info("No chunks fit within max context token budget")
        return {
            "final_answer": NO_CONTEXT_ANSWER,
            "sources": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_steps": [
                {
                    "node": "generate",
                    "decision": "answered",
                    "detail": "Generated answer using 0 context chunks",
                }
            ],
        }

    messages = build_messages(
        system_prompt=pipeline.system_prompt,
        context_block=context_block,
        user_question=query,
    )

    try:
        llm_res = pipeline.llm.generate(messages=messages)
        final_answer = llm_res.content
        if grading_result == "weak":
            disclaimer = "⚠️ *Low Confidence Notice: The retrieved context may be incomplete or weakly relevant to the query.*\n\n"
            final_answer = disclaimer + final_answer
        prompt_tokens = llm_res.prompt_tokens
        completion_tokens = llm_res.completion_tokens
    except Exception as e:
        logger.error("LLM generation failed in generate_node: %s", e)
        final_answer = NO_CONTEXT_ANSWER
        prompt_tokens = 0
        completion_tokens = 0

    sources = pipeline._build_source_chunks(used_chunks)

    logger.info("Generated answer using %d context chunks", len(sources))

    return {
        "final_answer": final_answer,
        "sources": sources,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_steps": [
            {
                "node": "generate",
                "decision": "answered",
                "detail": f"Generated answer using {len(sources)} context chunks",
            }
        ],
    }
