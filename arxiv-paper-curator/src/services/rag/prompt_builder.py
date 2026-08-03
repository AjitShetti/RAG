"""Prompt construction utilities for RAG pipeline.

Pure, deterministic functions with token budget estimation and citation formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...schemas.search import HybridSearchHit

logger = logging.getLogger(__name__)

# Default prompt path
DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).parent.parent / "llm" / "prompts" / "rag_system.txt"
)


def load_system_prompt(path: Path | str | None = None) -> str:
    """Load system prompt text from file."""
    prompt_file = Path(path) if path else DEFAULT_SYSTEM_PROMPT_PATH
    if not prompt_file.exists():
        logger.warning("System prompt file '%s' not found — using fallback", prompt_file)
        return (
            "You are an academic AI assistant. Answer the user question based strictly on the "
            "provided paper context chunks. Cite sources as [paper_id § section_name]. "
            "If the context is insufficient, state clearly that you don't know."
        )
    return prompt_file.read_text(encoding="utf-8").strip()


def estimate_tokens(text: str) -> int:
    """Estimate token count using 4 characters ≈ 1 token heuristic."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def format_chunk_context(chunk: HybridSearchHit, index: int) -> str:
    """Format a single retrieved chunk into a structured, labeled text block."""
    authors_str = ", ".join(chunk.authors) if chunk.authors else "Unknown"
    return (
        f"--- [Context Item {index}] ---\n"
        f"Paper ID: {chunk.paper_id}\n"
        f"Title: {chunk.title}\n"
        f"Section: {chunk.section_name}\n"
        f"Authors: {authors_str}\n"
        f"Text:\n{chunk.text}\n"
    )


def build_context_block(
    chunks: list[HybridSearchHit],
    max_tokens: int = 3000,
) -> tuple[str, list[HybridSearchHit]]:
    """Select chunks that fit within max_tokens budget.

    Returns a tuple of:
    - Assembled context string
    - List of HybridSearchHit chunks actually included
    """
    if not chunks:
        return "", []

    included_chunks: list[HybridSearchHit] = []
    formatted_blocks: list[str] = []
    current_token_count = 0

    for i, chunk in enumerate(chunks, start=1):
        block_text = format_chunk_context(chunk, i)
        block_tokens = estimate_tokens(block_text)

        if current_token_count + block_tokens > max_tokens:
            logger.info(
                "Reached context token budget limit (%d / %d tokens). Stopped after %d of %d chunks.",
                current_token_count,
                max_tokens,
                len(included_chunks),
                len(chunks),
            )
            break

        formatted_blocks.append(block_text)
        included_chunks.append(chunk)
        current_token_count += block_tokens

    context_str = "\n".join(formatted_blocks)
    return context_str, included_chunks


def build_messages(
    system_prompt: str,
    context_block: str,
    user_question: str,
) -> list[dict[str, str]]:
    """Assemble final OpenAI-format messages payload for the LLM."""
    if not context_block.strip():
        user_content = f"Question: {user_question}\n\n[No relevant context chunks were found.]"
    else:
        user_content = (
            f"RETRIEVED CONTEXT CHUNKS:\n\n"
            f"{context_block}\n"
            f"USER QUESTION: {user_question}"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
