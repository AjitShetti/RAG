"""Agentic RAG State Definitions for LangGraph workflow."""

import operator
from typing import Annotated, Any, TypedDict


class ReasoningStep(TypedDict):
    """Intermediate reasoning or decision step in agentic workflow."""

    node: str
    decision: str
    detail: str


class AgentState(TypedDict):
    """State object passed between nodes in the Agentic RAG graph."""

    original_query: str
    current_query: str
    retrieved_chunks: list[Any]
    grading_result: str
    rewrite_count: int
    final_answer: str
    sources: list[Any]
    reasoning_steps: Annotated[list[dict], operator.add]
    rejected: bool
    took_ms: float
    prompt_tokens: int
    completion_tokens: int
