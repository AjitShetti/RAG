"""Agentic RAG Orchestrator using LangGraph.

Constructs StateGraph with Guardrail, Retrieve, Grade, Rewrite, and Generate nodes.
Executes adaptive retrieval loops based on relevance grading and rewrite limits.
"""

import logging
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from ...config import settings
from ...schemas.rag import AgenticAskResponse, AskRequest, ReasoningStep
from ..langfuse.service import LangfuseService
from ..llm.client import LLMClient
from ..rag.pipeline import RAGPipeline
from .nodes import (
    generate_node,
    grade_node,
    guardrail_node,
    retrieve_node,
    rewrite_node,
)
from .state import AgentState

logger = logging.getLogger(__name__)


def build_agentic_rag_graph(
    rag_pipeline: RAGPipeline | None = None,
    llm_client: LLMClient | None = None,
) -> Any:
    """Build and compile the Agentic RAG StateGraph workflow."""

    def _guardrail(state: AgentState) -> dict[str, Any]:
        return guardrail_node(state, llm_client=llm_client)

    def _retrieve(state: AgentState) -> dict[str, Any]:
        return retrieve_node(state, rag_pipeline=rag_pipeline)

    def _grade(state: AgentState) -> dict[str, Any]:
        return grade_node(state, llm_client=llm_client)

    def _rewrite(state: AgentState) -> dict[str, Any]:
        return rewrite_node(state, llm_client=llm_client)

    def _generate(state: AgentState) -> dict[str, Any]:
        return generate_node(state, rag_pipeline=rag_pipeline)

    # 1. Initialize StateGraph
    workflow = StateGraph(AgentState)

    # 2. Add nodes
    workflow.add_node("guardrail", _guardrail)
    workflow.add_node("retrieve", _retrieve)
    workflow.add_node("grade", _grade)
    workflow.add_node("rewrite", _rewrite)
    workflow.add_node("generate", _generate)

    # 3. Add edges & conditional routing
    workflow.add_edge(START, "guardrail")

    def route_guardrail(state: AgentState) -> str:
        if state.get("rejected"):
            return END
        return "retrieve"

    workflow.add_conditional_edges(
        "guardrail",
        route_guardrail,
        {END: END, "retrieve": "retrieve"},
    )

    workflow.add_edge("retrieve", "grade")

    def route_grade(state: AgentState) -> str:
        grading = state.get("grading_result")
        rewrite_count = state.get("rewrite_count", 0)
        max_rewrites = getattr(settings, "agentic_max_rewrites", 2)

        if grading == "strong":
            return "generate"
        if grading == "weak" and rewrite_count < max_rewrites:
            return "rewrite"
        return "generate"

    workflow.add_conditional_edges(
        "grade",
        route_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_agentic_rag(
    request: AskRequest,
    rag_pipeline: RAGPipeline | None = None,
    llm_client: LLMClient | None = None,
    langfuse_service: LangfuseService | None = None,
) -> AgenticAskResponse:
    """Execute end-to-end Agentic RAG graph workflow for a user request."""
    start_time = time.perf_counter()

    pipeline = rag_pipeline or RAGPipeline()
    llm = llm_client or LLMClient()

    trace = (
        langfuse_service.start_trace(
            name="agentic_rag",
            metadata={"query": request.query, "mode": request.mode},
        )
        if langfuse_service
        else None
    )

    initial_state: AgentState = {
        "original_query": request.query,
        "current_query": request.query,
        "retrieved_chunks": [],
        "grading_result": "none",
        "rewrite_count": 0,
        "final_answer": "",
        "sources": [],
        "reasoning_steps": [],
        "rejected": False,
        "took_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    graph = build_agentic_rag_graph(
        rag_pipeline=pipeline,
        llm_client=llm,
    )

    final_state: AgentState = graph.invoke(initial_state)
    took_ms = round((time.perf_counter() - start_time) * 1000, 2)

    if langfuse_service and trace:
        langfuse_service.flush()

    raw_steps = final_state.get("reasoning_steps", [])
    reasoning_steps = [
        ReasoningStep(
            node=step.get("node", ""),
            decision=step.get("decision", ""),
            detail=step.get("detail", ""),
        )
        for step in raw_steps
    ]

    retrieved_chunks = final_state.get("retrieved_chunks", [])
    sources = final_state.get("sources", [])

    return AgenticAskResponse(
        answer=final_state.get("final_answer", ""),
        sources=sources,
        original_query=final_state.get("original_query", request.query),
        final_query=final_state.get("current_query", request.query),
        reasoning_steps=reasoning_steps,
        retrieved_chunk_count=len(retrieved_chunks),
        used_chunk_count=len(sources),
        rewrite_count=final_state.get("rewrite_count", 0),
        rejected=final_state.get("rejected", False),
        took_ms=took_ms,
        prompt_tokens=final_state.get("prompt_tokens", 0),
        completion_tokens=final_state.get("completion_tokens", 0),
        cached=False,
    )
