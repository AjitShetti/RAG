"""Unit tests for LangGraph Agentic RAG workflow orchestration."""

from unittest.mock import MagicMock

import pytest

from src.schemas.rag.ask import SourceChunk
from src.services.agents.agentic_rag import build_agentic_rag_graph
from src.services.agents.state import AgentState
from src.services.llm.client import LLMResponse


def test_agentic_graph_out_of_domain_short_circuit():
    """Test out-of-domain query short-circuits at guardrail node and retrieve is never invoked."""
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    # Guardrail LLM returns "NO" -> out-of-domain
    mock_llm.generate.return_value = LLMResponse(
        content="NO",
        model="llama-3.3-70b-versatile",
        prompt_tokens=10,
        completion_tokens=1,
        total_tokens=11,
        latency_ms=10.0,
    )

    graph = build_agentic_rag_graph(rag_pipeline=mock_pipeline, llm_client=mock_llm)

    initial_state: AgentState = {
        "original_query": "What is the capital of France?",
        "current_query": "What is the capital of France?",
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

    final_state = graph.invoke(initial_state)

    assert final_state["rejected"] is True
    assert "out of domain" in final_state["final_answer"].lower()
    assert len(final_state["reasoning_steps"]) == 1
    assert final_state["reasoning_steps"][0]["node"] == "guardrail"
    assert final_state["reasoning_steps"][0]["decision"] == "out_of_domain"

    # Crucial assertion: retrieve node never invoked
    mock_pipeline.retrieve.assert_not_called()


def test_agentic_graph_strong_grading_direct_path():
    """Test in-domain query with strong relevance proceeds directly from grade to generate."""
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    # LLM calls:
    # 1. Guardrail -> "YES"
    # 2. Grade -> "YES"
    # 3. Generate -> Answer text
    mock_llm.generate.side_effect = [
        LLMResponse(content="YES", model="m", prompt_tokens=10, completion_tokens=1, total_tokens=11, latency_ms=10.0),
        LLMResponse(content="YES", model="m", prompt_tokens=20, completion_tokens=1, total_tokens=21, latency_ms=15.0),
        LLMResponse(content="Attention models use QKV matrices.", model="m", prompt_tokens=100, completion_tokens=20, total_tokens=120, latency_ms=80.0),
    ]

    mock_chunk = MagicMock()
    mock_chunk.paper_id = "2301.00001"
    mock_chunk.title = "Attention Mechanisms"
    mock_chunk.section_name = "Abstract"
    mock_chunk.text = "Self-attention computes token similarity scores."
    mock_chunk.score = 0.90
    mock_pipeline.retrieve.return_value = [mock_chunk]
    mock_pipeline.max_context_tokens = 2048
    mock_pipeline.system_prompt = "System prompt"
    mock_pipeline.llm = mock_llm

    source_item = SourceChunk(
        paper_id="2301.00001",
        title="Attention Mechanisms",
        section_name="Abstract",
        snippet="Self-attention computes token similarity scores.",
        relevance_score=0.90,
        pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
    )
    mock_pipeline._build_source_chunks.return_value = [source_item]

    graph = build_agentic_rag_graph(rag_pipeline=mock_pipeline, llm_client=mock_llm)

    initial_state: AgentState = {
        "original_query": "Explain self-attention in transformers",
        "current_query": "Explain self-attention in transformers",
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

    final_state = graph.invoke(initial_state)

    assert final_state["rejected"] is False
    assert final_state["grading_result"] == "strong"
    assert final_state["rewrite_count"] == 0
    assert final_state["final_answer"] == "Attention models use QKV matrices."
    assert mock_pipeline.retrieve.call_count == 1

    nodes_in_order = [step["node"] for step in final_state["reasoning_steps"]]
    assert nodes_in_order == ["guardrail", "retrieve", "grade", "generate"]


def test_agentic_graph_weak_grading_rewrite_loop():
    """Test weak grading triggers rewrite loop up to MAX_REWRITES=2 and then proceeds to generate."""
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    # Sequence of LLM calls:
    # 1. Guardrail -> "YES"
    # 2. Grade #1 -> "NO" (weak)
    # 3. Rewrite #1 -> "rewritten query 1"
    # 4. Grade #2 -> "NO" (weak)
    # 5. Rewrite #2 -> "rewritten query 2"
    # 6. Grade #3 -> "NO" (weak) -> MAX_REWRITES reached (2), route to generate
    # 7. Generate -> fallback/answer
    mock_llm.generate.side_effect = [
        LLMResponse(content="YES", model="m", prompt_tokens=10, completion_tokens=1, total_tokens=11, latency_ms=10.0),
        LLMResponse(content="NO", model="m", prompt_tokens=20, completion_tokens=1, total_tokens=21, latency_ms=15.0),
        LLMResponse(content="rewritten query 1", model="m", prompt_tokens=30, completion_tokens=5, total_tokens=35, latency_ms=20.0),
        LLMResponse(content="NO", model="m", prompt_tokens=20, completion_tokens=1, total_tokens=21, latency_ms=15.0),
        LLMResponse(content="rewritten query 2", model="m", prompt_tokens=30, completion_tokens=5, total_tokens=35, latency_ms=20.0),
        LLMResponse(content="NO", model="m", prompt_tokens=20, completion_tokens=1, total_tokens=21, latency_ms=15.0),
        LLMResponse(content="Final generated response after rewrites.", model="m", prompt_tokens=100, completion_tokens=20, total_tokens=120, latency_ms=80.0),
    ]

    mock_chunk = MagicMock()
    mock_chunk.score = 0.3
    mock_chunk.text = "Low relevance text"
    mock_pipeline.retrieve.return_value = [mock_chunk]
    mock_pipeline.max_context_tokens = 2048
    mock_pipeline.system_prompt = "System prompt"
    mock_pipeline.llm = mock_llm
    mock_pipeline._build_source_chunks.return_value = []

    graph = build_agentic_rag_graph(rag_pipeline=mock_pipeline, llm_client=mock_llm)

    initial_state: AgentState = {
        "original_query": "vague query",
        "current_query": "vague query",
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

    final_state = graph.invoke(initial_state)

    assert final_state["rejected"] is False
    assert final_state["rewrite_count"] == 2
    assert final_state["current_query"] == "rewritten query 2"
    assert mock_pipeline.retrieve.call_count == 3  # Initial + 2 rewrites

    nodes_in_order = [step["node"] for step in final_state["reasoning_steps"]]
    assert nodes_in_order == [
        "guardrail",
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
    ]
