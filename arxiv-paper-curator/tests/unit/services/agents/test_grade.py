"""Unit tests for Grade Node in Agentic RAG workflow."""

from unittest.mock import MagicMock

import pytest

from src.services.agents.nodes.grade import grade_node
from src.services.llm.client import LLMResponse


def test_grade_node_relevant_chunks_strong():
    """Test relevant chunks evaluated as YES by LLM produce grading_result='strong'."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="YES",
        model="llama-3.3-70b-versatile",
        prompt_tokens=40,
        completion_tokens=1,
        total_tokens=41,
        latency_ms=60.0,
    )

    mock_chunk = MagicMock()
    mock_chunk.score = 0.85
    mock_chunk.text = "Transformer architecture uses multi-head self-attention mechanisms."

    state = {
        "current_query": "What is self-attention?",
        "retrieved_chunks": [mock_chunk],
    }

    result = grade_node(state, llm_client=mock_llm)

    assert result["grading_result"] == "strong"
    assert len(result["reasoning_steps"]) == 1
    assert result["reasoning_steps"][0]["node"] == "grade"
    assert result["reasoning_steps"][0]["decision"] == "strong"
    mock_llm.generate.assert_called_once()


def test_grade_node_empty_chunks_weak():
    """Test empty retrieved_chunks produce grading_result='weak' without LLM call."""
    mock_llm = MagicMock()

    state = {
        "current_query": "What is self-attention?",
        "retrieved_chunks": [],
    }

    result = grade_node(state, llm_client=mock_llm)

    assert result["grading_result"] == "weak"
    assert len(result["reasoning_steps"]) == 1
    assert result["reasoning_steps"][0]["node"] == "grade"
    assert result["reasoning_steps"][0]["decision"] == "weak"
    mock_llm.generate.assert_not_called()


def test_grade_node_irrelevant_chunks_weak():
    """Test irrelevant chunks evaluated as NO by LLM produce grading_result='weak'."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="NO",
        model="llama-3.3-70b-versatile",
        prompt_tokens=40,
        completion_tokens=1,
        total_tokens=41,
        latency_ms=60.0,
    )

    mock_chunk = MagicMock()
    mock_chunk.score = 0.12
    mock_chunk.text = "This paper discusses agricultural yield in southern France."

    state = {
        "current_query": "Explain quantum computing algorithms",
        "retrieved_chunks": [mock_chunk],
    }

    result = grade_node(state, llm_client=mock_llm)

    assert result["grading_result"] == "weak"
    assert len(result["reasoning_steps"]) == 1
    assert result["reasoning_steps"][0]["node"] == "grade"
    assert result["reasoning_steps"][0]["decision"] == "weak"
    mock_llm.generate.assert_called_once()
