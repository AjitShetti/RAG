"""Unit tests for Rewrite Node in Agentic RAG workflow."""

from unittest.mock import MagicMock

import pytest

from src.services.agents.nodes.rewrite import rewrite_node
from src.services.llm.client import LLMResponse


def test_rewrite_node_success():
    """Test query is rewritten to a new string and rewrite_count is incremented."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Transformer self-attention mechanism scaling laws",
        model="llama-3.3-70b-versatile",
        prompt_tokens=35,
        completion_tokens=8,
        total_tokens=43,
        latency_ms=75.0,
    )

    state = {
        "original_query": "attention models",
        "current_query": "attention models",
        "rewrite_count": 0,
    }

    result = rewrite_node(state, llm_client=mock_llm)

    assert result["current_query"] == "Transformer self-attention mechanism scaling laws"
    assert result["rewrite_count"] == 1
    assert len(result["reasoning_steps"]) == 1
    assert result["reasoning_steps"][0]["node"] == "rewrite"
    assert result["reasoning_steps"][0]["decision"] == "rewritten"
    assert "attempt 1" in result["reasoning_steps"][0]["detail"]
    mock_llm.generate.assert_called_once()


def test_rewrite_node_increment_multiple_attempts():
    """Test rewrite_count increments properly on subsequent rewrite attempts."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Deep learning transformer self-attention benchmarks",
        model="llama-3.3-70b-versatile",
        prompt_tokens=35,
        completion_tokens=8,
        total_tokens=43,
        latency_ms=75.0,
    )

    state = {
        "original_query": "attention models",
        "current_query": "Transformer self-attention mechanism scaling laws",
        "rewrite_count": 1,
    }

    result = rewrite_node(state, llm_client=mock_llm)

    assert result["current_query"] == "Deep learning transformer self-attention benchmarks"
    assert result["rewrite_count"] == 2
    assert result["reasoning_steps"][0]["node"] == "rewrite"
    assert "attempt 2" in result["reasoning_steps"][0]["detail"]
