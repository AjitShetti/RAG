"""Unit tests for Guardrail Node in Agentic RAG workflow."""

from unittest.mock import MagicMock

import pytest

from src.services.agents.nodes.guardrail import OUT_OF_DOMAIN_ANSWER, guardrail_node
from src.services.llm.client import LLMResponse


def test_guardrail_node_in_domain():
    """Test in-domain CS/AI query returns rejected=False and appends 'in_domain' reasoning step."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="YES",
        model="llama-3.3-70b-versatile",
        prompt_tokens=20,
        completion_tokens=1,
        total_tokens=21,
        latency_ms=50.0,
    )

    state = {
        "original_query": "How do transformer attention mechanisms scale with sequence length?",
        "current_query": "How do transformer attention mechanisms scale with sequence length?",
    }

    result = guardrail_node(state, llm_client=mock_llm)

    assert result["rejected"] is False
    assert len(result["reasoning_steps"]) == 1
    assert result["reasoning_steps"][0]["node"] == "guardrail"
    assert result["reasoning_steps"][0]["decision"] == "in_domain"
    mock_llm.generate.assert_called_once()


def test_guardrail_node_out_of_domain():
    """Test out-of-domain query returns rejected=True and appends 'out_of_domain' reasoning step."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="NO",
        model="llama-3.3-70b-versatile",
        prompt_tokens=20,
        completion_tokens=1,
        total_tokens=21,
        latency_ms=50.0,
    )

    state = {
        "original_query": "what is the capital of France?",
        "current_query": "what is the capital of France?",
    }

    result = guardrail_node(state, llm_client=mock_llm)

    assert result["rejected"] is True
    assert result["final_answer"] == OUT_OF_DOMAIN_ANSWER
    assert len(result["reasoning_steps"]) == 1
    assert result["reasoning_steps"][0]["node"] == "guardrail"
    assert result["reasoning_steps"][0]["decision"] == "out_of_domain"
    mock_llm.generate.assert_called_once()
