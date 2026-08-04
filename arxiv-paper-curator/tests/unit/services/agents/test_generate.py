"""Unit tests for Generate Node in Agentic RAG workflow."""

from unittest.mock import MagicMock

import pytest

from src.schemas.rag.ask import SourceChunk
from src.services.agents.nodes.generate import generate_node
from src.services.llm.client import LLMResponse


def test_generate_node_with_chunks():
    """Test answer is generated, sources are formatted, and reasoning step 'answered' is recorded."""
    mock_pipeline = MagicMock()
    mock_pipeline.max_context_tokens = 2048
    mock_pipeline.system_prompt = "You are a helpful arXiv assistant."

    mock_llm_res = LLMResponse(
        content="Transformer networks use attention to model token dependencies.",
        model="llama-3.3-70b-versatile",
        prompt_tokens=150,
        completion_tokens=25,
        total_tokens=175,
        latency_ms=120.0,
    )
    mock_pipeline.llm.generate.return_value = mock_llm_res

    mock_chunk = MagicMock()
    mock_chunk.paper_id = "2301.00001"
    mock_chunk.title = "Attention in Deep Learning"
    mock_chunk.section_name = "Abstract"
    mock_chunk.text = "Self-attention mechanisms calculate pairwise token weights."
    mock_chunk.score = 0.92

    source_item = SourceChunk(
        paper_id="2301.00001",
        title="Attention in Deep Learning",
        section_name="Abstract",
        snippet="Self-attention mechanisms calculate pairwise token weights.",
        relevance_score=0.92,
        pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
    )
    mock_pipeline._build_source_chunks.return_value = [source_item]

    state = {
        "current_query": "Explain self-attention",
        "retrieved_chunks": [mock_chunk],
    }

    result = generate_node(state, rag_pipeline=mock_pipeline)

    assert result["final_answer"] == "Transformer networks use attention to model token dependencies."
    assert len(result["sources"]) == 1
    assert result["sources"][0].paper_id == "2301.00001"
    assert result["prompt_tokens"] == 150
    assert result["completion_tokens"] == 25
    assert len(result["reasoning_steps"]) == 1
    assert result["reasoning_steps"][0]["node"] == "generate"
    assert result["reasoning_steps"][0]["decision"] == "answered"
    assert "1 context chunks" in result["reasoning_steps"][0]["detail"]


def test_generate_node_empty_chunks():
    """Test generate node handles empty retrieved_chunks gracefully."""
    mock_pipeline = MagicMock()

    state = {
        "current_query": "Explain self-attention",
        "retrieved_chunks": [],
    }

    result = generate_node(state, rag_pipeline=mock_pipeline)

    assert "don't have enough information" in result["final_answer"]
    assert result["sources"] == []
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["reasoning_steps"][0]["node"] == "generate"
    assert result["reasoning_steps"][0]["decision"] == "answered"
    assert "0 context chunks" in result["reasoning_steps"][0]["detail"]
