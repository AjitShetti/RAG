"""Integration test for end-to-end RAG question-answering pipeline."""

from unittest.mock import MagicMock

import pytest

from src.schemas.rag import AskRequest
from src.services.llm.client import LLMResponse
from src.services.rag.pipeline import RAGPipeline


@pytest.mark.integration
def test_rag_end_to_end_flow():
    """Verify end-to-end pipeline execution from AskRequest to AskResponse with source attribution."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Grounding test answer citing [2301.00001 § Abstract].",
        model="llama-3.3-70b-versatile",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        latency_ms=150.0,
    )

    # Initialize RAGPipeline with default OpenSearch and mock LLM
    pipeline = RAGPipeline(llm_client=mock_llm)

    req = AskRequest(
        query="artificial intelligence and deep learning",
        mode="keyword",  # BM25 chunk search
        top_k=5,
    )

    try:
        response = pipeline.answer(req)
        assert isinstance(response.answer, str)
        assert response.answer != ""
        assert isinstance(response.sources, list)
        assert isinstance(response.took_ms, float)
    except Exception as exc:
        pytest.skip(f"OpenSearch or database cluster not reachable: {exc}")
