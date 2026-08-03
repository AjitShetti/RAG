"""Unit tests for RAGPipeline orchestrator."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.schemas.rag import AskRequest
from src.schemas.search import HybridSearchHit, HybridSearchResponse
from src.services.llm.client import LLMResponse
from src.services.rag.pipeline import NO_CONTEXT_ANSWER, RAGPipeline


def make_dummy_hit(chunk_id: str = "c1", score: float = 0.5) -> HybridSearchHit:
    return HybridSearchHit(
        chunk_id=chunk_id,
        paper_id="2301.00001",
        section_name="Abstract",
        chunk_index=0,
        text="The attention mechanism is the core component of Transformer models.",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        category="cs.CL",
        published_date=date(2017, 6, 12),
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        score=score,
    )


@pytest.fixture
def mock_opensearch():
    service = MagicMock()
    return service


@pytest.fixture
def mock_llm():
    client = MagicMock()
    return client


def test_rag_pipeline_answer_success(mock_opensearch, mock_llm):
    # Mock OpenSearch returning 1 hit
    hit = make_dummy_hit("c1", score=0.8)
    mock_opensearch.hybrid_search.return_value = HybridSearchResponse(
        total=1,
        page=1,
        page_size=8,
        mode="hybrid",
        took_ms=10.0,
        results=[hit],
    )

    # Mock LLM returning response
    mock_llm.generate.return_value = LLMResponse(
        content="Attention mechanisms allow models to focus on specific input elements [2301.00001 § Abstract].",
        model="llama-3.3-70b-versatile",
        prompt_tokens=150,
        completion_tokens=30,
        total_tokens=180,
        latency_ms=250.0,
    )

    pipeline = RAGPipeline(
        opensearch_service=mock_opensearch,
        llm_client=mock_llm,
    )

    req = AskRequest(query="How does attention work?")
    res = pipeline.answer(req)

    assert res.answer.startswith("Attention mechanisms allow")
    assert len(res.sources) == 1
    assert res.sources[0].paper_id == "2301.00001"
    assert res.retrieved_chunk_count == 1
    assert res.used_chunk_count == 1
    assert res.prompt_tokens == 150
    assert mock_llm.generate.called


def test_rag_pipeline_zero_results_handling(mock_opensearch, mock_llm):
    # Mock OpenSearch returning 0 hits
    mock_opensearch.hybrid_search.return_value = HybridSearchResponse(
        total=0,
        page=1,
        page_size=8,
        mode="hybrid",
        took_ms=5.0,
        results=[],
    )

    pipeline = RAGPipeline(
        opensearch_service=mock_opensearch,
        llm_client=mock_llm,
    )

    req = AskRequest(query="Non-existent topic xyz123?")
    res = pipeline.answer(req)

    assert res.answer == NO_CONTEXT_ANSWER
    assert res.sources == []
    assert res.retrieved_chunk_count == 0
    assert res.used_chunk_count == 0
    # LLM should NEVER be called when 0 chunks are retrieved
    assert not mock_llm.generate.called


@pytest.mark.asyncio
async def test_rag_pipeline_stream_zero_results(mock_opensearch, mock_llm):
    mock_opensearch.hybrid_search.return_value = HybridSearchResponse(
        total=0,
        page=1,
        page_size=8,
        mode="hybrid",
        took_ms=5.0,
        results=[],
    )

    pipeline = RAGPipeline(
        opensearch_service=mock_opensearch,
        llm_client=mock_llm,
    )

    req = AskRequest(query="Unknown question?")
    items = []
    async for item in pipeline.answer_stream(req):
        items.append(item)

    assert len(items) == 2
    assert items[0]["event"] == "token"
    assert items[0]["data"] == NO_CONTEXT_ANSWER
    assert items[1]["event"] == "metadata"
    assert items[1]["data"]["retrieved_chunk_count"] == 0
