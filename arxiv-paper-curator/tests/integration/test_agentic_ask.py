"""Integration tests for POST /api/v1/agentic-ask endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.schemas.rag.agentic_ask import AgenticAskResponse, ReasoningStep
from src.schemas.rag.ask import SourceChunk
from src.services.llm.client import LLMResponse

client = TestClient(app)


def test_agentic_ask_valid_in_domain_question():
    """Test valid in-domain question returns HTTP 200 with AgenticAskResponse, non-empty reasoning_steps, and sources."""
    mock_source = SourceChunk(
        paper_id="2301.00001",
        title="Attention in Deep Learning",
        section_name="Abstract",
        snippet="Self-attention mechanisms calculate token weight matrices.",
        relevance_score=0.92,
        pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
    )

    mock_agent_response = AgenticAskResponse(
        answer="Self-attention allows transformers to dynamically weight token relevance.",
        sources=[mock_source],
        original_query="How does self-attention work in transformers?",
        final_query="How does self-attention work in transformers?",
        rewrite_count=0,
        reasoning_steps=[
            ReasoningStep(node="guardrail", decision="in_domain", detail="CS research query"),
            ReasoningStep(node="retrieve", decision="retrieved", detail="Retrieved 5 chunks"),
            ReasoningStep(node="grade", decision="strong", detail="Chunks sufficient"),
            ReasoningStep(node="generate", decision="answered", detail="Generated answer"),
        ],
        rejected=False,
        took_ms=150.0,
        prompt_tokens=100,
        completion_tokens=25,
        cached=False,
    )

    with patch("src.routers.agentic_ask.run_agentic_rag", return_value=mock_agent_response):
        # Override cache to avoid Redis connection requirement during test
        with patch("src.routers.agentic_ask.CacheService") as mock_cache_cls:
            mock_cache_instance = MagicMock()
            mock_cache_instance.get_agentic.return_value = None
            mock_cache_cls.return_value = mock_cache_instance

            response = client.post(
                "/api/v1/agentic-ask",
                json={"query": "How does self-attention work in transformers?"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["rejected"] is False
            assert data["answer"] == "Self-attention allows transformers to dynamically weight token relevance."
            assert len(data["reasoning_steps"]) == 4
            assert data["reasoning_steps"][0]["node"] == "guardrail"
            assert data["reasoning_steps"][0]["decision"] == "in_domain"
            assert len(data["sources"]) == 1
            assert data["sources"][0]["paper_id"] == "2301.00001"


def test_agentic_ask_out_of_domain_question():
    """Test out-of-domain question returns HTTP 200 with rejected=True and guardrail reasoning step."""
    mock_agent_response = AgenticAskResponse(
        answer="This question is out of domain. Please ask questions related to Computer Science.",
        sources=[],
        original_query="what is the capital of France?",
        final_query="what is the capital of France?",
        rewrite_count=0,
        reasoning_steps=[
            ReasoningStep(node="guardrail", decision="out_of_domain", detail="Rejected query as out-of-domain")
        ],
        rejected=True,
        took_ms=30.0,
        prompt_tokens=20,
        completion_tokens=15,
        cached=False,
    )

    with patch("src.routers.agentic_ask.run_agentic_rag", return_value=mock_agent_response):
        with patch("src.routers.agentic_ask.CacheService") as mock_cache_cls:
            mock_cache_instance = MagicMock()
            mock_cache_instance.get_agentic.return_value = None
            mock_cache_cls.return_value = mock_cache_instance

            response = client.post(
                "/api/v1/agentic-ask",
                json={"query": "what is the capital of France?"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["rejected"] is True
            assert len(data["reasoning_steps"]) == 1
            assert data["reasoning_steps"][0]["node"] == "guardrail"
            assert data["reasoning_steps"][0]["decision"] == "out_of_domain"
            assert "out of domain" in data["answer"].lower()


def test_agentic_ask_e2e_workflow_integration():
    """Test end-to-end FastAPI endpoint call executing full Agentic RAG graph with mocked LLM/retriever."""
    mock_llm_res = LLMResponse(
        content="YES",
        model="llama-3.3-70b-versatile",
        prompt_tokens=10,
        completion_tokens=1,
        total_tokens=11,
        latency_ms=20.0,
    )

    mock_gen_res = LLMResponse(
        content="Transformers utilize multi-head self-attention.",
        model="llama-3.3-70b-versatile",
        prompt_tokens=80,
        completion_tokens=15,
        total_tokens=95,
        latency_ms=100.0,
    )

    with patch("src.services.agents.agentic_rag.LLMClient") as mock_llm_cls, \
         patch("src.services.agents.agentic_rag.RAGPipeline") as mock_pipeline_cls, \
         patch("src.routers.agentic_ask.CacheService") as mock_cache_cls:

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [mock_llm_res, mock_llm_res, mock_gen_res]
        mock_llm_cls.return_value = mock_llm

        mock_chunk = MagicMock()
        mock_chunk.paper_id = "2301.00001"
        mock_chunk.title = "Attention in Deep Learning"
        mock_chunk.section_name = "Abstract"
        mock_chunk.text = "Self-attention mechanisms calculate token weight matrices."
        mock_chunk.score = 0.92

        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = [mock_chunk]
        mock_pipeline.max_context_tokens = 2048
        mock_pipeline.system_prompt = "System prompt"
        mock_pipeline.llm = mock_llm

        source_item = SourceChunk(
            paper_id="2301.00001",
            title="Attention in Deep Learning",
            section_name="Abstract",
            snippet="Self-attention mechanisms calculate token weight matrices.",
            relevance_score=0.92,
            pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
        )
        mock_pipeline._build_source_chunks.return_value = [source_item]
        mock_pipeline_cls.return_value = mock_pipeline

        mock_cache = MagicMock()
        mock_cache.get_agentic.return_value = None
        mock_cache_cls.return_value = mock_cache

        response = client.post(
            "/api/v1/agentic-ask",
            json={"query": "Explain self-attention in transformers"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["rejected"] is False
        assert data["answer"] == "Transformers utilize multi-head self-attention."
        assert len(data["reasoning_steps"]) >= 4
