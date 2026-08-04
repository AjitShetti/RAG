"""Unit tests for Agentic RAG graph state, nodes, and end-to-end execution flow."""

from unittest.mock import MagicMock

import pytest

from src.schemas.rag import AskRequest, SourceChunk
from src.schemas.search import HybridSearchHit
from src.services.agents.agentic_rag import build_agentic_rag_graph, run_agentic_rag
from src.services.agents.nodes.generate import generate_node
from src.services.agents.nodes.grade import grade_node
from src.services.agents.nodes.guardrail import guardrail_node
from src.services.agents.nodes.retrieve import retrieve_node
from src.services.agents.nodes.rewrite import rewrite_node
from src.services.agents.state import AgentState
from src.services.llm.client import LLMClient, LLMResponse
from src.services.rag.pipeline import RAGPipeline


@pytest.fixture
def mock_hit():
    return HybridSearchHit(
        chunk_id="c1",
        paper_id="2401.12345",
        title="Attention Is All You Need",
        section_name="Abstract",
        chunk_index=0,
        authors=["Vaswani et al."],
        category="cs.CL",
        text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        score=0.92,
        pdf_url="http://arxiv.org/pdf/2401.12345.pdf",
    )


@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.generate.return_value = LLMResponse(
        content="YES",
        model="llama-3.3-70b",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=100.0,
    )
    return client


@pytest.fixture
def mock_rag_pipeline(mock_hit):
    pipeline = MagicMock(spec=RAGPipeline)
    pipeline.retrieve.return_value = [mock_hit]
    pipeline.max_context_tokens = 3000
    pipeline.system_prompt = "System prompt"
    pipeline._build_source_chunks.return_value = [
        SourceChunk(
            paper_id=mock_hit.paper_id,
            title=mock_hit.title,
            section_name=mock_hit.section_name,
            snippet=mock_hit.text[:300],
            relevance_score=mock_hit.score,
            pdf_url=mock_hit.pdf_url,
        )
    ]
    pipeline.llm = MagicMock(spec=LLMClient)
    pipeline.llm.generate.return_value = LLMResponse(
        content="Transformers replace recurrence with self-attention.",
        model="llama-3.3-70b",
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
        latency_ms=200.0,
    )
    return pipeline


def test_guardrail_node_in_domain(mock_llm_client):
    mock_llm_client.generate.return_value.content = "YES"
    state: AgentState = {
        "original_query": "What is self-attention?",
        "current_query": "What is self-attention?",
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

    res = guardrail_node(state, llm_client=mock_llm_client)

    assert res["rejected"] is False
    assert len(res["reasoning_steps"]) == 1
    assert res["reasoning_steps"][0]["node"] == "guardrail"
    assert res["reasoning_steps"][0]["decision"] == "in_domain"


def test_guardrail_node_out_of_domain(mock_llm_client):
    mock_llm_client.generate.return_value.content = "NO"
    state: AgentState = {
        "original_query": "How do I bake a cake?",
        "current_query": "How do I bake a cake?",
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

    res = guardrail_node(state, llm_client=mock_llm_client)

    assert res["rejected"] is True
    assert "out of domain" in res["final_answer"].lower()
    assert res["reasoning_steps"][0]["decision"] == "out_of_domain"


def test_retrieve_node(mock_rag_pipeline, mock_hit):
    state: AgentState = {
        "original_query": "transformer architectures",
        "current_query": "transformer architectures",
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

    res = retrieve_node(state, rag_pipeline=mock_rag_pipeline)

    assert len(res["retrieved_chunks"]) == 1
    assert res["retrieved_chunks"][0].paper_id == "2401.12345"
    assert res["reasoning_steps"][0]["decision"] == "retrieved"


def test_grade_node_strong(mock_llm_client, mock_hit):
    mock_llm_client.generate.return_value.content = "YES"
    state: AgentState = {
        "original_query": "What is transformer architecture?",
        "current_query": "What is transformer architecture?",
        "retrieved_chunks": [mock_hit],
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

    res = grade_node(state, llm_client=mock_llm_client)

    assert res["grading_result"] == "strong"
    assert res["reasoning_steps"][0]["decision"] == "strong"


def test_grade_node_weak(mock_llm_client, mock_hit):
    mock_llm_client.generate.return_value.content = "NO"
    state: AgentState = {
        "original_query": "quantum computing",
        "current_query": "quantum computing",
        "retrieved_chunks": [mock_hit],
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

    res = grade_node(state, llm_client=mock_llm_client)

    assert res["grading_result"] == "weak"
    assert res["reasoning_steps"][0]["decision"] == "weak"


def test_rewrite_node(mock_llm_client):
    mock_llm_client.generate.return_value.content = "attention mechanisms in transformer neural networks"
    state: AgentState = {
        "original_query": "attn",
        "current_query": "attn",
        "retrieved_chunks": [],
        "grading_result": "weak",
        "rewrite_count": 0,
        "final_answer": "",
        "sources": [],
        "reasoning_steps": [],
        "rejected": False,
        "took_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    res = rewrite_node(state, llm_client=mock_llm_client)

    assert res["current_query"] == "attention mechanisms in transformer neural networks"
    assert res["rewrite_count"] == 1
    assert res["reasoning_steps"][0]["decision"] == "rewritten"


def test_generate_node(mock_rag_pipeline, mock_hit):
    state: AgentState = {
        "original_query": "What is transformer architecture?",
        "current_query": "What is transformer architecture?",
        "retrieved_chunks": [mock_hit],
        "grading_result": "strong",
        "rewrite_count": 0,
        "final_answer": "",
        "sources": [],
        "reasoning_steps": [],
        "rejected": False,
        "took_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    res = generate_node(state, rag_pipeline=mock_rag_pipeline)

    assert "Transformers replace recurrence" in res["final_answer"]
    assert len(res["sources"]) == 1
    assert res["prompt_tokens"] == 20
    assert res["completion_tokens"] == 10
    assert res["reasoning_steps"][0]["decision"] == "answered"


def test_run_agentic_rag_full_flow(mock_rag_pipeline, mock_llm_client):
    # Guardrail YES, Grade YES
    mock_llm_client.generate.side_effect = [
        LLMResponse("YES", "m", 5, 2, 7, 50.0),  # guardrail
        LLMResponse("YES", "m", 5, 2, 7, 50.0),  # grade
    ]

    request = AskRequest(query="Explain transformers in AI")
    res = run_agentic_rag(
        request=request,
        rag_pipeline=mock_rag_pipeline,
        llm_client=mock_llm_client,
    )

    assert res.rejected is False
    assert len(res.reasoning_steps) == 4  # guardrail -> retrieve -> grade -> generate
    assert res.reasoning_steps[0].node == "guardrail"
    assert res.reasoning_steps[1].node == "retrieve"
    assert res.reasoning_steps[2].node == "grade"
    assert res.reasoning_steps[3].node == "generate"
    assert res.answer == "Transformers replace recurrence with self-attention."
    assert len(res.sources) == 1
    assert res.rewrite_count == 0


def test_run_agentic_rag_rejected(mock_rag_pipeline, mock_llm_client):
    # Guardrail NO
    mock_llm_client.generate.return_value = LLMResponse("NO", "m", 5, 2, 7, 50.0)

    request = AskRequest(query="How to standardly fix a leaky faucet?")
    res = run_agentic_rag(
        request=request,
        rag_pipeline=mock_rag_pipeline,
        llm_client=mock_llm_client,
    )

    assert res.rejected is True
    assert len(res.reasoning_steps) == 1
    assert res.reasoning_steps[0].node == "guardrail"
    assert res.reasoning_steps[0].decision == "out_of_domain"
    assert "out of domain" in res.answer.lower()
