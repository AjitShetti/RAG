"""Unit tests for prompt builder utilities."""

from datetime import date
from src.schemas.search import HybridSearchHit
from src.services.rag.prompt_builder import (
    build_context_block,
    build_messages,
    estimate_tokens,
    format_chunk_context,
)


def make_dummy_hit(chunk_id: str, paper_id: str, text: str, score: float = 0.5) -> HybridSearchHit:
    return HybridSearchHit(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section_name="Introduction",
        chunk_index=0,
        text=text,
        title="Test Paper Title",
        authors=["Alice Smith", "Bob Jones"],
        category="cs.AI",
        published_date=date(2023, 1, 1),
        pdf_url="http://example.com/pdf",
        score=score,
    )


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_format_chunk_context():
    hit = make_dummy_hit("c1", "2301.00001", "This is the chunk content.")
    formatted = format_chunk_context(hit, 1)
    assert "--- [Context Item 1] ---" in formatted
    assert "Paper ID: 2301.00001" in formatted
    assert "Title: Test Paper Title" in formatted
    assert "Section: Introduction" in formatted
    assert "This is the chunk content." in formatted


def test_build_context_block_token_budget_respected():
    # 5 chunks of ~400 characters each (~100 tokens each formatted)
    chunks = [
        make_dummy_hit(f"c{i}", f"2301.0000{i}", f"Content chunk number {i} " + ("x" * 300))
        for i in range(5)
    ]

    # Set tight max_tokens budget of 250 tokens -> only ~2 chunks should fit
    context_str, used_chunks = build_context_block(chunks, max_tokens=250)

    assert len(used_chunks) < 5
    assert len(used_chunks) > 0
    assert "2301.00000" in context_str


def test_build_context_block_empty():
    context_str, used_chunks = build_context_block([], max_tokens=1000)
    assert context_str == ""
    assert used_chunks == []


def test_build_messages():
    system_prompt = "You are a research assistant."
    context = "Context chunk text here."
    question = "What is the method?"

    messages = build_messages(system_prompt, context, question)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompt
    assert messages[1]["role"] == "user"
    assert "RETRIEVED CONTEXT CHUNKS:" in messages[1]["content"]
    assert question in messages[1]["content"]
