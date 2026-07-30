"""Unit tests for TextChunker."""

import datetime

import pytest

from src.models.paper import Paper
from src.services.indexing.text_chunker import Chunk, TextChunker


@pytest.fixture
def chunker():
    return TextChunker(max_tokens=20, overlap_tokens=5)


def test_section_aware_chunking_preserves_section_names(chunker):
    """Test that section names are preserved in generated chunks."""
    paper = Paper(
        arxiv_id="2401.00001v1",
        title="Sample Title",
        authors=["Author One"],
        abstract="This is the abstract summary.",
        pdf_url="https://arxiv.org/pdf/2401.00001",
        published_date=datetime.date(2024, 1, 1),
        category="cs.AI",
        sections=[
            {"heading": "Introduction", "text": "This is introduction text."},
            {"heading": "Methods", "text": "This is methods text."},
        ],
        parse_status="success",
    )

    chunks = chunker.chunk_paper(paper)
    headings = [c.section_name for c in chunks]

    assert "Abstract" in headings
    assert "Introduction" in headings
    assert "Methods" in headings


def test_long_section_is_split_with_overlap(chunker):
    """Test that a long section is split into multiple overlapping chunks."""
    long_text = " ".join([f"word{i}" for i in range(40)])
    paper = Paper(
        arxiv_id="2401.00002v1",
        title="Long Section Title",
        authors=["Author Two"],
        abstract="",
        pdf_url="",
        published_date=datetime.date(2024, 1, 1),
        category="cs.AI",
        sections=[{"heading": "Long Section", "text": long_text}],
        parse_status="success",
    )

    chunks = chunker.chunk_paper(paper)
    long_chunks = [c for c in chunks if c.section_name == "Long Section"]

    assert len(long_chunks) > 1
    # Verify overlap: end of first chunk should overlap start of second chunk
    words_1 = long_chunks[0].text.split()
    words_2 = long_chunks[1].text.split()
    assert words_1[-5:] == words_2[:5]


def test_references_section_skipped(chunker):
    """Test that References and Acknowledgments sections are excluded from chunking."""
    paper = Paper(
        arxiv_id="2401.00003v1",
        title="Title",
        authors=["Author Three"],
        abstract="Abstract",
        pdf_url="",
        published_date=datetime.date(2024, 1, 1),
        category="cs.AI",
        sections=[
            {"heading": "Introduction", "text": "Intro text."},
            {"heading": "References", "text": "1. Vaswani et al. Attention is all you need."},
            {"heading": "7. Acknowledgments", "text": "We thank our sponsors."},
        ],
        parse_status="success",
    )

    chunks = chunker.chunk_paper(paper)
    headings = [c.section_name for c in chunks]

    assert "Introduction" in headings
    assert "References" not in headings
    assert "7. Acknowledgments" not in headings


def test_missing_sections_falls_back_to_abstract(chunker):
    """Test that a paper with no parsed sections falls back to abstract."""
    paper = Paper(
        arxiv_id="2401.00004v1",
        title="No Sections Title",
        authors=["Author Four"],
        abstract="Abstract content only.",
        pdf_url="",
        published_date=datetime.date(2024, 1, 1),
        category="cs.AI",
        sections=None,
        parse_status="success",
    )

    chunks = chunker.chunk_paper(paper)
    assert len(chunks) == 1
    assert chunks[0].section_name == "Abstract"
    assert chunks[0].text == "Abstract content only."


def test_short_section_not_split(chunker):
    """Test that a short section fitting max_tokens is not split into multiple chunks."""
    paper = Paper(
        arxiv_id="2401.00005v1",
        title="Short Section Title",
        authors=["Author Five"],
        abstract="",
        pdf_url="",
        published_date=datetime.date(2024, 1, 1),
        category="cs.AI",
        sections=[{"heading": "Conclusion", "text": "Short conclusion text."}],
        parse_status="success",
    )

    chunks = chunker.chunk_paper(paper)
    conclusion_chunks = [c for c in chunks if c.section_name == "Conclusion"]
    assert len(conclusion_chunks) == 1
