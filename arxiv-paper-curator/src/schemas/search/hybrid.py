"""Pydantic schemas for Hybrid Search (BM25 + kNN + RRF)."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class HybridSearchRequest(BaseModel):
    """Request payload for hybrid / keyword / semantic search."""

    query: str = Field(..., min_length=1, description="Search query string")
    mode: Literal["keyword", "semantic", "hybrid"] = Field(
        "hybrid",
        description="Retrieval mode: 'keyword' (BM25 only), 'semantic' (kNN vector only), or 'hybrid' (BM25 + kNN + RRF)",
    )
    category: str | None = Field(None, description="Filter by arXiv category (e.g. 'cs.AI')")
    date_from: date | None = Field(None, description="Filter papers published on or after this date")
    date_to: date | None = Field(None, description="Filter papers published on or before this date")
    group_by_paper: bool = Field(
        False,
        description="If True, populates grouped_results with chunks grouped under their parent paper.",
    )
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(10, ge=1, le=50, description="Results per page")


class HybridSearchHit(BaseModel):
    """Single matching chunk search result item."""

    chunk_id: str
    paper_id: str
    section_name: str
    chunk_index: int
    text: str
    title: str
    authors: list[str]
    category: str
    published_date: date | None = None
    pdf_url: str
    score: float = Field(..., description="Rank score (RRF score or raw search score)")
    keyword_rank: int | None = Field(None, description="1-indexed rank in BM25 results (if matched)")
    semantic_rank: int | None = Field(None, description="1-indexed rank in kNN vector results (if matched)")
    contributed_by: list[str] = Field(
        default_factory=list,
        description="List of retrieval modes that matched this chunk: ['keyword'], ['semantic'], or ['keyword', 'semantic']",
    )
    highlights: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Matched snippet fragments",
    )


class PaperGroupedHit(BaseModel):
    """Paper-level grouping container surfacing matching chunks."""

    paper_id: str
    title: str
    authors: list[str]
    category: str
    published_date: date | None = None
    pdf_url: str
    best_score: float
    chunks: list[HybridSearchHit]


class HybridSearchResponse(BaseModel):
    """Response payload for hybrid search."""

    total: int = Field(..., description="Total matching chunks count")
    page: int
    page_size: int
    mode: str = Field(..., description="Retrieval mode used ('keyword', 'semantic', 'hybrid')")
    took_ms: float = Field(..., description="Execution latency in milliseconds")
    results: list[HybridSearchHit] = Field(..., description="Raw chunk-level results for pipeline/RAG consumption")
    grouped_results: list[PaperGroupedHit] | None = Field(
        None,
        description="Opt-in paper-grouped results display (populated if group_by_paper=True)",
    )
