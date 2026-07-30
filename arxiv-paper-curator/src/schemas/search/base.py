"""Pydantic schemas for basic BM25 keyword search requests and responses."""

from datetime import date
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request payload for BM25 keyword search."""

    query: str = Field(..., min_length=1, description="Keywords to search for")
    category: str | None = Field(None, description="Filter by arXiv category (e.g. 'cs.AI')")
    date_from: date | None = Field(None, description="Filter papers published on or after this date")
    date_to: date | None = Field(None, description="Filter papers published on or before this date")
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(10, ge=1, le=50, description="Results per page")


class SearchHit(BaseModel):
    """Single ranked search result item."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    category: str
    published_date: date | None = None
    pdf_url: str
    score: float = Field(..., description="BM25 relevance score")
    highlights: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Matched snippet fragments keyed by field name",
    )


class SearchResponse(BaseModel):
    """Response payload for BM25 keyword search."""

    total: int = Field(..., description="Total matching documents count")
    page: int
    page_size: int
    took_ms: float = Field(..., description="Search query execution latency in milliseconds")
    results: list[SearchHit]
