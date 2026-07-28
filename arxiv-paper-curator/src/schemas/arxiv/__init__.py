"""Pydantic schemas for arXiv paper metadata returned by the arXiv API.

These are data transfer objects (DTOs) — not ORM models.
ArxivPaperMetadata is produced by ArxivClient and consumed by MetadataFetcher.
"""

import datetime
from pydantic import BaseModel, Field, HttpUrl


class ArxivPaperMetadata(BaseModel):
    """Structured metadata for a single arXiv paper, as returned by the API."""

    arxiv_id: str = Field(
        ...,
        description="Short arXiv identifier, e.g. '2401.00001' (entry_id stripped of URL prefix)",
    )
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(..., description="List of author full names")
    abstract: str = Field(..., description="Paper abstract text")
    pdf_url: str = Field(..., description="Direct URL to the PDF file")
    published_date: datetime.date = Field(..., description="Date the paper was first submitted")
    category: str = Field(
        ...,
        description="Primary arXiv category, e.g. 'cs.AI'",
    )

    model_config = {"frozen": True}
