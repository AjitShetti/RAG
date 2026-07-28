"""SQLAlchemy ORM model for arXiv papers.

Stores structured metadata plus parsed PDF content. The `parse_status` field
tracks whether PDF parsing succeeded, failed, or has not been attempted yet.
`sections` is a JSONB array of {heading, text} dicts produced by PdfParserService.
`authors` is a JSONB array of author name strings.
"""

import datetime
from typing import Any

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class Paper(Base):
    """Persisted arXiv paper with parsed PDF content."""

    __tablename__ = "papers"

    # ── Identity ──────────────────────────────────────────────
    arxiv_id: Mapped[str] = mapped_column(
        String(32), primary_key=True, index=True,
        comment="arXiv identifier, e.g. '2401.00001'",
    )

    # ── Metadata from arXiv API ───────────────────────────────
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="List of author name strings",
    )
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    pdf_url: Mapped[str] = mapped_column(Text, nullable=False)
    published_date: Mapped[datetime.date] = mapped_column(nullable=False)
    category: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
        comment="Primary arXiv category, e.g. 'cs.AI'",
    )

    # ── Parsed PDF content ────────────────────────────────────
    full_text: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Raw concatenated text extracted from the PDF",
    )
    sections: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB, nullable=True,
        comment="Structured sections: [{heading: str, text: str}, ...]",
    )
    parse_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="pending",
        index=True,
        comment="One of: pending | success | failed",
    )
    parse_error: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Error message if parse_status='failed'",
    )

    # ── Timestamps ────────────────────────────────────────────
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Paper arxiv_id={self.arxiv_id!r} title={self.title[:40]!r}>"
