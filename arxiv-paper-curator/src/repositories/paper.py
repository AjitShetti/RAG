"""Data-access layer for the Paper aggregate.

PaperRepository wraps all direct SQLAlchemy queries for Paper rows,
keeping ORM details out of services. Both methods are idempotent and
safe to call from Airflow tasks or CLI scripts.
"""

import logging
from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from ..models.paper import Paper

logger = logging.getLogger(__name__)


class PaperRepository:
    """CRUD operations for the Paper table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Reads ─────────────────────────────────────────────────

    def exists(self, arxiv_id: str) -> bool:
        """Return True if a row with this arXiv ID already exists."""
        return self._session.get(Paper, arxiv_id) is not None

    def get(self, arxiv_id: str) -> Paper | None:
        """Fetch a Paper by arXiv ID, or None if not found."""
        return self._session.get(Paper, arxiv_id)

    # ── Writes ────────────────────────────────────────────────

    def upsert(
        self,
        *,
        arxiv_id: str,
        title: str,
        authors: list[str],
        abstract: str,
        pdf_url: str,
        published_date: date,
        category: str,
        full_text: str | None = None,
        sections: list[dict[str, Any]] | None = None,
        parse_status: str = "pending",
        parse_error: str | None = None,
    ) -> Paper:
        """Insert a new Paper or update the parsed-content fields of an existing one.

        Metadata fields (title, authors, etc.) are always refreshed.
        Returns the persisted Paper instance (not yet committed — caller commits).
        """
        paper = self._session.get(Paper, arxiv_id)
        if paper is None:
            paper = Paper(arxiv_id=arxiv_id)
            self._session.add(paper)
            logger.debug("Inserting new paper %s", arxiv_id)
        else:
            logger.debug("Updating existing paper %s", arxiv_id)

        paper.title = title
        paper.authors = authors
        paper.abstract = abstract
        paper.pdf_url = pdf_url
        paper.published_date = published_date
        paper.category = category
        paper.full_text = full_text
        paper.sections = sections
        paper.parse_status = parse_status
        paper.parse_error = parse_error

        self._session.flush()  # assign DB defaults (e.g. timestamps) without committing
        return paper
