"""MetadataFetcher — the single orchestration point for arXiv paper ingestion.

Wires together ArxivClient (fetch), PdfParserService (parse), and
PaperRepository (store). Neither client nor parser knows about each other
or about the database, keeping them independently testable.

Can be invoked in two ways:
  1. From an Airflow PythonOperator:
       from src.services.metadata_fetcher import run_ingestion
       run_ingestion(query="cs.AI", max_results=20)

  2. Directly from the command line (manual / smoke-test run):
       python -m src.services.metadata_fetcher --query "cs.AI" --max-results 5

Both paths use the same database session and the same business logic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_session, init_db
from ..repositories.paper import PaperRepository
from ..schemas.arxiv import ArxivPaperMetadata
from .arxiv import ArxivClient
from .pdf_parser import PdfParserService

logger = logging.getLogger(__name__)


# ── Result summary ─────────────────────────────────────────────────────────────

@dataclass
class IngestionSummary:
    """Counts returned by a single MetadataFetcher.run() call."""

    fetched: int = 0
    skipped: int = 0   # already in DB
    stored: int = 0
    failed: int = 0    # parse failed but metadata stored with parse_status='failed'

    def __str__(self) -> str:
        return (
            f"fetched={self.fetched} skipped={self.skipped} "
            f"stored={self.stored} failed_parse={self.failed}"
        )


# ── Core service ───────────────────────────────────────────────────────────────

class MetadataFetcher:
    """Orchestrates arXiv fetch → PDF parse → Postgres upsert for a batch of papers.

    Args:
        session: An open SQLAlchemy :class:`Session`. The caller owns the
            transaction lifecycle — ``MetadataFetcher`` flushes but does not commit.
        arxiv_client: Defaults to a fresh :class:`ArxivClient` if not provided.
        pdf_parser: Defaults to a fresh :class:`PdfParserService` if not provided.
    """

    def __init__(
        self,
        session: Session,
        arxiv_client: ArxivClient | None = None,
        pdf_parser: PdfParserService | None = None,
    ) -> None:
        self._session = session
        self._client = arxiv_client or ArxivClient()
        self._parser = pdf_parser or PdfParserService()
        self._repo = PaperRepository(session)

    # ── Public entry point ────────────────────────────────────

    def run(
        self,
        query: str | None = None,
        max_results: int | None = None,
    ) -> IngestionSummary:
        """Fetch, parse, and store papers for *query*.

        Existing papers (matched by ``arxiv_id``) are skipped to ensure idempotency.

        Args:
            query: arXiv query string. Defaults to ``settings.arxiv_default_query``.
            max_results: Max papers to process. Defaults to ``settings.arxiv_max_results``.

        Returns:
            :class:`IngestionSummary` with counts for this run.
        """
        query = query or settings.arxiv_default_query
        max_results = max_results or settings.arxiv_max_results
        summary = IngestionSummary()

        # ── Stage 1: Fetch metadata from arXiv API ────────────
        logger.info("[MetadataFetcher] Stage 1 — fetching arXiv metadata: query=%r max=%d", query, max_results)
        papers: list[ArxivPaperMetadata] = self._client.search(
            query=query, max_results=max_results
        )
        summary.fetched = len(papers)
        logger.info("[MetadataFetcher] Fetched %d papers", summary.fetched)

        # ── Stage 2: Parse PDFs + Store to Postgres ───────────
        logger.info("[MetadataFetcher] Stage 2 — parsing PDFs and storing to Postgres")
        for paper in papers:
            self._process_paper(paper, summary)

        logger.info("[MetadataFetcher] Run complete: %s", summary)
        return summary

    # ── Internal helpers ──────────────────────────────────────

    def _process_paper(self, paper: ArxivPaperMetadata, summary: IngestionSummary) -> None:
        """Process a single paper: skip if exists, otherwise parse + store."""
        # Idempotency check
        if self._repo.exists(paper.arxiv_id):
            logger.debug("[%s] Already in DB — skipping", paper.arxiv_id)
            summary.skipped += 1
            return

        logger.info("[%s] Processing: %r", paper.arxiv_id, paper.title[:60])

        # Parse PDF
        parsed = self._parser.parse(paper.pdf_url, arxiv_id=paper.arxiv_id)

        if parsed is not None:
            parse_status = "success"
            full_text = parsed.full_text
            sections = parsed.sections or None
            parse_error = None
            logger.info("[%s] PDF parsed successfully (%d chars)", paper.arxiv_id, len(full_text))
        else:
            parse_status = "failed"
            full_text = None
            sections = None
            parse_error = "PDF parsing returned None — see logs for details"
            logger.warning("[%s] PDF parse failed — storing metadata without full text", paper.arxiv_id)
            summary.failed += 1

        # Upsert to Postgres
        self._repo.upsert(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            pdf_url=paper.pdf_url,
            published_date=paper.published_date,
            category=paper.category,
            full_text=full_text,
            sections=sections,
            parse_status=parse_status,
            parse_error=parse_error,
        )
        summary.stored += 1
        logger.info("[%s] Stored with parse_status=%r", paper.arxiv_id, parse_status)


# ── Module-level helper for Airflow tasks ──────────────────────────────────────

def run_ingestion(
    query: str | None = None,
    max_results: int | None = None,
) -> IngestionSummary:
    """Convenience function that creates a session and runs MetadataFetcher.

    Designed to be called directly from an Airflow PythonOperator::

        from src.services.metadata_fetcher import run_ingestion
        run_ingestion(query="cs.AI", max_results=20)

    Commits the session on success; rolls back on error.
    """
    with get_session() as session:
        fetcher = MetadataFetcher(session=session)
        return fetcher.run(query=query, max_results=max_results)


# ── CLI entry point ────────────────────────────────────────────────────────────

def _cli() -> None:
    """Run ingestion from the command line for manual / smoke-test invocations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run arXiv paper ingestion manually")
    parser.add_argument(
        "--query",
        default=settings.arxiv_default_query,
        help="arXiv query string (default: %(default)s)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=settings.arxiv_max_results,
        help="Max papers to fetch (default: %(default)s)",
    )
    args = parser.parse_args()

    logger.info("Initialising database tables …")
    init_db()

    logger.info("Starting ingestion: query=%r max_results=%d", args.query, args.max_results)
    summary = run_ingestion(query=args.query, max_results=args.max_results)
    print(f"\n✅  Ingestion complete — {summary}")


if __name__ == "__main__":
    _cli()
