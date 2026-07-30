"""One-off / periodic script to bulk index papers from PostgreSQL into OpenSearch.

Reads all successfully parsed Paper records from Postgres, ensures the OpenSearch index
and mapping exist, and bulk-indexes them into OpenSearch.
Safe to re-run anytime (idempotent upsert by arxiv_id).
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from src.database import get_session
from src.models.paper import Paper
from src.services.opensearch import OpenSearchService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("reindex_papers")


def reindex_all_papers() -> None:
    """Read all papers from Postgres and bulk-index them into OpenSearch."""
    opensearch_service = OpenSearchService()

    logger.info("Step 1: Ensuring OpenSearch index exists...")
    opensearch_service.create_index()

    logger.info("Step 2: Querying successfully parsed papers from PostgreSQL...")
    with get_session() as session:
        stmt = select(Paper).where(Paper.parse_status == "success")
        papers = list(session.scalars(stmt).all())
        logger.info("Found %d successfully parsed papers in PostgreSQL.", len(papers))

        if not papers:
            logger.info("No papers found to index.")
            return

        logger.info("Step 3: Bulk indexing %d papers into OpenSearch...", len(papers))
        success_count, errors = opensearch_service.bulk_index(papers)

        print(
            f"\n[OK] Bulk indexing complete:\n"
            f"  Total papers processed : {len(papers)}\n"
            f"  Successfully indexed   : {success_count}\n"
            f"  Failed index errors    : {len(errors) if isinstance(errors, list) else errors}\n"
        )


if __name__ == "__main__":
    reindex_all_papers()
