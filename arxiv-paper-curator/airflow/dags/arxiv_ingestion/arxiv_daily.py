"""Airflow DAG: arxiv_paper_ingestion

Scheduled daily. Fetches new arXiv papers matching configurable queries,
parses their PDFs with pdfminer, stores structured rows in Postgres, then
indexes them into OpenSearch for hybrid RAG search.

Task graph:
  ensure_db_tables >> fetch_metadata >> parse_and_store >> index_opensearch >> log_summary

Environment variables:
  ARXIV_QUERY       — space-separated categories e.g. "cs.AI cs.LG cs.CL" (default: cs.AI)
  ARXIV_MAX_RESULTS — max papers per category per run (default: 20)
  DATABASE_URL      — sync postgres URL (psycopg2)
  OPENSEARCH_URL    — OpenSearch endpoint (default: http://opensearch:9200)

To trigger manually: unpause DAG -> Trigger DAG in Airflow UI at http://localhost:8090.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta

# Ensure the app source is on sys.path when running inside the Airflow container.
# The container mounts arxiv-paper-curator/ to /opt/airflow/app.
_APP_ROOT = os.environ.get("APP_ROOT", "/opt/airflow/app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "arxiv-curator",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}


# ── Task callables ─────────────────────────────────────────────────────────────

def _ensure_db_tables(**context) -> None:
    """Create DB tables if they don't exist (idempotent)."""
    from src.database import init_db
    init_db()
    logger.info("DB tables verified.")


def _fetch_metadata(**context) -> dict:
    """Fetch paper metadata from arXiv across all configured categories."""
    from src.services.arxiv import ArxivClient

    # Support multiple space-separated categories e.g. "cs.AI cs.LG cs.CL"
    raw_query = os.environ.get("ARXIV_QUERY", "cs.AI")
    categories = raw_query.split()
    max_results = int(os.environ.get("ARXIV_MAX_RESULTS", "20"))

    client = ArxivClient()
    all_papers = []
    seen_ids: set[str] = set()

    for category in categories:
        logger.info("Fetching arXiv category=%r max_results=%d", category, max_results)
        papers = client.search(query=category, max_results=max_results)
        for p in papers:
            if p.arxiv_id not in seen_ids:
                seen_ids.add(p.arxiv_id)
                all_papers.append(p)

    logger.info(
        "Fetched %d unique papers across %d categories", len(all_papers), len(categories)
    )
    return {
        "query": raw_query,
        "max_results": max_results,
        "papers": [p.model_dump(mode="json") for p in all_papers],
    }


def _parse_and_store(**context) -> dict:
    """Parse PDFs and upsert papers into Postgres."""
    from src.database import get_session
    from src.repositories.paper import PaperRepository
    from src.schemas.arxiv import ArxivPaperMetadata
    from src.services.pdf_parser import PdfParserService

    ti = context["ti"]
    fetch_result: dict = ti.xcom_pull(task_ids="fetch_metadata")
    papers_raw: list[dict] = fetch_result["papers"]
    logger.info("Received %d papers from XCom", len(papers_raw))

    papers = [ArxivPaperMetadata(**p) for p in papers_raw]
    summary: dict = {"fetched": len(papers), "skipped": 0, "stored": 0, "failed": 0}

    with get_session() as session:
        repo = PaperRepository(session)
        parser = PdfParserService()

        for paper in papers:
            if repo.exists(paper.arxiv_id):
                logger.info("[%s] Already in DB — skipping", paper.arxiv_id)
                summary["skipped"] += 1
                continue

            logger.info("[%s] Parsing PDF: %s", paper.arxiv_id, paper.pdf_url)
            parsed = parser.parse(paper.pdf_url, arxiv_id=paper.arxiv_id)

            if parsed:
                parse_status = "success"
                full_text = parsed.full_text
                sections = parsed.sections or None
                parse_error = None
                logger.info("[%s] Parsed %d chars", paper.arxiv_id, len(full_text))
            else:
                parse_status = "failed"
                full_text = None
                sections = None
                parse_error = "PDF parsing returned None"
                summary["failed"] += 1
                logger.warning("[%s] Parse failed — storing metadata only", paper.arxiv_id)

            repo.upsert(
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
            summary["stored"] += 1

    logger.info("Parse+store summary: %s", summary)
    return summary


def _index_opensearch(**context) -> dict:
    """Chunk, embed, and bulk-index newly stored papers into OpenSearch.

    Reads all papers with parse_status='success' from Postgres and upserts
    them into the OpenSearch chunks index. Safe to re-run (idempotent).
    Skips indexing run entirely if no new papers were stored this run.
    """
    from sqlalchemy import select

    from src.database import get_session
    from src.models.paper import Paper
    from src.services.opensearch import OpenSearchService

    ti = context["ti"]
    parse_summary: dict = ti.xcom_pull(task_ids="parse_and_store") or {}
    newly_stored = parse_summary.get("stored", 0)

    if newly_stored == 0:
        logger.info("No new papers stored this run — skipping OpenSearch indexing.")
        return {"indexed": 0, "errors": 0}

    logger.info("%d new papers stored — indexing into OpenSearch...", newly_stored)

    opensearch_service = OpenSearchService()
    opensearch_service.create_index()

    with get_session() as session:
        stmt = select(Paper).where(Paper.parse_status == "success")
        papers = list(session.scalars(stmt).all())
        logger.info("Total successfully parsed papers in DB: %d", len(papers))

        if not papers:
            return {"indexed": 0, "errors": 0}

        success_count, errors = opensearch_service.bulk_index(papers)
        error_count = len(errors) if isinstance(errors, list) else errors
        logger.info(
            "OpenSearch indexing complete: indexed=%d errors=%d", success_count, error_count
        )
        return {"indexed": success_count, "errors": error_count}


def _log_summary(**context) -> None:
    """Emit a human-readable ingestion + indexing summary."""
    ti = context["ti"]
    parse_summary: dict = ti.xcom_pull(task_ids="parse_and_store") or {}
    index_summary: dict = ti.xcom_pull(task_ids="index_opensearch") or {}
    print(
        f"\n{'─' * 55}\n"
        f"  arXiv Ingestion & Indexing Summary\n"
        f"  fetched        : {parse_summary.get('fetched', 0)}\n"
        f"  skipped (dup)  : {parse_summary.get('skipped', 0)}\n"
        f"  stored to PG   : {parse_summary.get('stored', 0)}\n"
        f"  parse failed   : {parse_summary.get('failed', 0)}\n"
        f"  indexed to OS  : {index_summary.get('indexed', 0)}\n"
        f"  index errors   : {index_summary.get('errors', 0)}\n"
        f"{'─' * 55}"
    )


# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="arxiv_paper_ingestion",
    description="Daily ingestion of arXiv papers into Postgres + OpenSearch",
    default_args=_DEFAULT_ARGS,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["arxiv", "ingestion", "rag"],
    doc_md=__doc__,
) as dag:

    ensure_db_tables = PythonOperator(
        task_id="ensure_db_tables",
        python_callable=_ensure_db_tables,
    )

    fetch_metadata = PythonOperator(
        task_id="fetch_metadata",
        python_callable=_fetch_metadata,
    )

    parse_and_store = PythonOperator(
        task_id="parse_and_store",
        python_callable=_parse_and_store,
    )

    index_opensearch = PythonOperator(
        task_id="index_opensearch",
        python_callable=_index_opensearch,
    )

    log_summary = PythonOperator(
        task_id="log_summary",
        python_callable=_log_summary,
    )

    # Task chain: DB setup -> Fetch -> Parse -> Index -> Summary
    ensure_db_tables >> fetch_metadata >> parse_and_store >> index_opensearch >> log_summary
