"""Airflow DAG: arxiv_paper_ingestion

Scheduled daily. Fetches new arXiv papers matching a configurable query,
parses their PDFs with pdfminer, and stores structured rows in Postgres.
Papers already in the database are automatically skipped (idempotent).

Task graph:
  ensure_db_tables >> fetch_metadata >> parse_and_store >> log_summary

Environment variables (read from Airflow Variables or .env):
  ARXIV_QUERY       — arXiv query string (default: cs.AI)
  ARXIV_MAX_RESULTS — max papers per run (default: 20)
  DATABASE_URL      — sync postgres URL (psycopg2)

To trigger manually from the Airflow UI: unpause the DAG → Trigger DAG.
To trigger from CLI: airflow dags trigger arxiv_paper_ingestion
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta

# ── Ensure the app source is on sys.path when running inside the container ──
# The Airflow container mounts arxiv-paper-curator/ to /opt/airflow/app.
# If the path is not already on sys.path (e.g. local runs), add it.
_APP_ROOT = os.environ.get("APP_ROOT", "/opt/airflow/app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ── Default DAG arguments ──────────────────────────────────────────────────────
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

def _ensure_db_tables(**context) -> None:  # noqa: ARG001
    """Create the `papers` table if it does not exist (idempotent)."""
    from src.database import init_db
    init_db()
    logger.info("DB tables verified.")


def _fetch_metadata(**context) -> dict:
    """Fetch paper metadata from arXiv and push to XCom.

    Returns a dict with keys `query`, `max_results`, and `paper_ids`
    (list of arxiv_id strings). We push only IDs through XCom to keep
    payload small — full metadata is re-fetched during parse_and_store.
    """
    from src.services.arxiv import ArxivClient

    query = os.environ.get("ARXIV_QUERY", "cs.AI")
    max_results = int(os.environ.get("ARXIV_MAX_RESULTS", "20"))

    logger.info("Fetching arXiv metadata: query=%r max_results=%d", query, max_results)
    client = ArxivClient()
    papers = client.search(query=query, max_results=max_results)
    paper_ids = [p.arxiv_id for p in papers]
    logger.info("Fetched %d paper IDs", len(paper_ids))

    # Push full metadata as XCom value (small enough for a daily batch)
    return {
        "query": query,
        "max_results": max_results,
        "papers": [p.model_dump(mode="json") for p in papers],
    }


def _parse_and_store(**context) -> dict:
    """Parse PDFs and store papers into Postgres.

    Pulls the paper list from XCom (produced by _fetch_metadata).
    Returns a summary dict pushed to XCom for the log_summary task.
    """
    import json

    from src.database import get_session
    from src.repositories.paper import PaperRepository
    from src.schemas.arxiv import ArxivPaperMetadata
    from src.services.metadata_fetcher import MetadataFetcher
    from src.services.pdf_parser import PdfParserService

    ti = context["ti"]
    fetch_result: dict = ti.xcom_pull(task_ids="fetch_metadata")
    papers_raw: list[dict] = fetch_result["papers"]

    logger.info("Received %d papers from XCom", len(papers_raw))

    # Reconstruct Pydantic models from XCom-serialised dicts
    papers = [ArxivPaperMetadata(**p) for p in papers_raw]

    summary_dict: dict = {"fetched": len(papers), "skipped": 0, "stored": 0, "failed": 0}

    with get_session() as session:
        repo = PaperRepository(session)
        parser = PdfParserService()

        for paper in papers:
            if repo.exists(paper.arxiv_id):
                logger.info("[%s] Already in DB — skipping", paper.arxiv_id)
                summary_dict["skipped"] += 1
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
                summary_dict["failed"] += 1
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
            summary_dict["stored"] += 1

    logger.info("Ingestion summary: %s", summary_dict)
    return summary_dict


def _log_summary(**context) -> None:
    """Pull the ingestion summary from XCom and emit a human-readable log."""
    ti = context["ti"]
    summary: dict = ti.xcom_pull(task_ids="parse_and_store")
    if summary:
        print(
            f"\n{'─' * 50}\n"
            f"  arXiv Ingestion Summary\n"
            f"  fetched      : {summary.get('fetched', 0)}\n"
            f"  skipped (dup): {summary.get('skipped', 0)}\n"
            f"  stored       : {summary.get('stored', 0)}\n"
            f"  parse failed : {summary.get('failed', 0)}\n"
            f"{'─' * 50}"
        )


# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="arxiv_paper_ingestion",
    description="Daily ingestion of arXiv papers into Postgres",
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

    log_summary = PythonOperator(
        task_id="log_summary",
        python_callable=_log_summary,
    )

    # Task dependency chain
    ensure_db_tables >> fetch_metadata >> parse_and_store >> log_summary
