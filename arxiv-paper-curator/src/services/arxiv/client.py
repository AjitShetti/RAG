"""arXiv API client with rate limiting and tenacity retry/backoff.

ArxivClient is a pure service class — no Airflow, no SQLAlchemy, no FastAPI.
It can be instantiated and called from any context (DAG task, CLI, unit test).

Rate limiting strategy:
  - The underlying `arxiv.Client` already enforces a delay between requests.
  - We add an explicit threading.Semaphore + time-based token bucket on top so
    that concurrent callers (e.g. multiple Airflow tasks in the same process)
    also respect the limit.

Retry strategy (tenacity):
  - Retries on network errors and arXiv 5xx responses.
  - Exponential backoff: 2s → 4s → 8s → 16s (max 4 attempts).
  - After all retries exhausted, raises the original exception so callers can
    decide whether to fail the DAG task or skip the paper.
"""

import logging
import threading
import time

import arxiv
import tenacity

from ...config import settings
from ...schemas.arxiv import ArxivPaperMetadata

logger = logging.getLogger(__name__)


# ── Token-bucket rate limiter ──────────────────────────────────────────────────

class _RateLimiter:
    """Thread-safe token-bucket rate limiter.

    Allows `max_calls` calls per `period` seconds. Excess callers sleep until
    a token is available.
    """

    def __init__(self, max_calls: int, period: float) -> None:
        self._max_calls = max_calls
        self._period = period
        self._lock = threading.Lock()
        self._calls: list[float] = []

    def acquire(self) -> None:
        """Block until a request token is available."""
        with self._lock:
            now = time.monotonic()
            # Drop timestamps outside the current window
            self._calls = [t for t in self._calls if now - t < self._period]
            if len(self._calls) >= self._max_calls:
                oldest = self._calls[0]
                sleep_for = self._period - (now - oldest)
                if sleep_for > 0:
                    logger.debug("Rate limit reached — sleeping %.2fs", sleep_for)
                    time.sleep(sleep_for)
                # Recheck after sleep
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self._period]
            self._calls.append(time.monotonic())


# ── Client ────────────────────────────────────────────────────────────────────

class ArxivClient:
    """Fetches paper metadata from the arXiv API.

    Usage::

        client = ArxivClient()
        papers = client.search("cs.AI", max_results=10)
        for p in papers:
            print(p.arxiv_id, p.title)
    """

    def __init__(
        self,
        max_calls: int | None = None,
        period: float | None = None,
    ) -> None:
        max_calls = max_calls or settings.arxiv_rate_limit_calls
        period = period or settings.arxiv_rate_limit_period

        self._rate_limiter = _RateLimiter(max_calls=max_calls, period=period)
        # arxiv.Client handles pagination and per-request delays internally
        self._client = arxiv.Client(
            page_size=10,
            delay_seconds=max(period / max_calls, 0.33),
            num_retries=3,
        )

    # ── Public API ────────────────────────────────────────────

    def search(
        self,
        query: str,
        max_results: int | None = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
    ) -> list[ArxivPaperMetadata]:
        """Search arXiv and return structured metadata for up to `max_results` papers.

        Args:
            query: arXiv query string, e.g. ``"cat:cs.AI"`` or ``"LLM agent"``.
            max_results: Override ``settings.arxiv_max_results`` for this call.
            sort_by: Sorting criterion (default: newest first).

        Returns:
            List of :class:`ArxivPaperMetadata` sorted by *sort_by*.

        Raises:
            Exception: After all tenacity retries are exhausted.
        """
        limit = max_results or settings.arxiv_max_results
        logger.info("Searching arXiv: query=%r max_results=%d", query, limit)

        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=sort_by,
        )
        results = self._fetch_with_retry(search)
        logger.info("arXiv returned %d results for query=%r", len(results), query)
        return results

    # ── Internal helpers ──────────────────────────────────────

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(4),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=16),
        retry=tenacity.retry_if_exception_type((Exception,)),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _fetch_with_retry(self, search: arxiv.Search) -> list[ArxivPaperMetadata]:
        """Execute the arXiv search with exponential-backoff retry."""
        self._rate_limiter.acquire()
        papers: list[ArxivPaperMetadata] = []

        for result in self._client.results(search):
            arxiv_id = result.entry_id.split("/abs/")[-1]  # strip URL prefix
            papers.append(
                ArxivPaperMetadata(
                    arxiv_id=arxiv_id,
                    title=result.title.strip(),
                    authors=[a.name for a in result.authors],
                    abstract=result.summary.strip(),
                    pdf_url=result.pdf_url or "",
                    published_date=result.published.date(),
                    category=result.primary_category or "",
                )
            )
        return papers
