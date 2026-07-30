"""OpenSearch service providing BM25 keyword, kNN vector, and RRF Hybrid Search over paper documents & chunks.

Handles cluster connection, index lifecycle, bulk indexing, and hybrid query execution.
Pure service class — independent of HTTP routers and framework concerns.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from opensearchpy import OpenSearch, helpers

from ...config import settings
from ...models.paper import Paper
from ...schemas.search import (
    HybridSearchHit,
    HybridSearchRequest,
    HybridSearchResponse,
    PaperGroupedHit,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from ..embeddings.base import EmbeddingService
from .chunk_index_mapping import CHUNK_INDEX_MAPPING, CHUNK_INDEX_NAME
from .index_mapping import INDEX_NAME, PAPER_INDEX_MAPPING
from .rrf import rrf_fusion

logger = logging.getLogger(__name__)


class OpenSearchService:
    """Service layer for OpenSearch indexing, BM25, kNN vector, and RRF hybrid search operations."""

    def __init__(
        self,
        client: OpenSearch | None = None,
        url: str | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            opensearch_url = url or settings.opensearch_url
            self._client = OpenSearch(
                hosts=[opensearch_url],
                verify_certs=False,
                ssl_show_warn=False,
                timeout=10,
            )

    # ── Health check ──────────────────────────────────────────

    def ping(self) -> bool:
        """Check if OpenSearch cluster is reachable and healthy."""
        try:
            return bool(self._client.ping())
        except Exception as exc:
            logger.warning("OpenSearch ping failed: %s", exc)
            return False

    # ── Index creation ────────────────────────────────────────

    def create_index(self, index_name: str = INDEX_NAME) -> bool:
        """Create OpenSearch paper index if it does not already exist (idempotent)."""
        try:
            if self._client.indices.exists(index=index_name):
                logger.info("OpenSearch index '%s' already exists — skipping creation", index_name)
                return False

            logger.info("Creating OpenSearch index '%s' …", index_name)
            self._client.indices.create(index=index_name, body=PAPER_INDEX_MAPPING)
            logger.info("Index '%s' created successfully.", index_name)
            return True
        except Exception as exc:
            logger.error("Failed to create index '%s': %s", index_name, exc)
            raise

    def create_chunk_index(self, index_name: str = CHUNK_INDEX_NAME) -> bool:
        """Create OpenSearch chunk vector index if it does not already exist (idempotent)."""
        try:
            if self._client.indices.exists(index=index_name):
                logger.info("OpenSearch index '%s' already exists — skipping creation", index_name)
                return False

            logger.info("Creating OpenSearch vector chunk index '%s' …", index_name)
            self._client.indices.create(index=index_name, body=CHUNK_INDEX_MAPPING)
            logger.info("Index '%s' created successfully.", index_name)
            return True
        except Exception as exc:
            logger.error("Failed to create chunk index '%s': %s", index_name, exc)
            raise

    # ── Document transformation helper ────────────────────────

    @staticmethod
    def paper_to_document(paper: Paper) -> dict[str, Any]:
        """Convert a Postgres Paper ORM model instance into an OpenSearch document dict."""
        section_headings: list[str] = []
        section_bodies: list[str] = []

        if paper.sections:
            for s in paper.sections:
                if isinstance(s, dict):
                    if heading := s.get("heading"):
                        section_headings.append(heading)
                    if text := s.get("text"):
                        section_bodies.append(text)

        return {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors or [],
            "abstract": paper.abstract,
            "section_headings": " ".join(section_headings),
            "section_bodies": " ".join(section_bodies),
            "full_text": paper.full_text or "",
            "pdf_url": paper.pdf_url,
            "published_date": paper.published_date.isoformat() if paper.published_date else None,
            "category": paper.category,
            "parse_status": paper.parse_status,
        }

    # ── Indexing operations ───────────────────────────────────

    def index_paper(self, paper: Paper, index_name: str = INDEX_NAME) -> bool:
        """Index or update a single paper (upsert by arxiv_id)."""
        doc = self.paper_to_document(paper)
        res = self._client.index(
            index=index_name,
            id=paper.arxiv_id,
            body=doc,
            refresh=True,
        )
        logger.debug("Indexed paper %s: %s", paper.arxiv_id, res.get("result"))
        return res.get("result") in ("created", "updated")

    def bulk_index(
        self,
        papers: list[Paper],
        index_name: str = INDEX_NAME,
        chunk_size: int = 50,
    ) -> tuple[int, int | list[Any]]:
        """Bulk index multiple papers from Postgres into OpenSearch (idempotent upsert)."""
        if not papers:
            return 0, 0

        actions = [
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": paper.arxiv_id,
                "_source": self.paper_to_document(paper),
            }
            for paper in papers
        ]

        logger.info("Bulk indexing %d papers into '%s' …", len(papers), index_name)
        success_count, errors = helpers.bulk(
            self._client,
            actions,
            chunk_size=chunk_size,
            stats_only=False,
            raise_on_error=False,
        )
        error_count = len(errors) if isinstance(errors, list) else errors
        logger.info("Bulk index completed: %d succeeded, %d failed", success_count, error_count)
        return success_count, errors

    def delete_paper_chunks(self, paper_id: str, index_name: str = CHUNK_INDEX_NAME) -> int:
        """Delete all chunks for a paper_id from the chunk index (for idempotency)."""
        try:
            if not self._client.indices.exists(index=index_name):
                return 0
            query = {"query": {"term": {"paper_id": paper_id}}}
            res = self._client.delete_by_query(index=index_name, body=query, refresh=True)
            deleted = res.get("deleted", 0)
            logger.debug("Deleted %d existing chunks for paper %s", deleted, paper_id)
            return deleted
        except Exception as exc:
            logger.warning("Failed to delete existing chunks for paper %s: %s", paper_id, exc)
            return 0

    def index_chunks(
        self,
        chunk_documents: list[dict[str, Any]],
        index_name: str = CHUNK_INDEX_NAME,
        chunk_size: int = 50,
    ) -> tuple[int, int | list[Any]]:
        """Bulk index chunk documents into OpenSearch (idempotent upsert by chunk_id)."""
        if not chunk_documents:
            return 0, 0

        actions = [
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": doc["chunk_id"],
                "_source": doc,
            }
            for doc in chunk_documents
        ]

        logger.info("Bulk indexing %d chunks into '%s' …", len(chunk_documents), index_name)
        success_count, errors = helpers.bulk(
            self._client,
            actions,
            chunk_size=chunk_size,
            stats_only=False,
            raise_on_error=False,
        )
        error_count = len(errors) if isinstance(errors, list) else errors
        logger.info("Chunk bulk index completed: %d succeeded, %d failed", success_count, error_count)
        return success_count, errors

    # ── Pure BM25 Paper Search ────────────────────────────────

    def search(
        self,
        request: SearchRequest,
        index_name: str = INDEX_NAME,
    ) -> SearchResponse:
        """Execute BM25 keyword search over paper-level index."""
        start_time = time.perf_counter()

        multi_match_query = {
            "multi_match": {
                "query": request.query,
                "fields": [
                    "title^4",
                    "abstract^2",
                    "section_headings^1.5",
                    "section_bodies^1",
                ],
                "type": "best_fields",
            }
        }

        filter_clauses: list[dict[str, Any]] = []
        if request.category:
            filter_clauses.append({"term": {"category": request.category}})

        date_range: dict[str, Any] = {}
        if request.date_from:
            date_range["gte"] = request.date_from.isoformat()
        if request.date_to:
            date_range["lte"] = request.date_to.isoformat()
        if date_range:
            filter_clauses.append({"range": {"published_date": date_range}})

        query_body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [multi_match_query],
                    "filter": filter_clauses,
                }
            },
            "from": (request.page - 1) * request.page_size,
            "size": request.page_size,
            "highlight": {
                "fields": {
                    "title": {},
                    "abstract": {},
                    "section_bodies": {},
                },
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
                "fragment_size": 150,
                "number_of_fragments": 2,
            },
        }

        raw_res = self._client.search(index=index_name, body=query_body)
        took_ms = round((time.perf_counter() - start_time) * 1000, 2)
        hits_data = raw_res["hits"]
        total_hits = hits_data["total"]["value"]

        results: list[SearchHit] = []
        for hit in hits_data["hits"]:
            src = hit["_source"]
            highlights = hit.get("highlight", {})
            results.append(
                SearchHit(
                    arxiv_id=src.get("arxiv_id", hit["_id"]),
                    title=src.get("title", ""),
                    authors=src.get("authors", []),
                    abstract=src.get("abstract", ""),
                    category=src.get("category", ""),
                    published_date=src.get("published_date"),
                    pdf_url=src.get("pdf_url", ""),
                    score=hit.get("_score", 0.0),
                    highlights=highlights,
                )
            )

        return SearchResponse(
            total=total_hits,
            page=request.page,
            page_size=request.page_size,
            took_ms=took_ms,
            results=results,
        )

    # ── kNN Vector Chunk Search ───────────────────────────────

    def knn_search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        filters: list[dict[str, Any]] | None = None,
        index_name: str = CHUNK_INDEX_NAME,
    ) -> list[dict[str, Any]]:
        """Execute kNN vector search over chunk embeddings."""
        knn_clause = {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k,
                }
            }
        }

        body: dict[str, Any] = {
            "size": top_k,
            "query": (
                {"bool": {"must": [knn_clause], "filter": filters}}
                if filters
                else knn_clause
            ),
        }

        logger.debug("Executing kNN vector search in '%s' …", index_name)
        res = self._client.search(index=index_name, body=body)
        return res["hits"]["hits"]

    # ── BM25 Chunk Search ─────────────────────────────────────

    def bm25_chunk_search(
        self,
        query_text: str,
        top_k: int = 20,
        filters: list[dict[str, Any]] | None = None,
        index_name: str = CHUNK_INDEX_NAME,
    ) -> list[dict[str, Any]]:
        """Execute BM25 keyword search over chunk text."""
        multi_match_query = {
            "multi_match": {
                "query": query_text,
                "fields": ["title^3", "section_name^2", "text^1"],
                "type": "best_fields",
            }
        }

        body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [multi_match_query],
                    "filter": filters or [],
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "text": {},
                },
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
                "fragment_size": 150,
                "number_of_fragments": 2,
            },
        }

        logger.debug("Executing BM25 chunk search in '%s' …", index_name)
        res = self._client.search(index=index_name, body=body)
        return res["hits"]["hits"]

    # ── Hybrid Search with RRF Fusion ─────────────────────────

    def hybrid_search(
        self,
        request: HybridSearchRequest,
        embedding_service: EmbeddingService | None = None,
        index_name: str = CHUNK_INDEX_NAME,
    ) -> HybridSearchResponse:
        """Execute hybrid search combining BM25 keyword search and kNN vector search fused with RRF."""
        start_time = time.perf_counter()

        # Build filter clauses
        filter_clauses: list[dict[str, Any]] = []
        if request.category:
            filter_clauses.append({"term": {"category": request.category}})

        date_range: dict[str, Any] = {}
        if request.date_from:
            date_range["gte"] = request.date_from.isoformat()
        if request.date_to:
            date_range["lte"] = request.date_to.isoformat()
        if date_range:
            filter_clauses.append({"range": {"published_date": date_range}})

        top_k = request.page_size * 3  # fetch larger candidate pool for RRF fusion

        keyword_hits: list[dict[str, Any]] = []
        semantic_hits: list[dict[str, Any]] = []

        # 1. Execute keyword search if requested
        if request.mode in ("keyword", "hybrid"):
            try:
                keyword_hits = self.bm25_chunk_search(
                    query_text=request.query,
                    top_k=top_k,
                    filters=filter_clauses,
                    index_name=index_name,
                )
            except Exception as exc:
                logger.warning("BM25 chunk search failed: %s", exc)

        # 2. Execute semantic vector search if requested
        if request.mode in ("semantic", "hybrid") and embedding_service is not None:
            try:
                embedded_query = embedding_service.embed([request.query])
                if embedded_query and embedded_query[0] is not None:
                    semantic_hits = self.knn_search(
                        query_vector=embedded_query[0],
                        top_k=top_k,
                        filters=filter_clauses,
                        index_name=index_name,
                    )
            except Exception as exc:
                logger.warning("Semantic vector search failed: %s", exc)

        # 3. Combine rankings via RRF or pass-through single mode
        rrf_results = rrf_fusion(
            keyword_hits=keyword_hits,
            semantic_hits=semantic_hits,
            k=60,
            id_field="chunk_id",
        )

        total_hits = len(rrf_results)

        # Pagination
        offset = (request.page - 1) * request.page_size
        page_results = rrf_results[offset : offset + request.page_size]

        # 4. Map to HybridSearchHit DTOs
        raw_hits: list[HybridSearchHit] = []
        for r in page_results:
            src = r.source_doc
            raw_hits.append(
                HybridSearchHit(
                    chunk_id=r.doc_id,
                    paper_id=src.get("paper_id", ""),
                    section_name=src.get("section_name", ""),
                    chunk_index=src.get("chunk_index", 0),
                    text=src.get("text", ""),
                    title=src.get("title", ""),
                    authors=src.get("authors", []),
                    category=src.get("category", ""),
                    published_date=src.get("published_date"),
                    pdf_url=src.get("pdf_url", ""),
                    score=r.rrf_score,
                    keyword_rank=r.keyword_rank,
                    semantic_rank=r.semantic_rank,
                    contributed_by=r.contributed_by,
                    highlights=r.highlights,
                )
            )

        # 5. Optional group_by_paper logic (USER preference)
        grouped: list[PaperGroupedHit] | None = None
        if request.group_by_paper and raw_hits:
            grouped_dict: dict[str, PaperGroupedHit] = {}
            for hit in raw_hits:
                pid = hit.paper_id
                if pid not in grouped_dict:
                    grouped_dict[pid] = PaperGroupedHit(
                        paper_id=pid,
                        title=hit.title,
                        authors=hit.authors,
                        category=hit.category,
                        published_date=hit.published_date,
                        pdf_url=hit.pdf_url,
                        best_score=hit.score,
                        chunks=[hit],
                    )
                else:
                    grouped_dict[pid].chunks.append(hit)
                    if hit.score > grouped_dict[pid].best_score:
                        grouped_dict[pid].best_score = hit.score

            grouped = list(grouped_dict.values())

        took_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "[HybridSearch] Mode=%s query=%r hits=%d page=%d size=%d took_ms=%.2f",
            request.mode,
            request.query,
            total_hits,
            request.page,
            request.page_size,
            took_ms,
        )

        return HybridSearchResponse(
            total=total_hits,
            page=request.page,
            page_size=request.page_size,
            mode=request.mode,
            took_ms=took_ms,
            results=raw_hits,
            grouped_results=grouped,
        )
