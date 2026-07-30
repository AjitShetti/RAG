"""FastAPI router for BM25 keyword search endpoints.

Thin HTTP layer — delegates business logic and OpenSearch queries to OpenSearchService.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..schemas.search import SearchRequest, SearchResponse
from ..services.opensearch import OpenSearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


def get_opensearch_service() -> OpenSearchService:
    """Dependency provider for OpenSearchService."""
    return OpenSearchService()


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="BM25 Keyword Search",
    description="Search arXiv papers using BM25 relevance scoring across title, abstract, and section body fields with optional filters.",
)
def search_papers(
    request: SearchRequest,
    opensearch_service: Annotated[OpenSearchService, Depends(get_opensearch_service)],
) -> SearchResponse:
    """Execute BM25 keyword search over indexed arXiv papers."""
    try:
        return opensearch_service.search(request)
    except Exception as exc:
        logger.error("OpenSearch search query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service is currently unavailable or unreachable",
        ) from exc
