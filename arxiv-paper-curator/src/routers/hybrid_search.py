"""FastAPI router for Hybrid Search endpoints (BM25 + kNN vector + RRF)."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..schemas.search import HybridSearchRequest, HybridSearchResponse
from ..services.embeddings import EmbeddingService, get_embedding_service
from ..services.opensearch import OpenSearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hybrid-search", tags=["Hybrid Search"])


def get_opensearch_service() -> OpenSearchService:
    """Dependency provider for OpenSearchService."""
    return OpenSearchService()


@router.post(
    "",
    response_model=HybridSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid Search (BM25 + kNN + RRF)",
    description="Search arXiv paper chunks using BM25 keyword search, kNN vector similarity, or hybrid RRF fusion. Supports mode parameter ('keyword', 'semantic', 'hybrid') and opt-in paper-level grouping via 'group_by_paper'.",
)
def hybrid_search_chunks(
    request: HybridSearchRequest,
    opensearch_service: Annotated[OpenSearchService, Depends(get_opensearch_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> HybridSearchResponse:
    """Execute hybrid BM25 + vector kNN search fused with Reciprocal Rank Fusion."""
    try:
        return opensearch_service.hybrid_search(
            request=request,
            embedding_service=embedding_service,
        )
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Hybrid search service error: {exc}",
        ) from exc
