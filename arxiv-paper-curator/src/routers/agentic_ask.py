"""FastAPI route handler for Agentic RAG Question-Answering endpoints (/agentic-ask)."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..schemas.rag import AgenticAskResponse, AskRequest
from ..services.agents.agentic_rag import run_agentic_rag
from ..services.cache import CacheService
from ..services.langfuse import LangfuseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agentic-ask", tags=["Agentic RAG"])


def get_cache_service() -> CacheService:
    """Dependency provider for CacheService."""
    return CacheService()


def get_langfuse_service() -> LangfuseService:
    """Dependency provider for LangfuseService."""
    return LangfuseService()


@router.post(
    "",
    response_model=AgenticAskResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask Question (Agentic RAG)",
    description=(
        "Execute agentic workflow with guardrails, iterative retrieval, grading, "
        "query rewriting, and LLM answer generation."
    ),
)
def agentic_ask_question(
    request: AskRequest,
    cache_service: Annotated[CacheService, Depends(get_cache_service)],
    langfuse_service: Annotated[LangfuseService, Depends(get_langfuse_service)],
) -> AgenticAskResponse:
    """Execute blocking Agentic RAG pipeline with Redis caching and Langfuse tracing."""
    try:
        cache_key = cache_service.get_agentic_cache_key(request)
        cached_res = cache_service.get_agentic(cache_key)
        if cached_res is not None:
            trace = langfuse_service.start_trace(
                name="agentic_rag_ask",
                metadata={"query": request.query, "mode": request.mode, "cached": True},
            )
            langfuse_service.log_cache_hit(trace, cache_key)
            langfuse_service.flush()
            cached_res.cached = True
            return cached_res

        response = run_agentic_rag(request=request, langfuse_service=langfuse_service)
        cache_service.set_agentic(cache_key, response)
        return response
    except Exception as exc:
        logger.error("Agentic RAG pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agentic RAG service error: {exc}",
        ) from exc
