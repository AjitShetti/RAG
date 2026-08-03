"""FastAPI route handler for RAG Question-Answering endpoints (/ask and /ask/stream)."""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from ..schemas.rag import AskRequest, AskResponse
from ..services.cache import CacheService
from ..services.embeddings import EmbeddingService, get_embedding_service
from ..services.langfuse import LangfuseService
from ..services.llm import LLMClient
from ..services.opensearch import OpenSearchService
from ..services.rag import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ask", tags=["RAG Question Answering"])


def get_opensearch_service() -> OpenSearchService:
    """Dependency provider for OpenSearchService."""
    return OpenSearchService()


def get_llm_client() -> LLMClient:
    """Dependency provider for LLMClient."""
    return LLMClient()


def get_cache_service() -> CacheService:
    """Dependency provider for CacheService."""
    return CacheService()


def get_langfuse_service() -> LangfuseService:
    """Dependency provider for LangfuseService."""
    return LangfuseService()


def get_rag_pipeline(
    opensearch: Annotated[OpenSearchService, Depends(get_opensearch_service)],
    embedding: Annotated[EmbeddingService, Depends(get_embedding_service)],
    llm: Annotated[LLMClient, Depends(get_llm_client)],
    langfuse: Annotated[LangfuseService, Depends(get_langfuse_service)],
) -> RAGPipeline:
    """Dependency provider for RAGPipeline."""
    return RAGPipeline(
        opensearch_service=opensearch,
        embedding_service=embedding,
        llm_client=llm,
        langfuse_service=langfuse,
    )


@router.post(
    "",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask Question (Standard RAG)",
    description=(
        "Retrieve relevant paper chunks via hybrid search, build a grounded prompt, "
        "and generate a cited answer from the LLM."
    ),
)
def ask_question(
    request: AskRequest,
    rag_pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    cache_service: Annotated[CacheService, Depends(get_cache_service)],
    langfuse_service: Annotated[LangfuseService, Depends(get_langfuse_service)],
) -> AskResponse:
    """Execute standard blocking RAG pipeline with Redis caching and Langfuse tracing."""
    try:
        cache_key = cache_service.get_cache_key(request)
        cached_res = cache_service.get(cache_key)
        if cached_res is not None:
            trace = langfuse_service.start_trace(
                name="rag_ask", metadata={"query": request.query, "mode": request.mode, "cached": True}
            )
            langfuse_service.log_cache_hit(trace, cache_key)
            langfuse_service.flush()
            cached_res.cached = True
            return cached_res

        response = rag_pipeline.answer(request)
        cache_service.set(cache_key, response)
        return response
    except Exception as exc:
        logger.error("RAG pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG service error: {exc}",
        ) from exc


@router.post(
    "/stream",
    summary="Ask Question (Streaming SSE RAG)",
    description=(
        "Stream LLM answer tokens in real-time as Server-Sent Events (SSE). "
        "The final event contains metadata and source citations."
    ),
)
async def ask_question_stream(
    request: AskRequest,
    rag_pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    cache_service: Annotated[CacheService, Depends(get_cache_service)],
    langfuse_service: Annotated[LangfuseService, Depends(get_langfuse_service)],
):
    """Execute streaming RAG pipeline yielding Server-Sent Events with response caching.

    Trade-off note: On a cache hit, we return the cached AskResponse directly (not streaming)
    with cached=True. Returning cached JSON directly eliminates streaming overhead while
    delivering immediate response for previously computed queries.
    """
    cache_key = cache_service.get_cache_key(request)
    cached_res = cache_service.get(cache_key)
    if cached_res is not None:
        trace = langfuse_service.start_trace(
            name="rag_ask", metadata={"query": request.query, "mode": request.mode, "cached": True}
        )
        langfuse_service.log_cache_hit(trace, cache_key)
        langfuse_service.flush()
        cached_res.cached = True
        return cached_res

    async def event_generator():
        try:
            full_tokens: list[str] = []
            final_meta: dict[str, Any] = {}
            async for item in rag_pipeline.answer_stream(request):
                event_type = item.get("event", "message")
                data_val = item.get("data")
                if event_type == "token" and isinstance(data_val, str):
                    full_tokens.append(data_val)
                elif event_type == "metadata" and isinstance(data_val, dict):
                    final_meta = data_val

                if isinstance(data_val, dict):
                    yield {"event": event_type, "data": json.dumps(data_val)}
                else:
                    yield {"event": event_type, "data": str(data_val)}

            if full_tokens or final_meta:
                answer_text = "".join(full_tokens)
                sources = final_meta.get("sources", [])
                assembled_resp = AskResponse(
                    answer=answer_text,
                    sources=sources,
                    retrieved_chunk_count=final_meta.get("retrieved_chunk_count", 0),
                    used_chunk_count=final_meta.get("used_chunk_count", 0),
                    took_ms=final_meta.get("took_ms", 0.0),
                    prompt_tokens=final_meta.get("prompt_tokens", 0),
                    completion_tokens=final_meta.get("completion_tokens", 0),
                    cached=True,
                )
                cache_service.set(cache_key, assembled_resp)

        except Exception as exc:
            logger.error("RAG streaming pipeline failed: %s", exc, exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())
