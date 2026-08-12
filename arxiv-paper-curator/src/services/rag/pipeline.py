"""RAG Pipeline Orchestrator.

Coordinates hybrid retrieval, context building with token budget enforcement,
LLM generation/streaming, source attribution, zero-results handling, and Langfuse tracing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from ...config import settings
from ...schemas.rag import AskRequest, AskResponse, SourceChunk
from ...schemas.search import HybridSearchHit, HybridSearchRequest
from ..embeddings.base import EmbeddingService
from ..langfuse.service import LangfuseService
from ..llm.client import LLMClient
from ..opensearch.service import OpenSearchService
from .prompt_builder import (
    build_context_block,
    build_messages,
    load_system_prompt,
)

from ..embeddings import get_embedding_service

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = (
    "I don't have enough information in the retrieved papers to answer this question reliably."
)


class RAGPipeline:
    """End-to-end RAG orchestrator linking OpenSearch hybrid retrieval to LLM generation."""

    def __init__(
        self,
        opensearch_service: OpenSearchService | None = None,
        embedding_service: EmbeddingService | None = None,
        llm_client: LLMClient | None = None,
        system_prompt: str | None = None,
        langfuse_service: LangfuseService | None = None,
    ) -> None:
        self.opensearch = opensearch_service or OpenSearchService()
        self.embedding = embedding_service or get_embedding_service()
        self.llm = llm_client or LLMClient()
        self.system_prompt = system_prompt or load_system_prompt()
        self.relevance_threshold = settings.rag_relevance_threshold
        self.max_context_tokens = settings.rag_max_context_tokens
        self.langfuse = langfuse_service

    def retrieve(self, request: AskRequest) -> list[HybridSearchHit]:
        """Fetch candidate chunks from OpenSearch via hybrid search."""
        search_req = HybridSearchRequest(
            query=request.query,
            mode=request.mode,
            category=request.category,
            date_from=request.date_from,
            date_to=request.date_to,
            group_by_paper=False,
            page=1,
            page_size=request.top_k,
        )

        res = self.opensearch.hybrid_search(
            request=search_req,
            embedding_service=self.embedding,
        )

        # Filter out chunks below relevance threshold if configured
        if self.relevance_threshold > 0:
            filtered = [c for c in res.results if c.score >= self.relevance_threshold]
            logger.info(
                "Retrieved %d chunks (%d met relevance threshold >= %.4f)",
                len(res.results),
                len(filtered),
                self.relevance_threshold,
            )
            return filtered

        return res.results

    def _build_source_chunks(self, chunks: list[HybridSearchHit]) -> list[SourceChunk]:
        """Map retrieved chunks to user-facing SourceChunk attribution objects."""
        sources: list[SourceChunk] = []
        for c in chunks:
            snippet = c.text[:300] + "..." if len(c.text) > 300 else c.text
            sources.append(
                SourceChunk(
                    paper_id=c.paper_id,
                    title=c.title,
                    section_name=c.section_name,
                    snippet=snippet,
                    relevance_score=round(c.score, 4),
                    pdf_url=c.pdf_url,
                )
            )
        return sources

    def answer(self, request: AskRequest) -> AskResponse:
        """Execute full blocking RAG pipeline (retrieve → build prompt → generate answer)."""
        start_time = time.perf_counter()

        trace = (
            self.langfuse.start_trace(
                name="rag_ask",
                metadata={"query": request.query, "mode": request.mode},
            )
            if self.langfuse
            else None
        )

        # 1. Retrieve candidate chunks with hybrid_search span
        search_span = (
            self.langfuse.start_span(
                trace_or_span=trace,
                name="hybrid_search",
                input_data={
                    "query": request.query,
                    "mode": request.mode,
                    "category": request.category,
                    "top_k": request.top_k,
                },
            )
            if self.langfuse
            else None
        )

        retrieved_chunks = self.retrieve(request)

        if self.langfuse and search_span:
            top_scores = [round(c.score, 4) for c in retrieved_chunks[:5]]
            self.langfuse.end_span(
                search_span,
                output_data={"hit_count": len(retrieved_chunks), "top_scores": top_scores},
            )

        # 2. Handle zero results / no relevant context case
        if not retrieved_chunks:
            took_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info("No relevant chunks found for query %r — returning fallback answer", request.query)
            if self.langfuse:
                self.langfuse.flush()
            return AskResponse(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                retrieved_chunk_count=0,
                used_chunk_count=0,
                took_ms=took_ms,
                prompt_tokens=0,
                completion_tokens=0,
            )

        # 3. Select chunks that fit within context token budget with build_prompt span
        prompt_span = (
            self.langfuse.start_span(
                trace_or_span=trace,
                name="build_prompt",
                input_data={
                    "max_context_tokens": self.max_context_tokens,
                    "retrieved_chunk_count": len(retrieved_chunks),
                },
            )
            if self.langfuse
            else None
        )

        context_block, used_chunks = build_context_block(
            chunks=retrieved_chunks,
            max_tokens=self.max_context_tokens,
        )

        if self.langfuse and prompt_span:
            self.langfuse.end_span(
                prompt_span,
                output_data={"used_chunk_count": len(used_chunks)},
            )

        if not used_chunks:
            took_ms = round((time.perf_counter() - start_time) * 1000, 2)
            if self.langfuse:
                self.langfuse.flush()
            return AskResponse(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                retrieved_chunk_count=len(retrieved_chunks),
                used_chunk_count=0,
                took_ms=took_ms,
                prompt_tokens=0,
                completion_tokens=0,
            )

        # 4. Construct messages and call LLM with llm_generation span
        messages = build_messages(
            system_prompt=self.system_prompt,
            context_block=context_block,
            user_question=request.query,
        )

        gen_start = time.perf_counter()
        llm_res = self.llm.generate(messages=messages)
        gen_latency_ms = round((time.perf_counter() - gen_start) * 1000, 2)
        took_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if self.langfuse and trace:
            self.langfuse.log_generation(
                trace_or_span=trace,
                name="llm_generation",
                model=settings.llm_model,
                provider=settings.llm_provider,
                prompt=messages,
                completion=llm_res.content,
                prompt_tokens=llm_res.prompt_tokens,
                completion_tokens=llm_res.completion_tokens,
                latency_ms=gen_latency_ms,
            )
            self.langfuse.flush()

        # 5. Build source attributions
        sources = self._build_source_chunks(used_chunks)

        return AskResponse(
            answer=llm_res.content,
            sources=sources,
            retrieved_chunk_count=len(retrieved_chunks),
            used_chunk_count=len(used_chunks),
            took_ms=took_ms,
            prompt_tokens=llm_res.prompt_tokens,
            completion_tokens=llm_res.completion_tokens,
        )

    async def answer_stream(self, request: AskRequest) -> AsyncIterator[dict[str, Any]]:
        """Execute streaming RAG pipeline yielding token chunks and final metadata."""
        start_time = time.perf_counter()

        trace = (
            self.langfuse.start_trace(
                name="rag_ask",
                metadata={"query": request.query, "mode": request.mode, "stream": True},
            )
            if self.langfuse
            else None
        )

        # 1. Retrieve candidate chunks with hybrid_search span
        search_span = (
            self.langfuse.start_span(
                trace_or_span=trace,
                name="hybrid_search",
                input_data={
                    "query": request.query,
                    "mode": request.mode,
                    "category": request.category,
                    "top_k": request.top_k,
                },
            )
            if self.langfuse
            else None
        )

        retrieved_chunks = self.retrieve(request)

        if self.langfuse and search_span:
            top_scores = [round(c.score, 4) for c in retrieved_chunks[:5]]
            self.langfuse.end_span(
                search_span,
                output_data={"hit_count": len(retrieved_chunks), "top_scores": top_scores},
            )

        if not retrieved_chunks:
            took_ms = round((time.perf_counter() - start_time) * 1000, 2)
            if self.langfuse:
                self.langfuse.flush()
            yield {"event": "token", "data": NO_CONTEXT_ANSWER}
            yield {
                "event": "metadata",
                "data": {
                    "sources": [],
                    "retrieved_chunk_count": 0,
                    "used_chunk_count": 0,
                    "took_ms": took_ms,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            }
            return

        # 2. Build context block with build_prompt span
        prompt_span = (
            self.langfuse.start_span(
                trace_or_span=trace,
                name="build_prompt",
                input_data={
                    "max_context_tokens": self.max_context_tokens,
                    "retrieved_chunk_count": len(retrieved_chunks),
                },
            )
            if self.langfuse
            else None
        )

        context_block, used_chunks = build_context_block(
            chunks=retrieved_chunks,
            max_tokens=self.max_context_tokens,
        )

        if self.langfuse and prompt_span:
            self.langfuse.end_span(
                prompt_span,
                output_data={"used_chunk_count": len(used_chunks)},
            )

        if not used_chunks:
            took_ms = round((time.perf_counter() - start_time) * 1000, 2)
            if self.langfuse:
                self.langfuse.flush()
            yield {"event": "token", "data": NO_CONTEXT_ANSWER}
            yield {
                "event": "metadata",
                "data": {
                    "sources": [],
                    "retrieved_chunk_count": len(retrieved_chunks),
                    "used_chunk_count": 0,
                    "took_ms": took_ms,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            }
            return

        # 3. Construct messages and stream tokens
        messages = build_messages(
            system_prompt=self.system_prompt,
            context_block=context_block,
            user_question=request.query,
        )

        sources = self._build_source_chunks(used_chunks)

        full_tokens: list[str] = []
        gen_start = time.perf_counter()
        async for token in self.llm.stream(messages=messages):
            full_tokens.append(token)
            yield {"event": "token", "data": token}

        gen_latency_ms = round((time.perf_counter() - gen_start) * 1000, 2)
        took_ms = round((time.perf_counter() - start_time) * 1000, 2)
        completion_text = "".join(full_tokens)

        prompt_tokens = max(1, len(str(messages)) // 4)
        completion_tokens = max(1, len(completion_text) // 4)

        if self.langfuse and trace:
            self.langfuse.log_generation(
                trace_or_span=trace,
                name="llm_generation",
                model=settings.llm_model,
                provider=settings.llm_provider,
                prompt=messages,
                completion=completion_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=gen_latency_ms,
            )
            self.langfuse.flush()

        yield {
            "event": "metadata",
            "data": {
                "sources": [s.model_dump() for s in sources],
                "retrieved_chunk_count": len(retrieved_chunks),
                "used_chunk_count": len(used_chunks),
                "took_ms": took_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
