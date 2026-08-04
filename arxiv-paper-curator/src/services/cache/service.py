"""Redis Caching Service for RAG application."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from ...config import settings
from ...schemas.rag import AgenticAskResponse, AskRequest, AskResponse
from ..rag.pipeline import NO_CONTEXT_ANSWER

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-backed response caching service for Ask endpoints."""

    def __init__(self, redis_client: Any | None = None) -> None:
        self.enabled = settings.cache_enabled
        self.ttl = settings.cache_ttl_seconds
        self.client: Any | None = redis_client

        if self.enabled and self.client is None:
            try:
                import redis

                self.client = redis.Redis.from_url(
                    settings.redis_url, decode_responses=True
                )
            except Exception as exc:
                logger.warning("Failed to initialize Redis cache client: %s", exc)
                self.enabled = False
                self.client = None

    def get_cache_key(self, request: AskRequest) -> str:
        """Generate deterministic SHA256 cache key from AskRequest parameters."""
        raw_key = (
            f"query={request.query.lower().strip()}|"
            f"mode={request.mode}|"
            f"category={request.category or ''}|"
            f"date_from={request.date_from or ''}|"
            f"date_to={request.date_to or ''}|"
            f"top_k={request.top_k}|"
            f"model={settings.llm_model}"
        )
        hashed = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return f"v1:rag:{hashed}"

    def get(self, key: str) -> AskResponse | None:
        """Fetch cached AskResponse if enabled and available."""
        if not self.enabled or self.client is None:
            return None
        try:
            val = self.client.get(key)
            if val:
                data = json.loads(val)
                response = AskResponse.model_validate(data)
                response.cached = True
                return response
        except Exception as exc:
            logger.warning("Failed to get key %r from Redis cache: %s", key, exc)
        return None

    def set(
        self,
        key: str,
        response: AskResponse,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store AskResponse in Redis cache with TTL. Skip if no context or zero chunks used."""
        if not self.enabled or self.client is None:
            return False

        # Skip caching for zero-results / no-context answers
        if response.answer == NO_CONTEXT_ANSWER or response.used_chunk_count == 0:
            logger.info("Skipping cache storage for no-context answer")
            return False

        ttl = ttl_seconds if ttl_seconds is not None else self.ttl
        try:
            dump_data = response.model_dump()
            dump_data["cached"] = True
            self.client.set(name=key, value=json.dumps(dump_data, default=str), ex=ttl)
            return True
        except Exception as exc:
            logger.warning("Failed to set key %r in Redis cache: %s", key, exc)
            return False

    def get_agentic_cache_key(self, request: AskRequest) -> str:
        """Generate deterministic SHA256 cache key with v1:agentic: prefix."""
        raw_key = (
            f"query={request.query.lower().strip()}|"
            f"mode={request.mode}|"
            f"category={request.category or ''}|"
            f"date_from={request.date_from or ''}|"
            f"date_to={request.date_to or ''}|"
            f"top_k={request.top_k}|"
            f"model={settings.llm_model}"
        )
        hashed = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return f"v1:agentic:{hashed}"

    def get_agentic(self, key: str) -> AgenticAskResponse | None:
        """Fetch cached AgenticAskResponse if enabled and available."""
        if not self.enabled or self.client is None:
            return None
        try:
            val = self.client.get(key)
            if val:
                data = json.loads(val)
                response = AgenticAskResponse.model_validate(data)
                response.cached = True
                return response
        except Exception as exc:
            logger.warning("Failed to get key %r from Redis cache: %s", key, exc)
        return None

    def set_agentic(
        self,
        key: str,
        response: AgenticAskResponse,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store AgenticAskResponse in Redis cache with TTL."""
        if not self.enabled or self.client is None:
            return False

        if response.answer == NO_CONTEXT_ANSWER or response.rejected:
            logger.info("Skipping cache storage for rejected/no-context answer")
            return False

        ttl = ttl_seconds if ttl_seconds is not None else self.ttl
        try:
            dump_data = response.model_dump()
            dump_data["cached"] = True
            self.client.set(name=key, value=json.dumps(dump_data, default=str), ex=ttl)
            return True
        except Exception as exc:
            logger.warning("Failed to set key %r in Redis cache: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        """Delete specific key from Redis cache."""
        if not self.enabled or self.client is None:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception as exc:
            logger.warning("Failed to delete key %r from Redis cache: %s", key, exc)
            return False

    def flush_all(self) -> bool:
        """Flush database cache entries."""
        if not self.enabled or self.client is None:
            return False
        try:
            self.client.flushdb()
            return True
        except Exception as exc:
            logger.warning("Failed to flush Redis cache: %s", exc)
            return False

