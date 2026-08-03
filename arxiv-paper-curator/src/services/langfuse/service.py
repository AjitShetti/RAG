"""Langfuse Observability Service.

Provides exception-guarded tracing, spans, generation logging, and cache hit events.
Gracefully handles missing credentials, feature flags, or SDK failures.
"""

from __future__ import annotations

import logging
from typing import Any

from ...config import settings

logger = logging.getLogger(__name__)


class LangfuseService:
    """Service wrapper around Langfuse SDK for application-wide observability."""

    def __init__(self, client: Any | None = None) -> None:
        self.client: Any | None = client
        self._enabled: bool = False

        if client is not None:
            self.client = client
            self._enabled = True
            return

        try:
            if (
                not settings.langfuse_enabled
                or not settings.langfuse_public_key
                or not settings.langfuse_secret_key
            ):
                logger.warning("Langfuse tracing disabled or credentials missing.")
                self._enabled = False
                self.client = None
            else:
                from langfuse import Langfuse

                self.client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                self._enabled = True
        except Exception as exc:
            logger.warning("Failed to initialize Langfuse client: %s", exc)
            self._enabled = False
            self.client = None

    @property
    def is_enabled(self) -> bool:
        """Return True if Langfuse client is active and enabled."""
        return self._enabled

    def start_trace(self, name: str, metadata: dict[str, Any] | None = None) -> Any:
        """Create a Langfuse trace object if enabled."""
        if not self._enabled or self.client is None:
            return None
        try:
            kwargs: dict[str, Any] = {"name": name}
            if metadata is not None:
                kwargs["metadata"] = metadata
            return self.client.trace(**kwargs)
        except Exception as exc:
            logger.warning("Failed to start Langfuse trace %r: %s", name, exc)
            return None

    def start_span(
        self,
        trace_or_span: Any,
        name: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create a span on a trace or parent span."""
        if not self._enabled or trace_or_span is None:
            return None
        try:
            if hasattr(trace_or_span, "span"):
                kwargs: dict[str, Any] = {"name": name}
                if input_data is not None:
                    kwargs["input"] = input_data
                if metadata is not None:
                    kwargs["metadata"] = metadata
                return trace_or_span.span(**kwargs)
            return None
        except Exception as exc:
            logger.warning("Failed to start Langfuse span %r: %s", name, exc)
            return None

    def end_span(
        self,
        span: Any,
        output_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """End an active span with optional output data and metadata."""
        if not self._enabled or span is None:
            return
        try:
            kwargs: dict[str, Any] = {}
            if output_data is not None:
                kwargs["output"] = output_data
            if metadata is not None:
                kwargs["metadata"] = metadata

            if hasattr(span, "end"):
                span.end(**kwargs)
            elif hasattr(span, "update"):
                span.update(**kwargs)
        except Exception as exc:
            logger.warning("Failed to end Langfuse span: %s", exc)

    def log_generation(
        self,
        trace_or_span: Any,
        name: str,
        model: str,
        provider: str,
        prompt: Any,
        completion: Any,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Record an LLM generation on a trace or span."""
        if not self._enabled or trace_or_span is None:
            return None
        try:
            meta = dict(metadata or {})
            meta["provider"] = provider
            meta["latency_ms"] = latency_ms

            kwargs: dict[str, Any] = {
                "name": name,
                "model": model,
                "input": prompt,
                "output": completion,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "metadata": meta,
            }

            if hasattr(trace_or_span, "generation"):
                return trace_or_span.generation(**kwargs)
            return None
        except Exception as exc:
            logger.warning("Failed to log Langfuse generation %r: %s", name, exc)
            return None

    def log_cache_hit(self, trace: Any, key: str) -> Any:
        """Log a cache hit event or span on a trace."""
        if not self._enabled or trace is None:
            return None
        try:
            if hasattr(trace, "event"):
                return trace.event(name="cache_hit", input={"cache_key": key})
            elif hasattr(trace, "span"):
                span = trace.span(name="cache_hit", input={"cache_key": key})
                if hasattr(span, "end"):
                    span.end()
                return span
            return None
        except Exception as exc:
            logger.warning("Failed to log cache hit event: %s", exc)
            return None

    def flush(self) -> None:
        """Flush pending Langfuse telemetry events."""
        if not self._enabled or self.client is None:
            return
        try:
            if hasattr(self.client, "flush"):
                self.client.flush()
        except Exception as exc:
            logger.warning("Failed to flush Langfuse client: %s", exc)
