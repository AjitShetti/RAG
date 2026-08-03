"""Provider-agnostic LLM Client interface.

Handles provider routing, latency measurement, usage logging, and unified response wrapping.
Routers and RAG pipeline code interact exclusively with LLMClient.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from ...config import settings
from .exceptions import LLMError
from .providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standard response object returned by LLMClient.generate()."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


class LLMClient:
    """Unified client abstraction wrapping underlying LLM provider adapters."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.provider_name = (provider or settings.llm_provider).lower()
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url or settings.llm_base_url

        # Select provider implementation
        if self.provider_name in ("openai", "groq", "nvidia", "ollama", "vllm"):
            self._provider = OpenAIProvider(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
            )
        else:
            logger.warning(
                "Unknown LLM_PROVIDER '%s' — falling back to OpenAI/Groq compatible provider",
                self.provider_name,
            )
            self._provider = OpenAIProvider(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
            )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute blocking completion call and log performance metrics."""
        start_time = time.perf_counter()
        raw_res = self._provider.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        prompt_tokens = raw_res.get("prompt_tokens", 0)
        completion_tokens = raw_res.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        logger.info(
            "[LLMClient] Provider=%s model=%s prompt_tokens=%d completion_tokens=%d latency_ms=%.2f",
            self.provider_name,
            self.model,
            prompt_tokens,
            completion_tokens,
            latency_ms,
        )

        return LLMResponse(
            content=raw_res["content"],
            model=raw_res.get("model", self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens as an async generator."""
        logger.info(
            "[LLMClient.stream] Provider=%s model=%s starting streaming response",
            self.provider_name,
            self.model,
        )
        async for token in self._provider.stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield token
