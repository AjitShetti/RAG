"""Provider adapter for OpenAI-compatible LLM APIs (OpenAI, Groq, NVIDIA NIM, etc.).

Handles authentication, retries, rate limits, error mapping, and streaming response parsing.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ....config import settings
from ..exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Adapter for OpenAI API and any OpenAI-compatible REST endpoint (Groq, NIM, Ollama, etc.)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.llm_base_url

        if not self.api_key:
            logger.warning("LLM API key is empty. API calls will fail unless endpoint is unauthenticated.")

        self._sync_client = openai.OpenAI(
            api_key=self.api_key or "missing-key",
            base_url=self.base_url if self.base_url else None,
            timeout=60.0,
        )
        self._async_client = openai.AsyncOpenAI(
            api_key=self.api_key or "missing-key",
            base_url=self.base_url if self.base_url else None,
            timeout=60.0,
        )

    @retry(
        retry=retry_if_exception_type((openai.APITimeoutError, openai.APIConnectionError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=2),
        reraise=True,
    )
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute blocking completion call to OpenAI-compatible LLM.

        Returns dict with keys: 'content', 'model', 'prompt_tokens', 'completion_tokens'.
        """
        try:
            response = self._sync_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            usage = response.usage

            return {
                "content": content,
                "model": response.model or self.model,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            }
        except openai.AuthenticationError as exc:
            logger.error("LLM Authentication failed: %s", exc)
            raise LLMAuthenticationError(f"Invalid API key or authentication error: {exc}") from exc
        except openai.RateLimitError as exc:
            logger.error("LLM Rate limit exceeded: %s", exc)
            raise LLMRateLimitError(f"Rate limit exceeded: {exc}") from exc
        except openai.APITimeoutError as exc:
            logger.error("LLM Request timed out: %s", exc)
            raise LLMTimeoutError(f"Request timed out: {exc}") from exc
        except openai.APIError as exc:
            logger.error("LLM API error: %s", exc)
            raise LLMError(f"LLM Provider API error: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error in LLM generate: %s", exc)
            raise LLMError(f"Unexpected LLM error: {exc}") from exc

    async def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens as an async generator of delta text strings."""
        try:
            response_stream = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.AuthenticationError as exc:
            logger.error("LLM Streaming Authentication failed: %s", exc)
            raise LLMAuthenticationError(f"Authentication error: {exc}") from exc
        except openai.RateLimitError as exc:
            logger.error("LLM Streaming Rate limit exceeded: %s", exc)
            raise LLMRateLimitError(f"Rate limit exceeded: {exc}") from exc
        except openai.APITimeoutError as exc:
            logger.error("LLM Streaming Request timed out: %s", exc)
            raise LLMTimeoutError(f"Streaming timed out: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error in LLM stream: %s", exc)
            raise LLMError(f"LLM streaming error: {exc}") from exc
