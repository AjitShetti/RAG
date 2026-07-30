"""REST-based embedding service implementation.

Supports NVIDIA Nim, OpenAI, Jina AI, and any OpenAI-compatible /v1/embeddings endpoint.
Handles batching, retries with exponential backoff on rate limits, and graceful fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ...config import settings
from .base import EmbeddingService

logger = logging.getLogger(__name__)


class RestEmbeddingService(EmbeddingService):
    """Embedding service using REST API endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_url: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key or settings.embeddings_api_key
        self._model = model or settings.embeddings_model
        self._api_url = api_url or settings.embeddings_api_url
        self._dimensions = dimensions or settings.embeddings_dimensions
        self._batch_size = batch_size or settings.embeddings_batch_size
        self._client = http_client or httpx.Client(timeout=30.0)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _send_batch_request(self, batch: list[str]) -> list[list[float]]:
        """Send a single batch request to the embeddings REST API."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Truncate text to max 250 words to strictly enforce NVIDIA 512 subword token limit
        safe_batch = [" ".join(text.split()[:250]) for text in batch]

        payload: dict[str, Any] = {
            "input": safe_batch,
            "model": self._model,
        }

        # Add input_type if NVIDIA endpoint
        if "nvidia" in self._api_url.lower():
            payload["input_type"] = "passage"

        response = self._client.post(self._api_url, headers=headers, json=payload)
        if not response.is_success:
            logger.error("Embedding API HTTP %d: %s", response.status_code, response.text)
        response.raise_for_status()

        data = response.json().get("data", [])
        # Sort by index to preserve input order
        sorted_data = sorted(data, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in sorted_data]

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a list of text strings into vector representations."""
        if not texts:
            return []

        results: list[list[float] | None] = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                batch_embeddings = self._send_batch_request(batch)
                results.extend(batch_embeddings)
            except Exception as exc:
                logger.warning(
                    "Embedding API call failed for batch [%d:%d]: %s — skipping batch",
                    i,
                    i + len(batch),
                    exc,
                )
                results.extend([None] * len(batch))

        return results
