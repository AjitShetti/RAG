"""Embedding provider re-exports and factory function."""

from ...config import settings
from .base import EmbeddingService
from .rest_provider import RestEmbeddingService


def get_embedding_service() -> EmbeddingService:
    """Factory returning the configured embedding service instance."""
    provider = settings.embeddings_provider.lower()
    if provider in ("nvidia", "jina", "openai", "rest"):
        return RestEmbeddingService()

    raise ValueError(f"Unsupported embeddings provider: '{provider}'")


__all__ = ["EmbeddingService", "RestEmbeddingService", "get_embedding_service"]
