"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    """Abstract interface for text embedding models."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a list of text strings into vector representations.

        Returns a list of vector floats (or None for failed texts).
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the vector dimensionality of the embedding model."""
        pass
