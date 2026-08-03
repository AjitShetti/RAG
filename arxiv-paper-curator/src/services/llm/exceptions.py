"""Exceptions for LLM service and provider interactions."""


class LLMError(Exception):
    """Base exception for all LLM errors."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider returns a rate-limit error (HTTP 429)."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when request to LLM provider times out."""

    pass


class LLMAuthenticationError(LLMError):
    """Raised when API key or authentication fails."""

    pass
