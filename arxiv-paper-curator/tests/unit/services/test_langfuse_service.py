"""Unit tests for LangfuseService."""

from unittest.mock import MagicMock, patch

from src.services.langfuse.service import LangfuseService


def test_langfuse_disabled_when_keys_missing():
    with patch("src.services.langfuse.service.settings") as mock_settings:
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = ""
        mock_settings.langfuse_secret_key = ""
        mock_settings.langfuse_host = "https://cloud.langfuse.com"

        service = LangfuseService()
        assert not service.is_enabled
        assert service.start_trace("test") is None
        assert service.start_span(None, "span") is None


def test_langfuse_disabled_when_feature_flag_off():
    with patch("src.services.langfuse.service.settings") as mock_settings:
        mock_settings.langfuse_enabled = False
        mock_settings.langfuse_public_key = "pk-123"
        mock_settings.langfuse_secret_key = "sk-123"
        mock_settings.langfuse_host = "https://cloud.langfuse.com"

        service = LangfuseService()
        assert not service.is_enabled


def test_langfuse_with_mock_client():
    mock_client = MagicMock()
    service = LangfuseService(client=mock_client)
    assert service.is_enabled

    # Test start_trace
    service.start_trace("test_trace", metadata={"key": "val"})
    mock_client.trace.assert_called_once_with(name="test_trace", metadata={"key": "val"})

    # Test start_span
    mock_trace = MagicMock()
    service.start_span(mock_trace, "test_span", input_data={"q": "hi"})
    mock_trace.span.assert_called_once_with(name="test_span", input={"q": "hi"})

    # Test end_span
    mock_span = MagicMock()
    service.end_span(mock_span, output_data={"res": "ok"})
    mock_span.end.assert_called_once_with(output={"res": "ok"})

    # Test log_generation
    service.log_generation(
        mock_trace,
        name="llm_gen",
        model="llama-3.3-70b-versatile",
        provider="groq",
        prompt="hello",
        completion="world",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=120.0,
    )
    mock_trace.generation.assert_called_once()

    # Test log_cache_hit
    service.log_cache_hit(mock_trace, "key123")
    mock_trace.event.assert_called_once_with(name="cache_hit", input={"cache_key": "key123"})

    # Test flush
    service.flush()
    mock_client.flush.assert_called_once()


def test_langfuse_exception_resilience():
    mock_client = MagicMock()
    mock_client.trace.side_effect = Exception("Langfuse API error")

    service = LangfuseService(client=mock_client)
    # Should not raise exception
    res = service.start_trace("failing_trace")
    assert res is None


from unittest.mock import MagicMock, patch

import pytest

from src.services.langfuse import LangfuseService


def test_init_with_explicit_client():
    mock_client = MagicMock()
    service = LangfuseService(client=mock_client)

    assert service.is_enabled is True
    assert service.client == mock_client


@patch("src.services.langfuse.service.settings")
def test_init_disabled_by_settings_flag(mock_settings):
    mock_settings.langfuse_enabled = False
    mock_settings.langfuse_public_key = "pk-123"
    mock_settings.langfuse_secret_key = "sk-123"
    mock_settings.langfuse_host = "https://cloud.langfuse.com"

    service = LangfuseService()
    assert service.is_enabled is False
    assert service.client is None


@patch("src.services.langfuse.service.settings")
def test_init_disabled_by_missing_keys(mock_settings):
    mock_settings.langfuse_enabled = True
    mock_settings.langfuse_public_key = ""
    mock_settings.langfuse_secret_key = ""
    mock_settings.langfuse_host = "https://cloud.langfuse.com"

    service = LangfuseService()
    assert service.is_enabled is False
    assert service.client is None


@patch("src.services.langfuse.service.settings")

def test_init_exception_fallback(mock_settings):
    mock_settings.langfuse_enabled = True
    mock_settings.langfuse_public_key = "pk-123"
    mock_settings.langfuse_secret_key = "sk-123"
    mock_settings.langfuse_host = "https://cloud.langfuse.com"

    with patch("langfuse.Langfuse", side_effect=Exception("Connection error")):
        service = LangfuseService()
        assert service.is_enabled is False
        assert service.client is None


def test_start_trace():
    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_client.trace.return_value = mock_trace

    service = LangfuseService(client=mock_client)
    trace = service.start_trace(name="test_trace", metadata={"user": "test"})

    assert trace == mock_trace
    mock_client.trace.assert_called_once_with(name="test_trace", metadata={"user": "test"})


def test_start_trace_disabled():
    service = LangfuseService()  # Disabled by default in test env without keys
    trace = service.start_trace(name="test_trace")
    assert trace is None


def test_start_span():
    mock_client = MagicMock()
    service = LangfuseService(client=mock_client)

    mock_trace = MagicMock()
    mock_span = MagicMock()
    mock_trace.span.return_value = mock_span

    span = service.start_span(
        trace_or_span=mock_trace,
        name="test_span",
        input_data={"q": "test"},
        metadata={"step": 1},
    )

    assert span == mock_span
    mock_trace.span.assert_called_once_with(
        name="test_span", input={"q": "test"}, metadata={"step": 1}
    )


def test_start_span_when_none():
    mock_client = MagicMock()
    service = LangfuseService(client=mock_client)

    span = service.start_span(trace_or_span=None, name="test_span")
    assert span is None


def test_end_span():
    mock_client = MagicMock()
    service = LangfuseService(client=mock_client)

    mock_span = MagicMock()
    service.end_span(
        span=mock_span,
        output_data={"results": 5},
        metadata={"status": "ok"},
    )

    mock_span.end.assert_called_once_with(
        output={"results": 5}, metadata={"status": "ok"}
    )


def test_log_generation():
    mock_client = MagicMock()
    service = LangfuseService(client=mock_client)

    mock_trace = MagicMock()
    mock_gen = MagicMock()
    mock_trace.generation.return_value = mock_gen

    gen = service.log_generation(
        trace_or_span=mock_trace,
        name="llm_generation",
        model="llama-3.3-70b",
        provider="groq",
        prompt="Hello",
        completion="World",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=123.4,
        metadata={"temp": 0.7},
    )

    assert gen == mock_gen
    mock_trace.generation.assert_called_once_with(
        name="llm_generation",
        model="llama-3.3-70b",
        input="Hello",
        output="World",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        metadata={"temp": 0.7, "provider": "groq", "latency_ms": 123.4},
    )


def test_log_cache_hit():
    mock_client = MagicMock()
    service = LangfuseService(client=mock_client)

    mock_trace = MagicMock()
    service.log_cache_hit(trace=mock_trace, key="rag:ask:12345")

    mock_trace.event.assert_called_once_with(
        name="cache_hit", input={"cache_key": "rag:ask:12345"}
    )


def test_flush():
    mock_client = MagicMock()
    service = LangfuseService(client=mock_client)

    service.flush()
    mock_client.flush.assert_called_once()


def test_exception_guarding():
    """Ensure all service methods catch exceptions without propagating."""
    mock_client = MagicMock()
    mock_client.trace.side_effect = Exception("SDK error on trace")
    service = LangfuseService(client=mock_client)

    # All these calls should catch the exception and log a warning without raising
    assert service.start_trace(name="faulty_trace") is None

    faulty_trace = MagicMock()
    faulty_trace.span.side_effect = Exception("SDK error on span")
    assert service.start_span(faulty_trace, name="faulty_span") is None

    faulty_span = MagicMock()
    faulty_span.end.side_effect = Exception("SDK error on end_span")
    service.end_span(faulty_span)  # Should not raise

    faulty_trace.generation.side_effect = Exception("SDK error on generation")
    assert (
        service.log_generation(
            faulty_trace,
            name="gen",
            model="m",
            provider="p",
            prompt="",
            completion="",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1.0,
        )
        is None
    )

    faulty_trace.event.side_effect = Exception("SDK error on event")
    assert service.log_cache_hit(faulty_trace, "key") is None

    mock_client.flush.side_effect = Exception("SDK error on flush")
    service.flush()  # Should not raise
