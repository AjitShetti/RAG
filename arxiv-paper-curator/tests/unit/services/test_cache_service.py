"""Unit tests for CacheService."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
import redis

from src.config import settings
from src.schemas.rag.ask import AskRequest, AskResponse, SourceChunk
from src.services.cache.service import CacheService
from src.services.rag.pipeline import NO_CONTEXT_ANSWER


def make_dummy_ask_response(
    answer: str = "This is a grounded answer.",
    used_chunk_count: int = 1,
    cached: bool = False,
) -> AskResponse:
    """Helper to construct AskResponse objects for testing."""
    return AskResponse(
        answer=answer,
        sources=[
            SourceChunk(
                paper_id="2301.00001",
                title="Test Paper",
                section_name="Abstract",
                snippet="Test snippet",
                relevance_score=0.9,
                pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
            )
        ],
        retrieved_chunk_count=1,
        used_chunk_count=used_chunk_count,
        took_ms=120.0,
        prompt_tokens=100,
        completion_tokens=20,
        cached=cached,
    )


@pytest.fixture
def mock_redis():
    return MagicMock(spec=redis.Redis)


@pytest.fixture
def cache_service(mock_redis):
    return CacheService(redis_client=mock_redis)


def test_deterministic_key_generation(cache_service):
    """Test deterministic key generation and field sensitivity."""
    req1 = AskRequest(
        query=" What is Transformer? ",
        mode="hybrid",
        category="cs.AI",
        date_from=date(2023, 1, 1),
        date_to=date(2023, 12, 31),
        top_k=8,
    )
    req2 = AskRequest(
        query="what is transformer?",
        mode="hybrid",
        category="cs.AI",
        date_from=date(2023, 1, 1),
        date_to=date(2023, 12, 31),
        top_k=8,
    )
    key1 = cache_service.get_cache_key(req1)
    key2 = cache_service.get_cache_key(req2)

    assert key1.startswith("v1:rag:")
    assert key1 == key2

    # Changing category produces different key
    req_cat = AskRequest(
        query="what is transformer?",
        mode="hybrid",
        category="cs.CL",
        date_from=date(2023, 1, 1),
        date_to=date(2023, 12, 31),
        top_k=8,
    )
    assert cache_service.get_cache_key(req_cat) != key1

    # Changing mode produces different key
    req_mode = AskRequest(
        query="what is transformer?",
        mode="semantic",
        category="cs.AI",
        date_from=date(2023, 1, 1),
        date_to=date(2023, 12, 31),
        top_k=8,
    )
    assert cache_service.get_cache_key(req_mode) != key1

    # Changing top_k produces different key
    req_topk = AskRequest(
        query="what is transformer?",
        mode="hybrid",
        category="cs.AI",
        date_from=date(2023, 1, 1),
        date_to=date(2023, 12, 31),
        top_k=5,
    )
    assert cache_service.get_cache_key(req_topk) != key1


def test_get_hit(cache_service, mock_redis):
    """Test cache get hit returns AskResponse with cached=True."""
    dummy_res = make_dummy_ask_response(cached=False)
    mock_redis.get.return_value = dummy_res.model_dump_json()

    res = cache_service.get("v1:rag:dummy_key")
    assert res is not None
    assert res.answer == dummy_res.answer
    assert res.cached is True
    mock_redis.get.assert_called_once_with("v1:rag:dummy_key")


def test_get_miss(cache_service, mock_redis):
    """Test cache get miss returns None."""
    mock_redis.get.return_value = None

    res = cache_service.get("v1:rag:dummy_key")
    assert res is None


def test_set_success(cache_service, mock_redis):
    """Test setting cache entry with custom TTL."""
    dummy_res = make_dummy_ask_response()

    success = cache_service.set("v1:rag:dummy_key", dummy_res, ttl_seconds=1800)
    assert success is True
    mock_redis.set.assert_called_once()
    args, kwargs = mock_redis.set.call_args
    key_name = kwargs.get("name") if "name" in kwargs else args[0]
    assert key_name == "v1:rag:dummy_key"
    assert kwargs.get("ex") == 1800


def test_set_default_ttl(cache_service, mock_redis):
    """Test setting cache entry with default configured TTL."""
    dummy_res = make_dummy_ask_response()

    success = cache_service.set("v1:rag:dummy_key", dummy_res)
    assert success is True
    args, kwargs = mock_redis.set.call_args
    assert kwargs.get("ex") == settings.cache_ttl_seconds


def test_set_skip_no_context_answer(cache_service, mock_redis):
    """Test skipping cache set when answer is NO_CONTEXT_ANSWER."""
    res = make_dummy_ask_response(answer=NO_CONTEXT_ANSWER, used_chunk_count=1)

    success = cache_service.set("v1:rag:dummy_key", res)
    assert success is False
    assert not mock_redis.set.called


def test_set_skip_zero_used_chunks(cache_service, mock_redis):
    """Test skipping cache set when used_chunk_count is 0."""
    res = make_dummy_ask_response(answer="Some answer", used_chunk_count=0)

    success = cache_service.set("v1:rag:dummy_key", res)
    assert success is False
    assert not mock_redis.set.called


def test_redis_exceptions_handled_gracefully(cache_service, mock_redis):
    """Test Redis exceptions are caught and return None/False cleanly."""
    mock_redis.get.side_effect = redis.RedisError("Connection lost")
    mock_redis.set.side_effect = redis.RedisError("Connection lost")
    mock_redis.delete.side_effect = redis.RedisError("Connection lost")
    mock_redis.flushdb.side_effect = redis.RedisError("Connection lost")

    dummy_res = make_dummy_ask_response()

    assert cache_service.get("key") is None
    assert cache_service.set("key", dummy_res) is False
    assert cache_service.delete("key") is False
    assert cache_service.flush_all() is False


def test_cache_disabled_flag(mock_redis):
    """Test cache_enabled=False feature flag disables get, set, delete."""
    with patch.object(settings, "cache_enabled", False):
        service = CacheService(redis_client=mock_redis)
        dummy_res = make_dummy_ask_response()

        assert service.get("key") is None
        assert service.set("key", dummy_res) is False
        assert service.delete("key") is False
        assert not mock_redis.get.called
        assert not mock_redis.set.called
        assert not mock_redis.delete.called


def test_delete_and_flush_all(cache_service, mock_redis):
    """Test invalidation methods delete and flush_all."""
    mock_redis.delete.return_value = 1
    mock_redis.flushdb.return_value = True

    assert cache_service.delete("v1:rag:key") is True
    mock_redis.delete.assert_called_once_with("v1:rag:key")

    assert cache_service.flush_all() is True
    mock_redis.flushdb.assert_called_once()
