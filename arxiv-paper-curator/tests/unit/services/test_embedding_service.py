"""Unit tests for RestEmbeddingService."""

from unittest.mock import MagicMock

import httpx
import pytest

from src.services.embeddings.rest_provider import RestEmbeddingService


@pytest.fixture
def mock_http_client():
    return MagicMock(spec=httpx.Client)


def test_batching_respects_batch_size(mock_http_client):
    """Test that texts are split into batches according to batch_size."""

    def mock_post_side_effect(url, headers=None, json=None):
        batch_input = json.get("input", [])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"index": idx, "embedding": [0.1 * (idx + 1), 0.2]}
                for idx in range(len(batch_input))
            ]
        }
        return mock_resp

    mock_http_client.post.side_effect = mock_post_side_effect

    service = RestEmbeddingService(
        api_key="test_key",
        api_url="http://test.api/embeddings",
        batch_size=2,
        http_client=mock_http_client,
    )

    texts = ["text1", "text2", "text3", "text4", "text5"]
    embeddings = service.embed(texts)

    # 5 texts / batch_size 2 = 3 requests
    assert mock_http_client.post.call_count == 3
    assert len(embeddings) == 5
    assert all(emb is not None for emb in embeddings)


def test_failed_batch_returns_none_slots(mock_http_client):
    """Test that a failing batch logs warning and returns None for that batch without crashing."""
    mock_http_client.post.side_effect = httpx.ConnectError("Connection refused")

    service = RestEmbeddingService(
        api_key="test_key",
        api_url="http://test.api/embeddings",
        batch_size=2,
        http_client=mock_http_client,
    )

    texts = ["text1", "text2"]
    embeddings = service.embed(texts)

    assert len(embeddings) == 2
    assert embeddings[0] is None
    assert embeddings[1] is None
