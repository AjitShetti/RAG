"""Unit tests for OpenSearchService."""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.models.paper import Paper
from src.schemas.search import SearchRequest
from src.services.opensearch import INDEX_NAME, OpenSearchService


@pytest.fixture
def mock_opensearch_client():
    """Fixture providing a mocked OpenSearch client."""
    client = MagicMock()
    return client


@pytest.fixture
def opensearch_service(mock_opensearch_client):
    """Fixture providing an OpenSearchService initialized with a mock client."""
    return OpenSearchService(client=mock_opensearch_client)


@pytest.fixture
def sample_paper():
    """Fixture providing a sample Paper instance."""
    return Paper(
        arxiv_id="2401.00001v1",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        pdf_url="https://arxiv.org/pdf/2401.00001v1",
        published_date=datetime.date(2024, 1, 1),
        category="cs.AI",
        full_text="Full paper content text here...",
        sections=[{"heading": "Introduction", "text": "Self-attention mechanism..."}],
        parse_status="success",
    )


def test_create_index_when_not_exists(opensearch_service, mock_opensearch_client):
    """Test that create_index creates the index when it does not exist."""
    mock_opensearch_client.indices.exists.return_value = False

    created = opensearch_service.create_index("test_index")

    assert created is True
    mock_opensearch_client.indices.exists.assert_called_once_with(index="test_index")
    mock_opensearch_client.indices.create.assert_called_once()


def test_create_index_idempotent_when_exists(opensearch_service, mock_opensearch_client):
    """Test that create_index skips creation when the index already exists."""
    mock_opensearch_client.indices.exists.return_value = True

    created = opensearch_service.create_index("test_index")

    assert created is False
    mock_opensearch_client.indices.exists.assert_called_once_with(index="test_index")
    mock_opensearch_client.indices.create.assert_not_called()


def test_search_query_boosting(opensearch_service, mock_opensearch_client):
    """Test that search constructs a multi_match query with correctly boosted fields."""
    mock_opensearch_client.search.return_value = {
        "took": 5,
        "hits": {"total": {"value": 1}, "hits": []},
    }

    req = SearchRequest(query="attention mechanism", page=1, page_size=10)
    opensearch_service.search(req, index_name="test_index")

    mock_opensearch_client.search.assert_called_once()
    call_args = mock_opensearch_client.search.call_args
    body = call_args.kwargs["body"]

    multi_match = body["query"]["bool"]["must"][0]["multi_match"]
    assert multi_match["query"] == "attention mechanism"
    assert "title^4" in multi_match["fields"]
    assert "abstract^2" in multi_match["fields"]
    assert "section_headings^1.5" in multi_match["fields"]
    assert "section_bodies^1" in multi_match["fields"]


def test_search_category_and_date_filters(opensearch_service, mock_opensearch_client):
    """Test that search applies category term filter and date range filter correctly."""
    mock_opensearch_client.search.return_value = {
        "took": 3,
        "hits": {"total": {"value": 0}, "hits": []},
    }

    req = SearchRequest(
        query="transformers",
        category="cs.AI",
        date_from=datetime.date(2024, 1, 1),
        date_to=datetime.date(2024, 12, 31),
        page=2,
        page_size=5,
    )
    opensearch_service.search(req, index_name="test_index")

    call_args = mock_opensearch_client.search.call_args
    body = call_args.kwargs["body"]

    filters = body["query"]["bool"]["filter"]
    assert len(filters) == 2
    assert {"term": {"category": "cs.AI"}} in filters
    assert {"range": {"published_date": {"gte": "2024-01-01", "lte": "2024-12-31"}}} in filters

    assert body["from"] == 5
    assert body["size"] == 5


@patch("src.services.opensearch.service.helpers.bulk")
def test_bulk_index(mock_bulk, opensearch_service, sample_paper):
    """Test that bulk_index formats actions and delegates to opensearchpy.helpers.bulk."""
    mock_bulk.return_value = (1, [])

    success_count, errors = opensearch_service.bulk_index([sample_paper], index_name="test_index")

    assert success_count == 1
    assert errors == []
    mock_bulk.assert_called_once()
    call_args = mock_bulk.call_args
    actions = call_args[0][1]
    assert len(actions) == 1
    assert actions[0]["_id"] == "2401.00001v1"
    assert actions[0]["_source"]["title"] == "Attention Is All You Need"
