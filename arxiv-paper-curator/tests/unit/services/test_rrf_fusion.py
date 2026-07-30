"""Unit tests for RRF Fusion logic."""

from src.services.opensearch.rrf import rrf_fusion


def test_rrf_score_formula():
    """Test that a document at rank 1 gets score 1/(60+1)."""
    keyword_hits = [{"chunk_id": "doc1", "title": "Doc One"}]
    semantic_hits = []

    results = rrf_fusion(keyword_hits, semantic_hits, k=60, id_field="chunk_id")

    assert len(results) == 1
    assert results[0].doc_id == "doc1"
    expected_score = round(1.0 / 61.0, 6)
    assert results[0].rrf_score == expected_score
    assert results[0].keyword_rank == 1
    assert results[0].semantic_rank is None
    assert results[0].contributed_by == ["keyword"]


def test_docs_in_both_lists_score_higher():
    """Test that a document appearing in both keyword and semantic lists gets a combined score."""
    keyword_hits = [
        {"chunk_id": "doc_keyword_only", "title": "K Only"},
        {"chunk_id": "doc_both", "title": "Both"},
    ]
    semantic_hits = [
        {"chunk_id": "doc_both", "title": "Both"},
        {"chunk_id": "doc_semantic_only", "title": "S Only"},
    ]

    results = rrf_fusion(keyword_hits, semantic_hits, k=60, id_field="chunk_id")

    assert results[0].doc_id == "doc_both"
    assert "keyword" in results[0].contributed_by
    assert "semantic" in results[0].contributed_by
    assert results[0].keyword_rank == 2
    assert results[0].semantic_rank == 1

    # Score for doc_both = 1/(60+2) + 1/(60+1)
    expected_both_score = round((1.0 / 62.0) + (1.0 / 61.0), 6)
    assert results[0].rrf_score == expected_both_score


def test_output_sorted_descending():
    """Test that RRF fusion results are sorted in descending order of score."""
    keyword_hits = [
        {"chunk_id": "rank1", "title": "Rank 1"},
        {"chunk_id": "rank2", "title": "Rank 2"},
        {"chunk_id": "rank3", "title": "Rank 3"},
    ]
    semantic_hits = []

    results = rrf_fusion(keyword_hits, semantic_hits, k=60, id_field="chunk_id")

    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_semantic_list():
    """Test that an empty semantic list acts as a clean keyword pass-through."""
    keyword_hits = [
        {"chunk_id": "docA"},
        {"chunk_id": "docB"},
    ]

    results = rrf_fusion(keyword_hits, [], k=60, id_field="chunk_id")

    assert len(results) == 2
    assert results[0].doc_id == "docA"
    assert results[1].doc_id == "docB"
