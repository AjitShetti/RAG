"""Reciprocal Rank Fusion (RRF) logic.

Combines ranked search result lists from multiple retrieval strategies (e.g. BM25 keyword + kNN vector)
without requiring raw score normalization.

Formula: RRF_Score(d) = Σ 1 / (k + rank(d))
Default constant k = 60.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class RRFResult:
    """Merged search result item produced by RRF fusion."""

    doc_id: str
    rrf_score: float
    normalized_score: float = 0.0
    keyword_rank: int | None = None
    semantic_rank: int | None = None
    contributed_by: list[str] = field(default_factory=list)
    source_doc: dict[str, Any] = field(default_factory=dict)
    highlights: dict[str, list[str]] = field(default_factory=dict)


def rrf_fusion(
    keyword_hits: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    k: int = 60,
    id_field: str = "chunk_id",
) -> list[RRFResult]:
    """Combine keyword and semantic ranked result lists using Reciprocal Rank Fusion.

    :param keyword_hits: List of hit dicts from BM25 search, ordered by rank desc.
    :param semantic_hits: List of hit dicts from kNN search, ordered by rank desc.
    :param k: RRF constant smoothing factor (default 60).
    :param id_field: Field name used to identify documents across lists.
    :return: List of RRFResult instances sorted descending by rrf_score.
    """
    fused_scores: dict[str, float] = {}
    keyword_ranks: dict[str, int] = {}
    semantic_ranks: dict[str, int] = {}
    doc_sources: dict[str, dict[str, Any]] = {}
    doc_highlights: dict[str, dict[str, list[str]]] = {}
    contributed_map: dict[str, set[str]] = {}

    # Process keyword hits
    for rank_0, hit in enumerate(keyword_hits):
        rank = rank_0 + 1  # 1-indexed rank
        doc_id = hit.get(id_field) or hit.get("_id", "")
        if not doc_id:
            continue

        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        keyword_ranks[doc_id] = rank

        if doc_id not in doc_sources:
            doc_sources[doc_id] = hit.get("_source", hit)
        if doc_id not in doc_highlights and "highlight" in hit:
            doc_highlights[doc_id] = hit["highlight"]

        contributed_map.setdefault(doc_id, set()).add("keyword")

    # Process semantic hits
    for rank_0, hit in enumerate(semantic_hits):
        rank = rank_0 + 1  # 1-indexed rank
        doc_id = hit.get(id_field) or hit.get("_id", "")
        if not doc_id:
            continue

        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        semantic_ranks[doc_id] = rank

        if doc_id not in doc_sources:
            doc_sources[doc_id] = hit.get("_source", hit)
        if doc_id not in doc_highlights and "highlight" in hit:
            doc_highlights[doc_id] = hit["highlight"]

        contributed_map.setdefault(doc_id, set()).add("semantic")

    # Determine max theoretical score for normalization
    # If both legs produced hits, max possible score is 2 / (k + 1); if single leg, 1 / (k + 1).
    num_active_legs = (1 if keyword_hits else 0) + (1 if semantic_hits else 0)
    max_theoretical_rrf = (num_active_legs / (k + 1)) if num_active_legs > 0 else (1.0 / (k + 1))

    # Build final RRFResult list
    results: list[RRFResult] = []
    for doc_id, score in fused_scores.items():
        contrib = sorted(list(contributed_map[doc_id]))
        norm_score = min(1.0, score / max_theoretical_rrf)
        results.append(
            RRFResult(
                doc_id=doc_id,
                rrf_score=round(score, 6),
                normalized_score=round(norm_score, 4),
                keyword_rank=keyword_ranks.get(doc_id),
                semantic_rank=semantic_ranks.get(doc_id),
                contributed_by=contrib,
                source_doc=doc_sources[doc_id],
                highlights=doc_highlights.get(doc_id, {}),
            )
        )

    # Sort descending by fused score
    results.sort(key=lambda r: r.rrf_score, reverse=True)
    return results
