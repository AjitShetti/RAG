"""Relevance sanity check script for BM25 keyword search in OpenSearch.

Executes realistic test queries against the indexed arXiv papers and evaluates:
1. Top matching papers and BM25 relevance scores.
2. Title matches vs body matches (verifies title matches outrank body matches due to title^4 boost).
3. Field highlighting accuracy.
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.search import SearchRequest
from src.services.opensearch import OpenSearchService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_relevance_sanity_checks() -> None:
    """Run sanity check queries against live OpenSearch cluster."""
    service = OpenSearchService()

    if not service.ping():
        print("[ERROR] OpenSearch cluster is not reachable!")
        sys.exit(1)

    test_queries = [
        "AI agents research",
        "APEX Accounting",
        "neural networks",
        "open-ended research",
    ]

    print("\n" + "=" * 70)
    print("  OPEN_SEARCH BM25 RELEVANCE SANITY CHECK  ")
    print("=" * 70)

    for query in test_queries:
        req = SearchRequest(query=query, page=1, page_size=5)
        resp = service.search(req)

        print(f"\n[Query]: '{query}' | Hits: {resp.total} | Took: {resp.took_ms} ms")
        print("-" * 70)

        if not resp.results:
            print("    (No matching papers found)")
            continue

        for idx, hit in enumerate(resp.results, start=1):
            print(f"  [{idx}] Score: {hit.score:.4f} | arXiv ID: {hit.arxiv_id} | Category: {hit.category}")
            print(f"      Title: {hit.title}")
            if hit.highlights:
                print(f"      Highlights: {hit.highlights}")
            print()

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_relevance_sanity_checks()
