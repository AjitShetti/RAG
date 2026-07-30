"""Comparison script for Keyword (BM25), Semantic (kNN), and Hybrid (RRF) search modes.

Executes test queries across all three search modes and evaluates:
1. Retrieval mode contribution ('keyword', 'semantic', or both 'keyword', 'semantic').
2. Ranking shifts between BM25 exact match vs kNN vector similarity.
3. RRF score combination.
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.search import HybridSearchRequest
from src.services.embeddings import get_embedding_service
from src.services.opensearch import OpenSearchService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def compare_modes() -> None:
    """Run comparison queries through keyword, semantic, and hybrid search modes."""
    opensearch_service = OpenSearchService()
    embedding_service = get_embedding_service()

    if not opensearch_service.ping():
        print("[ERROR] OpenSearch cluster is not reachable!")
        sys.exit(1)

    queries = [
        "AI agents research",
        "APEX Accounting",
        "how AI models conduct open-ended scientific tasks",
        "measuring performance of accountants",
    ]

    print("\n" + "=" * 80)
    print("  SEARCH MODE COMPARISON: KEYWORD vs SEMANTIC vs HYBRID (RRF)  ")
    print("=" * 80)

    for query in queries:
        print(f"\n[Query]: '{query}'")
        print("-" * 80)

        for mode in ["keyword", "semantic", "hybrid"]:
            req = HybridSearchRequest(query=query, mode=mode, page=1, page_size=3)
            resp = opensearch_service.hybrid_search(
                request=req,
                embedding_service=embedding_service,
            )

            print(f"  Mode: {mode.upper():<8} | Hits: {resp.total} | Took: {resp.took_ms:.1f} ms")

            for hit in resp.results:
                contrib = ", ".join(hit.contributed_by) if hit.contributed_by else mode
                print(
                    f"    • [{hit.score:.4f}] Section: {hit.section_name:<15} | "
                    f"Paper: {hit.paper_id} | Contrib: [{contrib}]"
                )
                clean_snippet = hit.text[:100].encode("ascii", "ignore").decode("ascii").replace("\n", " ")
                print(f"      Text snippet: {clean_snippet}…")
            if not resp.results:
                print("    • (No results returned)")
            print()

    print("=" * 80 + "\n")


if __name__ == "__main__":
    compare_modes()
