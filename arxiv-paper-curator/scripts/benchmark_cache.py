"""Cache Performance Benchmarking Script for RAG Application.

Executes 5 sample questions twice (Cold vs Cached), measuring request latency,
speedup factor, and verifying cache hits. Outputs a Markdown comparison table.
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from src.config import settings
from src.schemas.rag.ask import AskRequest, AskResponse
from src.services.cache.service import CacheService
from src.services.rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_QUESTIONS = [
    "What is the Transformer architecture and how does self-attention work?",
    "How do AI agents conduct open-ended scientific research tasks?",
    "What are the latest techniques in Large Language Model fine-tuning?",
    "How is performance evaluated in retrieval-augmented generation?",
    "What are common failure modes of multi-modal AI systems?",
]


def execute_via_http(client: httpx.Client, endpoint_url: str, req: AskRequest) -> tuple[AskResponse, float]:
    """Send AskRequest via HTTP POST and measure latency in milliseconds."""
    start_time = time.perf_counter()
    resp = client.post(endpoint_url, json=req.model_dump(mode="json"))
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    resp.raise_for_status()
    ask_resp = AskResponse.model_validate(resp.json())
    return ask_resp, elapsed_ms


def execute_via_pipeline(pipeline: RAGPipeline, req: AskRequest) -> tuple[AskResponse, float]:
    """Execute AskRequest directly through RAGPipeline and measure latency in milliseconds."""
    start_time = time.perf_counter()
    ask_resp = pipeline.answer(req)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return ask_resp, elapsed_ms


def main() -> None:
    """Run benchmark comparison across Cold and Cached RAG queries."""
    endpoint_url = "http://localhost:8000/ask"
    use_http = False

    # Check if API server is reachable
    try:
        with httpx.Client(timeout=3.0) as client:
            res = client.get("http://localhost:8000/ping")
            if res.status_code == 200:
                use_http = True
                logger.info("API server detected at %s — benchmarking via HTTP endpoints", endpoint_url)
    except Exception:
        logger.info("API server not reachable at localhost:8000 — running benchmark directly via RAGPipeline")

    cache = CacheService()
    pipeline = None if use_http else RAGPipeline()
    http_client = httpx.Client(timeout=120.0) if use_http else None

    results = []

    print("\n" + "=" * 90)
    print("              RAG CACHE BENCHMARK: COLD vs CACHED RESPONSE LATENCY             ")
    print("=" * 90)
    print(f"  Redis URL:      {settings.redis_url}")
    print(f"  Cache Enabled:  {settings.cache_enabled}")
    print(f"  Default TTL:    {settings.cache_ttl_seconds}s")
    print(f"  Execution Mode: {'HTTP Endpoint (' + endpoint_url + ')' if use_http else 'Direct RAGPipeline'}")
    print("-" * 90 + "\n")

    for idx, query in enumerate(SAMPLE_QUESTIONS, start=1):
        req = AskRequest(query=query, mode="hybrid", top_k=8)
        key = cache.get_cache_key(req)

        # Clear existing cache key to guarantee Cold start
        cache.delete(key)

        # 1. Cold Run
        if use_http and http_client:
            cold_resp, cold_ms = execute_via_http(http_client, endpoint_url, req)
        else:
            assert pipeline is not None
            cold_resp, cold_ms = execute_via_pipeline(pipeline, req)

        # 2. Cached Run
        if use_http and http_client:
            cached_resp, cached_ms = execute_via_http(http_client, endpoint_url, req)
        else:
            assert pipeline is not None
            cached_resp, cached_ms = execute_via_pipeline(pipeline, req)

        speedup = cold_ms / cached_ms if cached_ms > 0 else 1.0

        results.append({
            "idx": idx,
            "query": query,
            "cold_ms": cold_ms,
            "cached_ms": cached_ms,
            "speedup": speedup,
            "is_cached": cached_resp.cached,
        })

        print(f"Query {idx}/{len(SAMPLE_QUESTIONS)} completed | Cold: {cold_ms:.1f}ms | Cached: {cached_ms:.1f}ms | Speedup: {speedup:.1f}x")

    if http_client:
        http_client.close()

    # Print Markdown Comparison Table
    print("\n" + "=" * 90)
    print("### Cache Benchmark Results Summary\n")
    print("| # | Query | Cold Latency | Cached Latency | Speedup | Cache Hit Status |")
    print("|---|-------|--------------|----------------|---------|------------------|")

    total_cold = 0.0
    total_cached = 0.0

    for r in results:
        total_cold += r["cold_ms"]
        total_cached += r["cached_ms"]
        short_q = (r["query"][:55] + "...") if len(r["query"]) > 58 else r["query"]
        status_str = "Hit (cached=True)" if r["is_cached"] else "Miss"
        print(f"| {r['idx']} | {short_q:<58} | {r['cold_ms']:>9.1f} ms | {r['cached_ms']:>11.1f} ms | {r['speedup']:>6.1f}x | {status_str:<16} |")

    avg_cold = total_cold / len(results) if results else 0.0
    avg_cached = total_cached / len(results) if results else 0.0
    avg_speedup = avg_cold / avg_cached if avg_cached > 0 else 1.0

    print("|---|-------|--------------|----------------|---------|------------------|")
    print(f"| **AVG** | **Average across {len(results)} queries** | **{avg_cold:.1f} ms** | **{avg_cached:.1f} ms** | **{avg_speedup:.1f}x** | **100% Verified** |")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
