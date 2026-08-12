"""Batch Evaluation Script for Agentic RAG Pipeline.

Evaluates latency (p50/p95), retrieval count, rewrite attempts, grader verdicts,
and normalized relevance scores across a benchmark of domain-specific and off-topic queries.
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.rag import AskRequest
from src.services.agents.agentic_rag import run_agentic_rag

logging.basicConfig(level=logging.WARNING)

BENCHMARK_QUERIES = [
    # Domain-specific technical queries (Known-Good)
    "How does SSM state injection work in Structured Memory for Edge Language Models?",
    "What is the PRECOG mechanism for edge language models?",
    "How does CMuon accelerate diffusion transformer training?",
    "What is Qwen-CUA and how does native computer use work?",
    "What is LiveMem architecture for LLM inference?",
    "How does CTRAG automate compliance checking using LLMs?",
    "What is UEmbed unified sparse and dense multimodal embeddings?",
    "How does GradCuit enable robust test-time latent reasoning?",
    "What is AURORA-LM continuous-latent diffusion language modeling?",
    "How does DyFrDet perform small object detection via dynamic frequency suppression?",
    "What is SWE-Touch coding agent benchmark?",
    "How does RoMeRL balance feedback coverage and memory reward trap?",
    # Off-topic / Non-existent queries (Known-Bad)
    "What is the capital of France?",
    "How do you bake a chocolate chip cake?",
    "Who won the 1998 FIFA World Cup final?",
]


def run_batch_evaluation():
    """Run batch evaluation across benchmark queries and print stats."""
    results = []
    latencies = []

    print("=" * 80)
    print("RUNNING AGENTIC RAG BATCH EVALUATION BENCHMARK (15 QUERIES)")
    print("=" * 80)

    for idx, query in enumerate(BENCHMARK_QUERIES, 1):
        req = AskRequest(query=query)
        t0 = time.perf_counter()
        resp = run_agentic_rag(req)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        top_score = resp.sources[0].relevance_score if resp.sources else 0.0
        distinct_papers = len(set(s.paper_id for s in resp.sources))
        grade_decisions = [s.decision for s in resp.reasoning_steps if s.node == "grade"]
        final_grade = grade_decisions[-1] if grade_decisions else "none"

        is_off_topic = idx >= 13
        has_disclaimer = "Low Confidence Notice" in resp.answer or "I don't have enough information" in resp.answer

        results.append(
            {
                "query": query[:60],
                "latency_ms": elapsed_ms,
                "retrieved": resp.retrieved_chunk_count,
                "used": resp.used_chunk_count,
                "papers": distinct_papers,
                "rewrites": resp.rewrite_count,
                "grade": final_grade,
                "top_score": top_score,
                "disclaimer": has_disclaimer,
            }
        )

        print(
            f"[{idx:02d}/15] Latency: {elapsed_ms:6.0f}ms | Grade: {final_grade:6s} | "
            f"Rewrites: {resp.rewrite_count} | Chunks: {resp.used_chunk_count}/{resp.retrieved_chunk_count} | "
            f"TopScore: {top_score:.4f} | LowConf: {has_disclaimer} | Q: {query[:50]}"
        )

    # Compute p50 / p95 latency
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print("\n" + "=" * 80)
    print("BATCH EVALUATION SUMMARY STATS")
    print("=" * 80)
    print(f"Total Queries Evaluated : {len(BENCHMARK_QUERIES)}")
    print(f"p50 Latency            : {p50:.1f} ms")
    print(f"p95 Latency            : {p95:.1f} ms")
    print(
        f"Fast-path 1-round count : {sum(1 for r in results if r['rewrites'] == 0)} / {len(BENCHMARK_QUERIES)}"
    )
    print(
        f"Off-topic low-conf count: {sum(1 for r in results[12:] if r['disclaimer'])} / 3 off-topic queries"
    )
    print("=" * 80)


if __name__ == "__main__":
    run_batch_evaluation()
