"""End-to-End Live Demonstration of New Paper Ingestion & Agentic RAG QA."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import SessionLocal
from src.models.paper import Paper
from src.schemas.rag import AskRequest
from src.services.agents.agentic_rag import run_agentic_rag
from src.services.arxiv.client import ArxivClient
from src.services.embeddings import get_embedding_service
from src.services.indexing.text_chunker import TextChunker
from src.services.opensearch.service import OpenSearchService
from src.services.pdf_parser import PdfParserService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("demo_end_to_end")


def main():
    target_arxiv_id = "2403.09611"
    print("=" * 100)
    print(f"STARTING END-TO-END DEMONSTRATION FOR NEW PAPER: {target_arxiv_id}")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # STEP 1: Fetch paper metadata & parse PDF from arXiv
    # -------------------------------------------------------------------------
    print("\n[STEP 1/5] Fetching metadata & downloading PDF from arXiv...")
    arxiv_client = ArxivClient()
    papers = arxiv_client.search(f"id:{target_arxiv_id}", max_results=1)
    if not papers:
        raise RuntimeError(f"Could not find paper {target_arxiv_id} on arXiv")

    meta = papers[0]
    print(f"   -> ArXiv ID  : {meta.arxiv_id}")
    print(f"   -> Title     : {meta.title}")
    print(f"   -> Authors   : {', '.join(meta.authors[:3])} et al.")
    print(f"   -> Category  : {meta.category}")
    print(f"   -> PDF URL   : {meta.pdf_url}")

    pdf_parser = PdfParserService()
    pdf_res = pdf_parser.parse(meta.pdf_url, arxiv_id=meta.arxiv_id)
    if not pdf_res:
        raise RuntimeError(f"Failed to download/parse PDF from {meta.pdf_url}")
    print(f"   -> PDF Status: Success=True | Chars={len(pdf_res.full_text)}")
    print(f"   -> Sections  : Extracted {len(pdf_res.sections)} structured section headers")

    # -------------------------------------------------------------------------
    # STEP 2: Ingest into PostgreSQL Database
    # -------------------------------------------------------------------------
    print("\n[STEP 2/5] Storing paper record in PostgreSQL...")
    db = SessionLocal()
    try:
        existing = db.query(Paper).filter(Paper.arxiv_id == meta.arxiv_id).first()
        if existing:
            print("   -> Paper already exists in DB — deleting previous record for fresh demonstration...")
            db.delete(existing)
            db.commit()

        paper_obj = Paper(
            arxiv_id=meta.arxiv_id,
            title=meta.title,
            authors=meta.authors,
            abstract=meta.abstract,
            pdf_url=meta.pdf_url,
            published_date=meta.published_date,
            category=meta.category,
            full_text=pdf_res.full_text,
            sections=pdf_res.sections,
            parse_status="success",
        )
        db.add(paper_obj)
        db.commit()
        db.refresh(paper_obj)
        print(f"   -> Successfully inserted Paper ID={paper_obj.id} (arXiv: {paper_obj.arxiv_id}) into PostgreSQL!")
    finally:
        db.close()

    # -------------------------------------------------------------------------
    # STEP 3: Chunking & OpenSearch Vector Indexing
    # -------------------------------------------------------------------------
    print("\n[STEP 3/5] Chunking paper text and indexing into OpenSearch...")
    chunker = TextChunker()
    chunks = chunker.chunk_paper(paper_obj)
    print(f"   -> Generated {len(chunks)} chunks for paper '{paper_obj.title[:50]}...'")

    embedding_service = get_embedding_service()
    opensearch_service = OpenSearchService()

    # Generate embeddings in batches of 16
    chunk_texts = [c.text for c in chunks]
    embeddings = []
    batch_size = 16
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i : i + batch_size]
        batch_embs = embedding_service.embed(batch)
        embeddings.extend(batch_embs)

    # Attach embeddings to chunk models
    for c, emb in zip(chunks, embeddings):
        c.embedding = emb

    # Bulk index to OpenSearch
    indexed_count = opensearch_service.bulk_index_chunks(chunks)
    print(f"   -> Successfully indexed {indexed_count} vector chunk documents into OpenSearch 'paper_chunks_v1'!")

    # -------------------------------------------------------------------------
    # STEP 4: Query the Agentic RAG System
    # -------------------------------------------------------------------------
    test_query = "What are the key architectural decisions and pre-training insights of MM1 multimodal LLM?"
    print(f"\n[STEP 4/5] Executing Agentic RAG graph for query:\n   '{test_query}'")

    t0 = time.perf_counter()
    rag_response = run_agentic_rag(AskRequest(query=test_query))
    took_ms = (time.perf_counter() - t0) * 1000

    # -------------------------------------------------------------------------
    # STEP 5: Display Final Answer & Citation Verification
    # -------------------------------------------------------------------------
    print("\n[STEP 5/5] Final RAG System Output & Citation Verification:")
    print("=" * 100)
    print(f"Answer:\n{rag_response.answer}\n")
    print("-" * 100)
    print(f"Metrics & Reasoning Graph:")
    print(f"   -> Total Latency     : {took_ms:.1f} ms")
    print(f"   -> Guardrail Rejected: {rag_response.rejected}")
    print(f"   -> Rewrite Count     : {rag_response.rewrite_count}")
    print(f"   -> Retrieved Chunks  : {rag_response.retrieved_chunk_count}")
    print(f"   -> Used Sources      : {rag_response.used_chunk_count}")

    print("\n   Reasoning Graph Trajectory:")
    for step in rag_response.reasoning_steps:
        print(f"      Node [{step.node:10s}] -> Decision: {step.decision:15s} | Detail: {step.detail}")

    print("\n   Cited Sources Details:")
    for idx, s in enumerate(rag_response.sources, 1):
        print(f"      #{idx}: Score={s.score:.4f} | Section='{s.section_name}' | Paper='{s.title}'")
        print(f"         Snippet: {s.text[:180]}...")

    print("=" * 100)
    print("END-TO-END DEMONSTRATION COMPLETE SUCCESSFULLY!")
    print("=" * 100)


if __name__ == "__main__":
    main()
