"""Indexing pipeline script: re-chunk and re-embed papers from PostgreSQL into OpenSearch vector index.

Reads successfully parsed papers from Postgres, applies section-aware chunking, generates embeddings,
and bulk-indexes chunk documents into OpenSearch paper_chunks_v1.
Re-runnable and idempotent (deletes existing paper_id chunks before re-indexing).
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from src.database import get_session
from src.models.paper import Paper
from src.services.embeddings import get_embedding_service
from src.services.indexing import TextChunker
from src.services.opensearch import OpenSearchService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("index_chunks")


def process_and_index_chunks() -> None:
    """Chunk, embed, and index papers from Postgres into OpenSearch."""
    opensearch_service = OpenSearchService()
    embedding_service = get_embedding_service()
    chunker = TextChunker()

    logger.info("Step 1: Ensuring OpenSearch chunk index exists...")
    opensearch_service.create_chunk_index()

    logger.info("Step 2: Querying papers from PostgreSQL...")
    with get_session() as session:
        stmt = select(Paper).where(Paper.parse_status == "success")
        papers = list(session.scalars(stmt).all())
        logger.info("Found %d successfully parsed papers in PostgreSQL.", len(papers))

        if not papers:
            logger.info("No parsed papers found to index.")
            return

        total_chunks_indexed = 0
        total_chunks_skipped = 0

        for paper in papers:
            logger.info("Processing paper %s (%s) …", paper.arxiv_id, paper.title[:50])

            # Delete previous chunks for idempotency
            opensearch_service.delete_paper_chunks(paper.arxiv_id)

            # Generate section-aware chunks
            chunks = chunker.chunk_paper(paper)
            if not chunks:
                logger.info("No chunks produced for paper %s.", paper.arxiv_id)
                continue

            logger.info("Generated %d chunks for paper %s. Generating embeddings…", len(chunks), paper.arxiv_id)

            # Embed chunk texts
            texts = [c.text for c in chunks]
            embeddings = embedding_service.embed(texts)

            # Build chunk documents
            chunk_docs = []
            for chunk, emb in zip(chunks, embeddings):
                if emb is None:
                    total_chunks_skipped += 1
                    logger.warning("Skipping chunk %s due to embedding failure", chunk.chunk_id)
                    continue

                doc = {
                    "chunk_id": chunk.chunk_id,
                    "paper_id": paper.arxiv_id,
                    "section_name": chunk.section_name,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "embedding": emb,
                    "title": paper.title,
                    "authors": paper.authors or [],
                    "category": paper.category,
                    "published_date": paper.published_date.isoformat() if paper.published_date else None,
                    "pdf_url": paper.pdf_url,
                    "parse_status": paper.parse_status,
                }
                chunk_docs.append(doc)

            if chunk_docs:
                success_count, errors = opensearch_service.index_chunks(chunk_docs)
                total_chunks_indexed += success_count
                logger.info("Indexed %d chunks for paper %s", success_count, paper.arxiv_id)

        print(
            f"\n[OK] Chunk indexing complete:\n"
            f"  Total papers processed : {len(papers)}\n"
            f"  Total chunks indexed   : {total_chunks_indexed}\n"
            f"  Total chunks skipped   : {total_chunks_skipped}\n"
        )


if __name__ == "__main__":
    process_and_index_chunks()
