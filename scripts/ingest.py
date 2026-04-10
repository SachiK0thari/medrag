"""Ingest GLP-1 / cardiovascular outcome papers from PubMed/PMC into Pinecone.

Usage:
    python scripts/ingest.py [--max-papers 500] [--query "..."]
"""

from __future__ import annotations

import argparse
import asyncio

from medrag.config import get_settings
from medrag.ingestion.chunker import chunk_paper
from medrag.ingestion.embedder import embed_chunks
from medrag.ingestion.pubmed import PubMedClient


async def main(query: str, max_papers: int) -> None:
    settings = get_settings()
    print(f"Settings: index={settings.pinecone_index_name} model={settings.embedding_model}")
    print(f"Query: {query!r}")

    async with PubMedClient() as pubmed:
        print("1. esearch ...")
        pmids = await pubmed.search(query, max_results=max_papers)
        print(f"   found {len(pmids)} PMIDs")

        print("2. efetch metadata ...")
        papers = await pubmed.fetch_in_batches(pmids, batch_size=100)
        print(f"   fetched {len(papers)} papers")

    # TODO(saturday): enrich with PMC full text where pmcid is set
    print("3. chunk ...")
    all_chunks = [c for p in papers for c in chunk_paper(p)]
    print(f"   produced {len(all_chunks)} chunks")

    print("4. embed ...")
    embedded = embed_chunks(all_chunks)
    print(f"   embedded {len(embedded)} chunks")

    # TODO(saturday): persist papers + chunks to SQLite, upsert vectors to Pinecone
    print("5. upsert -> Pinecone (TODO)")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    args = parser.parse_args()
    settings = get_settings()
    asyncio.run(
        main(
            query=args.query or settings.ingest_topic_query,
            max_papers=args.max_papers or settings.ingest_max_papers,
        )
    )
