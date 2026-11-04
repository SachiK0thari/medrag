"""Ingest GLP-1 / cardiovascular outcome papers from PubMed/PMC into Pinecone.

Pipeline:
    esearch -> efetch -> (TODO: PMC full text) -> chunk -> embed
    -> SQLite persistence -> Pinecone upsert

Usage:
    python scripts/ingest.py [--max-papers 500] [--query "..."] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

from medrag.config import get_settings
from medrag.ingestion.chunker import chunk_paper
from medrag.ingestion.embedder import embed_chunks
from medrag.ingestion.pmc import enrich_with_full_text
from medrag.ingestion.pubmed import PubMedClient
from medrag.retrieval.dense import DenseRetriever
from medrag.store import Store


async def main(query: str, max_papers: int, dry_run: bool) -> None:
    settings = get_settings()
    print(
        f"Settings: index={settings.pinecone_index_name} "
        f"model={settings.embedding_model} dry_run={dry_run}"
    )
    print(f"Query: {query!r}")
    t0 = time.perf_counter()

    async with PubMedClient() as pubmed:
        print("1. esearch ...")
        pmids = await pubmed.search(query, max_results=max_papers)
        print(f"   found {len(pmids)} PMIDs")

        print("2. efetch metadata ...")
        papers = await pubmed.fetch_in_batches(pmids, batch_size=100)
        print(f"   fetched {len(papers)} papers")

    if not papers:
        print("No papers fetched — nothing to do.")
        return

    n_with_pmcid = sum(1 for p in papers if p.pmcid)
    print(f"2b. PMC full-text enrichment ({n_with_pmcid} candidates) ...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        papers = await enrich_with_full_text(papers, client)
    n_full = sum(1 for p in papers if p.has_full_text)
    print(f"   enriched {n_full} papers with PMC body text")

    print("3. chunk ...")
    all_chunks = [c for p in papers for c in chunk_paper(p)]
    print(f"   produced {len(all_chunks)} chunks")

    print("4. embed ...")
    embedded = embed_chunks(all_chunks)
    print(f"   embedded {len(embedded)} chunks")

    print("5. persist to SQLite ...")
    with Store() as store:
        store.save_papers(papers)
        store.save_chunks(all_chunks)
        print(
            f"   sqlite: {store.count_papers()} papers / "
            f"{store.count_chunks()} chunks at {settings.sqlite_path}"
        )

    if dry_run:
        print("6. Pinecone upsert skipped (--dry-run)")
    else:
        print("6. ensure Pinecone index ...")
        dense = DenseRetriever()
        dense.ensure_index()
        print("   upserting vectors ...")
        n_upserted = dense.upsert(embedded)
        print(f"   upserted {n_upserted} vectors")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Pinecone upsert (still writes SQLite).",
    )
    args = parser.parse_args()
    settings = get_settings()
    asyncio.run(
        main(
            query=args.query or settings.ingest_topic_query,
            max_papers=args.max_papers or settings.ingest_max_papers,
            dry_run=args.dry_run,
        )
    )
