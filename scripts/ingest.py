"""Ingest GLP-1 / cardiovascular outcome papers from PubMed/PMC into Pinecone.

Two modes:

    python scripts/ingest.py               # build a corpus from a PubMed search
    python scripts/ingest.py --in-place    # rebuild the corpus already stored

Both run the same pipeline over their papers:

    PMC full text -> chunk -> embed -> SQLite -> Pinecone

The default mode gets its papers from esearch + efetch, newest first, so
running it twice months apart gives two different corpora. `--in-place` takes
the papers already in SQLite instead and searches for nothing: it asks PubMed
only for each paper's current PMC ID, then fetches full text and re-chunks.
Use it after a change to the PMC parser or the chunker, or to pick up papers
deposited in PMC since the first run, when the paper set itself has to stay
fixed (the eval's gold labels are PMIDs).

In either mode a paper's chunks are replaced as a set, so chunks the chunker
no longer produces are removed from SQLite and from Pinecone.

Usage:
    python scripts/ingest.py [--in-place] [--max-papers 500] [--query "..."] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

from medrag.config import get_settings
from medrag.ingestion.chunker import chunk_paper
from medrag.ingestion.embedder import count_tokens, embed_chunks, max_input_tokens
from medrag.ingestion.pmc import enrich_with_full_text
from medrag.ingestion.pubmed import PubMedClient, with_current_pmcids
from medrag.models import Paper
from medrag.retrieval.dense import DenseRetriever
from medrag.store import Store


async def search_papers(query: str, max_papers: int) -> list[Paper]:
    """Default mode: the newest `max_papers` PubMed records matching `query`."""
    print(f"Query: {query!r}")
    async with PubMedClient() as pubmed:
        print("1. esearch ...")
        pmids = await pubmed.search(query, max_results=max_papers)
        print(f"   found {len(pmids)} PMIDs")

        print("2. efetch metadata ...")
        papers = await pubmed.fetch_in_batches(pmids, batch_size=100)
        print(f"   fetched {len(papers)} papers")
    return papers


async def stored_papers() -> list[Paper]:
    """--in-place mode: the stored papers, with the PMC ID PubMed lists today."""
    print("1. load stored papers ...")
    with Store() as store:
        stored = store.load_all_papers()
    print(f"   {len(stored)} papers, {sum(1 for p in stored if p.pmcid)} with a PMC ID")
    if not stored:
        return []

    print("2. refresh PMC IDs from PubMed ...")
    async with PubMedClient() as pubmed:
        fresh = await pubmed.fetch_in_batches([p.pmid for p in stored], batch_size=100)
    papers = with_current_pmcids(stored, fresh)
    before = {p.pmid: p.pmcid for p in stored}
    gained = sum(1 for p in papers if p.pmcid and not before[p.pmid])
    dropped = sum(1 for p in papers if before[p.pmid] and not p.pmcid)
    replaced = sum(1 for p in papers if p.pmcid and before[p.pmid] not in (None, p.pmcid))
    print(
        f"   {sum(1 for p in papers if p.pmcid)} with a PMC ID now "
        f"({gained} gained, {dropped} dropped, {replaced} replaced; "
        f"{len(stored) - len(fresh)} not returned by PubMed)"
    )
    return papers


async def main(query: str, max_papers: int, in_place: bool, dry_run: bool) -> None:
    settings = get_settings()
    print(
        f"Settings: index={settings.pinecone_index_name} model={settings.embedding_model} "
        f"mode={'in-place' if in_place else 'search'} dry_run={dry_run}"
    )
    t0 = time.perf_counter()

    papers = await (stored_papers() if in_place else search_papers(query, max_papers))
    if not papers:
        print("No papers — nothing to do.")
        return

    n_with_pmcid = sum(1 for p in papers if p.pmcid)
    print(f"3. PMC full text ({n_with_pmcid} candidates) ...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        papers = await enrich_with_full_text(papers, client)
    n_full = sum(1 for p in papers if p.has_full_text)
    print(f"   {n_full} papers have PMC body text; {len(papers) - n_full} are abstract only")

    print("4. chunk ...")
    limit = max_input_tokens()
    all_chunks = [
        c for p in papers for c in chunk_paper(p, count_tokens=count_tokens, max_tokens=limit)
    ]
    longest = max((c.n_tokens for c in all_chunks), default=0)
    print(f"   {len(all_chunks)} chunks; longest is {longest} tokens (embedder limit {limit})")

    if dry_run:
        print("Stopping before embedding (--dry-run): nothing was written.")
        return

    print("5. embed ...")
    embedded = embed_chunks(all_chunks)
    print(f"   embedded {len(embedded)} chunks")

    print("6. persist to SQLite ...")
    with Store() as store:
        store.save_papers(papers)
        stale_ids = store.replace_chunks([p.pmid for p in papers], all_chunks)
        print(
            f"   sqlite: {store.count_papers()} papers / {store.count_chunks()} chunks "
            f"at {settings.sqlite_path}; {len(stale_ids)} old chunks removed"
        )

    print("7. Pinecone ...")
    dense = DenseRetriever()
    dense.ensure_index()
    n_upserted = dense.upsert(embedded)
    n_deleted = dense.delete(stale_ids)
    print(f"   upserted {n_upserted} vectors, deleted {n_deleted} stale ones")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rebuild the stored corpus (full text, chunks, vectors) without searching PubMed.",
    )
    parser.add_argument("--query", default=None, help="PubMed query (default mode only).")
    parser.add_argument("--max-papers", type=int, default=None, help="Default mode only.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and chunk, print the counts, and write nothing.",
    )
    args = parser.parse_args()
    if args.in_place and (args.query or args.max_papers):
        parser.error("--in-place rebuilds the stored papers; it takes no --query or --max-papers")
    settings = get_settings()
    asyncio.run(
        main(
            query=args.query or settings.ingest_topic_query,
            max_papers=args.max_papers or settings.ingest_max_papers,
            in_place=args.in_place,
            dry_run=args.dry_run,
        )
    )
