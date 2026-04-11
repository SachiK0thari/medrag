"""BM25 sparse retrieval over the in-memory chunk corpus.

Loaded once from SQLite at process start; not meant to scale beyond
a few hundred thousand chunks. For this project (~5-10k chunks) it's trivial
and keeps the sparse index latency dominated by tokenization, not IO.

Hyphenated tokens (GLP-1, SUSTAIN-6, liraglutide) are preserved on purpose —
medical terminology is the entire point of the sparse channel.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from medrag.models import Chunk, Paper, RetrievalMethod, RetrievedChunk
from medrag.store import Store

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "these",
        "those",
        "to",
        "was",
        "were",
        "will",
        "with",
        "we",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase + regex word split + small stopword drop."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    """In-memory BM25 over the full chunk corpus.

    Papers are held alongside chunks so a search result can be hydrated into
    a `RetrievedChunk` without a second DB hop per hit.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        papers_by_pmid: dict[str, Paper],
    ) -> None:
        self._chunks = chunks
        self._papers = papers_by_pmid
        tokenized = [tokenize(c.text) for c in chunks]
        self._bm25: BM25Okapi | None = BM25Okapi(tokenized) if tokenized else None

    @classmethod
    def from_store(cls, store: Store) -> BM25Retriever:
        """Load the full corpus from SQLite and build the index."""
        chunks = store.load_all_chunks()
        papers = {p.pmid: p for p in store.load_all_papers()}
        return cls(chunks, papers)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[RetrievedChunk] = []
        for idx in ranked[:top_k]:
            if scores[idx] <= 0:
                break
            chunk = self._chunks[idx]
            paper = self._papers.get(chunk.pmid)
            if paper is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    paper=paper,
                    score=float(scores[idx]),
                    rank=len(results) + 1,
                    method=RetrievalMethod.BM25,
                )
            )
        return results
