"""BM25 sparse retrieval over the in-memory chunk corpus.

Loaded once from SQLite at process start; not meant to scale beyond
a few hundred thousand chunks. For this project (~5–10k chunks) it's trivial.
"""

from __future__ import annotations

from medrag.models import Chunk, RetrievedChunk

# TODO(saturday): implement rank_bm25.BM25Okapi over tokenized chunk texts.
# Tokenization: lowercase + simple regex split, drop stopwords.
# search(query, top_k) -> list[RetrievedChunk] with method=BM25.


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError("implement Saturday evening")
