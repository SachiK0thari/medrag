"""Cross-encoder reranking with bge-reranker-v2-m3."""

from __future__ import annotations

from medrag.models import RetrievalMethod, RetrievedChunk

# TODO(saturday): wrap sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")
# rerank(query, chunks, top_k) -> re-sorted RetrievedChunk list with method=HYBRID_RERANKED


class Reranker:
    def __init__(self) -> None:
        pass

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        raise NotImplementedError("implement Saturday evening")
