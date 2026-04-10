"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

RRF score for doc d = Σ over retrievers r of  1 / (k + rank_r(d))
where k is a constant (60 in the original RRF paper). Simple, parameter-free,
and consistently competitive with learned fusion methods.
"""

from __future__ import annotations

from collections import defaultdict

from medrag.config import get_settings
from medrag.models import RetrievalMethod, RetrievedChunk
from medrag.retrieval.bm25 import BM25Retriever
from medrag.retrieval.dense import DenseRetriever


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    k: int = 60,
    top_k: int = 30,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists into one via RRF."""
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, RetrievedChunk] = {}

    for ranking in rankings:
        for rc in ranking:
            cid = rc.chunk.chunk_id
            scores[cid] += 1.0 / (k + rc.rank)
            # Keep the highest-ranked instance for metadata
            if cid not in by_id or rc.rank < by_id[cid].rank:
                by_id[cid] = rc

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(
            chunk=by_id[cid].chunk,
            paper=by_id[cid].paper,
            score=score,
            rank=i + 1,
            method=RetrievalMethod.HYBRID_RRF,
        )
        for i, (cid, score) in enumerate(fused)
    ]


class HybridRetriever:
    """Runs BM25 and dense retrieval in parallel and fuses with RRF."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._settings = get_settings()

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        sparse_results = self._bm25.search(query, self._settings.bm25_top_k)
        dense_results = self._dense.search(query, self._settings.dense_top_k)
        return reciprocal_rank_fusion(
            [sparse_results, dense_results],
            k=self._settings.rrf_k,
            top_k=top_k or self._settings.dense_top_k,
        )
