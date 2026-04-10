"""Tests for RRF fusion (the only retrieval logic that's pure-functional)."""

from __future__ import annotations

from medrag.models import (
    Chunk,
    Paper,
    PaperSource,
    RetrievalMethod,
    RetrievedChunk,
)
from medrag.retrieval.hybrid import reciprocal_rank_fusion


def _make_rc(chunk_id: str, rank: int, method: RetrievalMethod) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        pmid=chunk_id.split("-")[0],
        text="x",
        section="abstract",
        chunk_index=0,
        n_tokens=1,
    )
    paper = Paper(pmid=chunk.pmid, title="t", source=PaperSource.PUBMED_ABSTRACT)
    return RetrievedChunk(
        chunk=chunk, paper=paper, score=1.0 / rank, rank=rank, method=method
    )


def test_rrf_orders_by_combined_rank() -> None:
    bm25 = [
        _make_rc("p1-c1", 1, RetrievalMethod.BM25),
        _make_rc("p2-c1", 2, RetrievalMethod.BM25),
        _make_rc("p3-c1", 3, RetrievalMethod.BM25),
    ]
    dense = [
        _make_rc("p2-c1", 1, RetrievalMethod.DENSE),  # appears top in dense
        _make_rc("p1-c1", 2, RetrievalMethod.DENSE),
        _make_rc("p4-c1", 3, RetrievalMethod.DENSE),
    ]
    fused = reciprocal_rank_fusion([bm25, dense], k=60, top_k=10)
    fused_ids = [rc.chunk.chunk_id for rc in fused]
    # p1 and p2 both appear in both lists at top ranks -> should beat p3 and p4
    assert fused_ids[0] in {"p1-c1", "p2-c1"}
    assert fused_ids[1] in {"p1-c1", "p2-c1"}
    assert "p3-c1" in fused_ids
    assert "p4-c1" in fused_ids


def test_rrf_method_is_hybrid_rrf() -> None:
    bm25 = [_make_rc("p1-c1", 1, RetrievalMethod.BM25)]
    dense = [_make_rc("p1-c1", 1, RetrievalMethod.DENSE)]
    fused = reciprocal_rank_fusion([bm25, dense])
    assert fused[0].method is RetrievalMethod.HYBRID_RRF


def test_rrf_handles_empty_input() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
