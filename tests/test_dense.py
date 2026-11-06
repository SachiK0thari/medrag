"""Tests for the Pinecone client's write path. No network: the index is a stub."""

from __future__ import annotations

from medrag.models import Chunk, EmbeddedChunk
from medrag.retrieval.dense import DenseRetriever


class _FakeIndex:
    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.delete_calls: list[list[str]] = []

    def upsert(self, vectors: list[dict[str, object]]) -> None:
        for v in vectors:
            self.vectors[str(v["id"])] = v["values"]  # type: ignore[assignment]

    def delete(self, ids: list[str]) -> None:
        self.delete_calls.append(ids)
        for i in ids:
            self.vectors.pop(i, None)

    def describe_index_stats(self) -> dict[str, int]:
        return {"total_vector_count": len(self.vectors)}


def _retriever(index: _FakeIndex) -> DenseRetriever:
    retriever = DenseRetriever()
    retriever._index = index
    return retriever


def _embedded(n: int) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            chunk=Chunk(chunk_id=f"c{i}", pmid="1", text="t", chunk_index=i, n_tokens=1),
            embedding=[0.0, 1.0],
        )
        for i in range(n)
    ]


def test_upsert_then_delete_leaves_only_the_live_vectors() -> None:
    index = _FakeIndex()
    retriever = _retriever(index)
    assert retriever.upsert(_embedded(250)) == 250
    assert retriever.count_vectors() == 250

    assert retriever.delete([f"c{i}" for i in range(50)]) == 50
    assert retriever.count_vectors() == 200
    assert "c49" not in index.vectors
    assert "c50" in index.vectors


def test_delete_batches_stay_within_pinecones_limit() -> None:
    index = _FakeIndex()
    _retriever(index).delete([f"c{i}" for i in range(2500)])
    assert [len(call) for call in index.delete_calls] == [1000, 1000, 500]


def test_delete_nothing_makes_no_request() -> None:
    index = _FakeIndex()
    assert _retriever(index).delete([]) == 0
    assert index.delete_calls == []
