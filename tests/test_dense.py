"""Tests for the Pinecone client. No network: the index is a stub, embeddings are faked."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from medrag.models import Chunk, EmbeddedChunk, Paper, RetrievalMethod
from medrag.retrieval import dense
from medrag.retrieval.dense import DenseRetriever
from medrag.store import Store


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


# ---- query path ----------------------------------------------------------- #


class _QueryIndex:
    """Answers `query` from a canned match list, after `failures` dropped connections."""

    def __init__(self, matches: list[dict[str, object]], *, failures: int = 0) -> None:
        self.matches = matches
        self.failures = failures
        self.calls: list[dict[str, object]] = []

    def query(self, **kwargs: object) -> dict[str, object]:
        from urllib3.exceptions import ProtocolError

        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise ProtocolError("connection reset")
        return {"matches": self.matches}


@pytest.fixture
def store(tmp_path, sample_paper: Paper) -> Iterator[Store]:
    with Store(tmp_path / "medrag.db") as s:
        s.save_papers([sample_paper.model_copy(update={"pmid": p}) for p in ("1", "2")])
        s.save_chunks(
            [
                Chunk(chunk_id="a", pmid="1", text="alpha", chunk_index=0, n_tokens=1),
                Chunk(chunk_id="b", pmid="2", text="beta", chunk_index=0, n_tokens=1),
                Chunk(chunk_id="orphan", pmid="9", text="no paper", chunk_index=0, n_tokens=1),
            ]
        )
        yield s


@pytest.fixture(autouse=True)
def _no_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dense, "embed_texts", lambda texts: [[0.1, 0.2]] * len(texts))


def _searcher(store: Store, index: _QueryIndex) -> DenseRetriever:
    retriever = DenseRetriever(store=store)
    retriever._index = index
    return retriever


def test_search_hydrates_matches_from_the_store_in_pinecone_order(store: Store) -> None:
    index = _QueryIndex([{"id": "b", "score": 0.9}, {"id": "a", "score": 0.7}])
    results = _searcher(store, index).search("q", top_k=2)

    assert [(r.chunk.chunk_id, r.paper.pmid, r.rank) for r in results] == [
        ("b", "2", 1),
        ("a", "1", 2),
    ]
    assert [r.score for r in results] == [0.9, 0.7]
    assert all(r.method is RetrievalMethod.DENSE for r in results)
    assert index.calls == [
        {"vector": [0.1, 0.2], "top_k": 2, "include_values": False, "include_metadata": False}
    ]


def test_stale_vectors_and_orphan_chunks_are_skipped(store: Store) -> None:
    """A vector whose chunk was re-chunked away, or whose paper is gone, is dropped."""
    index = _QueryIndex(
        [{"id": "gone", "score": 0.95}, {"id": "orphan", "score": 0.9}, {"id": "a", "score": 0.5}]
    )
    results = _searcher(store, index).search("q", top_k=3)
    assert [(r.chunk.chunk_id, r.rank) for r in results] == [("a", 1)]


def test_no_matches_returns_nothing(store: Store) -> None:
    assert _searcher(store, _QueryIndex([])).search("q", top_k=5) == []


def test_dropped_connections_are_retried(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    index = _QueryIndex([{"id": "a", "score": 0.5}], failures=2)
    assert len(_searcher(store, index).search("q", top_k=1)) == 1
    assert len(index.calls) == 3


def test_search_without_a_store_says_how_to_fix_it() -> None:
    with pytest.raises(RuntimeError, match="needs a Store"):
        DenseRetriever().search("q", top_k=1)
