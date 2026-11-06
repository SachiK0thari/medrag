"""Reranker tests: fixed-shape batch planning and the choice of scoring backend.

No model is loaded and no network is used: the local scorer and the Pinecone
client are replaced with fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from medrag.config import get_settings
from medrag.models import Chunk, Paper, PaperSource, RetrievalMethod, RetrievedChunk
from medrag.retrieval import reranker as reranker_mod
from medrag.retrieval.reranker import (
    _BASE_BATCH,
    _BASE_LENGTH,
    _HOSTED_MAX_DOCS,
    Reranker,
    _bucket_length,
    _plan_batches,
)

_MAX = 8192


def test_bucket_length_rounds_up_to_a_doubling_ladder() -> None:
    assert _bucket_length(1, _MAX) == _BASE_LENGTH
    assert _bucket_length(_BASE_LENGTH, _MAX) == _BASE_LENGTH
    assert _bucket_length(_BASE_LENGTH + 1, _MAX) == 2 * _BASE_LENGTH
    assert _bucket_length(1070, _MAX) == 4 * _BASE_LENGTH


def test_bucket_length_is_capped_at_the_model_maximum() -> None:
    assert _bucket_length(50_000, _MAX) == _MAX
    # A cap that is not on the ladder still wins.
    assert _bucket_length(700, 600) == 600


def test_plan_covers_every_index_exactly_once() -> None:
    lengths = [300, 520, 90, 1070, 511, 512, 513, 2000]
    plan = _plan_batches(lengths, _MAX)
    seen = [i for _, _, indices in plan for i in indices]
    assert sorted(seen) == list(range(len(lengths)))


def test_plan_uses_a_bounded_set_of_shapes() -> None:
    # 500 pairs with arbitrary lengths must still map to a handful of shapes:
    # that bound is what keeps MPS driver memory from growing.
    lengths = [17 + (i * 37) % 1500 for i in range(500)]
    shapes = {(padded, batch_size) for padded, batch_size, _ in _plan_batches(lengths, _MAX)}
    assert shapes <= {
        (_BASE_LENGTH, _BASE_BATCH),
        (2 * _BASE_LENGTH, _BASE_BATCH // 2),
        (4 * _BASE_LENGTH, _BASE_BATCH // 4),
    }


def test_plan_batches_never_exceed_their_batch_size() -> None:
    plan = _plan_batches([100] * (2 * _BASE_BATCH + 3), _MAX)
    assert [len(indices) for _, _, indices in plan] == [_BASE_BATCH, _BASE_BATCH, 3]
    assert all(batch_size == _BASE_BATCH for _, batch_size, _ in plan)


def test_plan_shrinks_batch_size_as_length_grows() -> None:
    plan = _plan_batches([100, 1000, 8000], _MAX)
    by_length = {padded: batch_size for padded, batch_size, _ in plan}
    assert by_length[_BASE_LENGTH] == _BASE_BATCH
    assert by_length[2 * _BASE_LENGTH] == _BASE_BATCH // 2
    assert by_length[_MAX] == 1


def test_plan_of_nothing_is_empty() -> None:
    assert _plan_batches([], _MAX) == []


# --------------------------------------------------------------------------- #
# Scoring backends                                                             #
# --------------------------------------------------------------------------- #


def _candidates(n: int) -> list[RetrievedChunk]:
    paper = Paper(
        pmid="1",
        title="Paper",
        authors=["Ada Lovelace"],
        journal="J Test",
        year=2024,
        source=PaperSource.PUBMED_ABSTRACT,
    )
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"1-{i}",
                pmid="1",
                text=f"text {i}",
                section="abstract",
                chunk_index=i,
                n_tokens=2,
            ),
            paper=paper,
            score=0.0,
            rank=i + 1,
            method=RetrievalMethod.HYBRID_RRF,
        )
        for i in range(n)
    ]


def _score_of(text: str) -> float:
    """A distinct score per text: 'text 7' scores 0.007, so higher index ranks first."""
    return int(text.split()[1]) / 1000


class FakeInference:
    """Stands in for `Pinecone().inference`; answers best-first like the real API."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.requests: list[dict[str, object]] = []
        self._fail_with = fail_with

    def rerank(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        if self._fail_with is not None:
            raise self._fail_with
        documents = kwargs["documents"]
        assert isinstance(documents, list)
        rows = [SimpleNamespace(index=i, score=_score_of(t)) for i, t in enumerate(documents)]
        return SimpleNamespace(data=sorted(rows, key=lambda r: r.score, reverse=True))


@pytest.fixture
def backends(monkeypatch: pytest.MonkeyPatch):
    """Install fakes for both backends; returns a builder for a Reranker on one of them."""
    local_calls: list[int] = []

    def fake_local(query: str, texts: list[str]) -> list[float]:
        local_calls.append(len(texts))
        return [1 - _score_of(t) for t in texts]  # the reverse of the hosted order

    monkeypatch.setattr(reranker_mod, "_score_local", fake_local)

    def build(backend: str, fail_with: Exception | None = None) -> tuple[Reranker, FakeInference]:
        inference = FakeInference(fail_with)
        monkeypatch.setattr(
            reranker_mod, "_pinecone_client", lambda: SimpleNamespace(inference=inference)
        )
        reranker = Reranker()
        reranker._settings = get_settings().model_copy(update={"reranker_backend": backend})
        return reranker, inference

    return build, local_calls


def test_local_backend_is_the_default_and_never_calls_pinecone(backends) -> None:
    build, local_calls = backends
    assert get_settings().reranker_backend == "local"
    reranker, inference = build("local")

    ranked = reranker.rerank("q", _candidates(3), top_k=3)

    assert [rc.chunk.chunk_id for rc in ranked] == ["1-0", "1-1", "1-2"]
    assert local_calls == [3]
    assert inference.requests == []


def test_hosted_backend_ranks_by_the_hosted_scores(backends) -> None:
    build, local_calls = backends
    reranker, inference = build("pinecone")

    ranked = reranker.rerank("q", _candidates(5), top_k=2)

    assert [rc.chunk.chunk_id for rc in ranked] == ["1-4", "1-3"]
    assert [rc.score for rc in ranked] == [0.004, 0.003]
    assert [rc.rank for rc in ranked] == [1, 2]
    assert all(rc.method is RetrievalMethod.HYBRID_RERANKED for rc in ranked)
    assert local_calls == []
    (request,) = inference.requests
    assert request["model"] == "bge-reranker-v2-m3"  # no Hugging Face organisation prefix
    assert request["query"] == "q"
    assert request["top_n"] == 5  # every pair scored, whatever top_k is


def test_hosted_backend_splits_requests_at_the_document_limit(backends) -> None:
    build, _ = backends
    reranker, inference = build("pinecone")
    n = 2 * _HOSTED_MAX_DOCS + 30

    ranked = reranker.rerank("q", _candidates(n), top_k=n)

    assert [len(r["documents"]) for r in inference.requests] == [  # type: ignore[arg-type]
        _HOSTED_MAX_DOCS,
        _HOSTED_MAX_DOCS,
        30,
    ]
    # Scores land on the right chunk across batches: best is the last candidate.
    assert [rc.chunk.chunk_id for rc in ranked] == [f"1-{i}" for i in reversed(range(n))]


def test_hosted_failure_falls_back_to_the_local_model(backends, caplog) -> None:
    build, local_calls = backends
    reranker, inference = build("pinecone", fail_with=RuntimeError("429 quota exceeded"))

    with caplog.at_level("WARNING", logger="medrag.retrieval.reranker"):
        ranked = reranker.rerank("q", _candidates(3), top_k=3)

    assert len(inference.requests) == 1
    assert local_calls == [3]
    assert [rc.chunk.chunk_id for rc in ranked] == ["1-0", "1-1", "1-2"]  # the local order
    assert "429 quota exceeded" in caplog.text
    assert "scoring 3 pairs locally" in caplog.text
