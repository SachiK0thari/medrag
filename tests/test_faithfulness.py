"""compute_faithfulness retries once when RAGAS reports a truncated reply."""

from __future__ import annotations

from typing import Any

import pytest

# ragas is in the `eval` extra, which CI does not install.
pytest.importorskip("ragas")

from ragas.exceptions import LLMDidNotFinishException  # noqa: E402

from medrag.eval import faithfulness  # noqa: E402
from medrag.models import Chunk, Paper, PaperSource, RetrievalMethod, RetrievedChunk  # noqa: E402


def _contexts() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="1-0",
                pmid="1",
                text="Semaglutide cut MACE by 20%.",
                chunk_index=0,
                n_tokens=8,
            ),
            paper=Paper(pmid="1", title="T", source=PaperSource.PUBMED_ABSTRACT),
            score=1.0,
            rank=1,
            method=RetrievalMethod.HYBRID_RERANKED,
        )
    ]


class _FakeScorer:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.llm: object = None

    async def single_turn_ascore(self, sample: object) -> float:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return float(outcome)


@pytest.fixture
def scorer(monkeypatch: pytest.MonkeyPatch) -> _FakeScorer:
    fake = _FakeScorer([])
    monkeypatch.setattr(faithfulness, "Faithfulness", lambda: fake)
    monkeypatch.setattr(faithfulness, "_get_llm", lambda: object())
    return fake


def _score() -> float | None:
    return faithfulness.compute_faithfulness("q?", "Semaglutide cut MACE.", _contexts())


def test_retries_once_after_truncation(scorer: _FakeScorer) -> None:
    scorer.outcomes = [LLMDidNotFinishException(), 0.75]
    assert _score() == 0.75
    assert scorer.calls == 2


def test_gives_up_after_second_truncation(scorer: _FakeScorer) -> None:
    scorer.outcomes = [LLMDidNotFinishException(), LLMDidNotFinishException()]
    assert _score() is None
    assert scorer.calls == 2


def test_other_errors_are_not_retried(scorer: _FakeScorer) -> None:
    scorer.outcomes = [ValueError("bad json"), 0.9]
    assert _score() is None
    assert scorer.calls == 1


def test_empty_answer_is_skipped(scorer: _FakeScorer) -> None:
    assert faithfulness.compute_faithfulness("q?", "  ", _contexts()) is None
    assert scorer.calls == 0
