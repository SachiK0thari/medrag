"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from medrag.models import (
    Answer,
    Citation,
    Chunk,
    Paper,
    PaperSource,
    RetrievalMethod,
    RetrievedChunk,
)


def test_paper_display_citation(sample_paper: Paper) -> None:
    assert sample_paper.display_citation == "Smith 2016, N Engl J Med"


def test_paper_has_full_text_false_for_abstract_only(sample_paper: Paper) -> None:
    assert sample_paper.has_full_text is False


def test_paper_has_full_text_true_for_real_body(sample_full_text_paper: Paper) -> None:
    assert sample_full_text_paper.has_full_text is True


def test_paper_is_frozen(sample_paper: Paper) -> None:
    with pytest.raises(ValidationError):
        sample_paper.title = "new title"  # type: ignore[misc]


def test_chunk_requires_non_negative_index() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="abc",
            pmid="123",
            text="hello",
            section="abstract",
            chunk_index=-1,
            n_tokens=1,
        )


def test_answer_confidence_bounded() -> None:
    base = {
        "question": "q",
        "text": "a [1]",
        "citations": [
            Citation(
                marker="[1]",
                pmid="1",
                chunk_id="c1",
                display_citation="Smith 2020",
                snippet="evidence",
            )
        ],
        "n_retrieval_calls": 1,
        "total_chunks_considered": 5,
    }
    Answer(**base, confidence=0.5)
    with pytest.raises(ValidationError):
        Answer(**base, confidence=1.5)


def test_retrieved_chunk_method_enum() -> None:
    chunk = Chunk(
        chunk_id="c1",
        pmid="1",
        text="hello",
        section="abstract",
        chunk_index=0,
        n_tokens=1,
    )
    paper = Paper(
        pmid="1",
        title="t",
        source=PaperSource.PUBMED_ABSTRACT,
    )
    rc = RetrievedChunk(
        chunk=chunk,
        paper=paper,
        score=0.9,
        rank=1,
        method=RetrievalMethod.HYBRID_RRF,
    )
    assert rc.method is RetrievalMethod.HYBRID_RRF
