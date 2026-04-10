"""Tests for the recursive chunker."""

from __future__ import annotations

from medrag.ingestion.chunker import (
    chunk_paper,
    estimate_tokens,
    _recursive_split,
)
from medrag.models import Paper


def test_estimate_tokens_nonzero() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("hello world") >= 2


def test_recursive_split_short_text_returns_single_chunk() -> None:
    out = _recursive_split("short text", target_chars=100, overlap_chars=10)
    assert out == ["short text"]


def test_recursive_split_long_text_produces_multiple_chunks() -> None:
    long = "Sentence one. " * 200
    out = _recursive_split(long, target_chars=200, overlap_chars=20)
    assert len(out) > 1
    assert all(len(c) <= 400 for c in out)  # generous upper bound w/ overlap


def test_chunk_paper_abstract_only(sample_paper: Paper) -> None:
    chunks = chunk_paper(sample_paper)
    assert len(chunks) >= 1
    assert all(c.pmid == sample_paper.pmid for c in chunks)
    assert all(c.section == "abstract" for c in chunks)
    assert all(c.chunk_id.startswith(sample_paper.pmid) for c in chunks)


def test_chunk_paper_full_text_has_multiple_sections(
    sample_full_text_paper: Paper,
) -> None:
    chunks = chunk_paper(sample_full_text_paper)
    sections = {c.section for c in chunks}
    # Should detect at least the abstract plus one IMRaD section
    assert "abstract" in sections
    assert len(sections) >= 2


def test_chunk_ids_are_unique(sample_full_text_paper: Paper) -> None:
    chunks = chunk_paper(sample_full_text_paper)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
