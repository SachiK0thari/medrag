"""Tests for the recursive chunker."""

from __future__ import annotations

import re

from medrag.config import get_settings
from medrag.ingestion.chunker import (
    _recursive_split,
    _split_by_section,
    chunk_paper,
    estimate_tokens,
)
from medrag.models import Paper

# One paragraph, far longer than a chunk, with no line break in it.
_LONG_PARAGRAPH = " ".join(
    f"In subgroup {i} the hazard ratio was 0.{70 + i % 20} (95% CI 0.61 to 0.95)."
    for i in range(120)
)


def _max_chunk_chars() -> int:
    settings = get_settings()
    return (settings.chunk_size_tokens + settings.chunk_overlap_tokens) * 4 + 1


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


def test_paragraph_longer_than_the_target_is_split() -> None:
    """A single long paragraph used to come back whole, however long it was."""
    out = _recursive_split("Short opening line.\n" + _LONG_PARAGRAPH, 400, 40)
    assert len(out) > 10
    assert max(len(c) for c in out) <= 400 + 40 + 1


def test_splitting_a_long_paragraph_loses_no_words() -> None:
    out = _recursive_split(_LONG_PARAGRAPH, 400, 0)
    assert re.findall(r"\w+", " ".join(out)) == re.findall(r"\w+", _LONG_PARAGRAPH)


def test_text_without_any_separator_is_hard_wrapped() -> None:
    out = _recursive_split("x" * 1000, 300, 0)
    assert [len(c) for c in out] == [300, 300, 300, 100]


def test_no_chunk_exceeds_target_plus_overlap(sample_paper: Paper) -> None:
    body = f"Methods\nStudy population\n{_LONG_PARAGRAPH}\nResults\n{_LONG_PARAGRAPH}"
    chunks = chunk_paper(sample_paper.model_copy(update={"full_text": body}))
    assert max(len(c.text) for c in chunks) <= _max_chunk_chars()
    assert {c.section for c in chunks} == {"abstract", "methods", "results"}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_text_before_the_first_header_is_kept() -> None:
    """Many articles open with an untitled introduction; it used to be dropped."""
    sections = list(_split_by_section("Opening paragraph.\nMethods\nWe randomized patients."))
    assert sections == [("body", "Opening paragraph."), ("methods", "We randomized patients.")]


def test_text_with_no_header_is_one_body_section() -> None:
    assert list(_split_by_section("Just prose.\nMore prose.")) == [
        ("body", "Just prose.\nMore prose.")
    ]


def test_numbered_and_variant_headers_are_recognised() -> None:
    text = (
        "1. Introduction\nWhy.\n2. Materials and Methods\nHow.\n"
        "RESULTS\nWhat.\n4. Discussions\nSo what.\n5. Conclusions\nThe end."
    )
    assert [name for name, _ in _split_by_section(text)] == [
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
    ]


def test_a_sentence_that_mentions_a_section_word_is_not_a_header() -> None:
    text = "Methods\nThe results were adjudicated blind.\nResults of the trial\nMACE fell."
    assert [name for name, _ in _split_by_section(text)] == ["methods"]


def test_subheading_stays_with_the_paragraph_it_introduces(sample_paper: Paper) -> None:
    body = f"Methods\nStudy population\n{_LONG_PARAGRAPH}"
    chunks = chunk_paper(sample_paper.model_copy(update={"full_text": body}))
    methods = [c.text for c in chunks if c.section == "methods"]
    assert methods[0].startswith("Study population\nIn subgroup 0")


def test_heading_only_section_is_dropped(sample_paper: Paper) -> None:
    """A section whose content was a table leaves only its heading behind."""
    body = "Key Points\nMethods\n" + "We randomized patients. " * 10
    chunks = chunk_paper(sample_paper.model_copy(update={"full_text": body}))
    assert {c.section for c in chunks} == {"abstract", "methods"}


def test_token_limit_is_enforced_with_the_given_counter(sample_paper: Paper) -> None:
    """The character bound alone lets number-heavy text exceed the embedder's limit."""

    def count_words(text: str) -> int:
        return len(text.split())

    paper = sample_paper.model_copy(update={"full_text": f"Results\n{_LONG_PARAGRAPH}"})
    unbounded = chunk_paper(paper)
    assert max(count_words(c.text) for c in unbounded) > 150

    bounded = chunk_paper(paper, count_tokens=count_words, max_tokens=150)
    assert max(c.n_tokens for c in bounded) <= 150
    assert all(c.n_tokens == count_words(c.text) for c in bounded)
    assert len({c.chunk_id for c in bounded}) == len(bounded) > len(unbounded)
    # The abstract is short, so it is the same chunk either way.
    assert bounded[0] == unbounded[0].model_copy(update={"n_tokens": bounded[0].n_tokens})
