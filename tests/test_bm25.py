"""BM25 over a toy corpus: tokenization, ranking and hydration."""

from __future__ import annotations

import pytest

from medrag.models import Chunk, Paper, RetrievalMethod
from medrag.retrieval.bm25 import BM25Retriever, tokenize
from medrag.store import Store


def _chunk(cid: str, pmid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, pmid=pmid, text=text, chunk_index=0, n_tokens=len(text.split()))


def _paper(sample_paper: Paper, pmid: str) -> Paper:
    return sample_paper.model_copy(update={"pmid": pmid, "title": f"Paper {pmid}"})


@pytest.fixture
def corpus(sample_paper: Paper) -> tuple[list[Chunk], dict[str, Paper]]:
    chunks = [
        _chunk("c1", "1", "Semaglutide lowered body weight in adults with obesity."),
        _chunk("c2", "2", "SUSTAIN-6 tested semaglutide against placebo for MACE."),
        _chunk("c3", "3", "Liraglutide in LEADER reduced cardiovascular death."),
        _chunk("c4", "4", "Dulaglutide and blood pressure in type 2 diabetes."),
        _chunk("c5", "5", "GLP-1 receptor agonists and heart failure hospitalisation."),
    ]
    return chunks, {c.pmid: _paper(sample_paper, c.pmid) for c in chunks}


def test_tokenize_keeps_hyphenated_terms_and_drops_stopwords() -> None:
    assert tokenize("The SUSTAIN-6 trial of GLP-1 RAs") == ["sustain-6", "trial", "glp-1", "ras"]


def test_exact_trial_acronym_ranks_first(corpus) -> None:
    chunks, papers = corpus
    results = BM25Retriever(chunks, papers).search("What did SUSTAIN-6 show?", top_k=3)
    assert results[0].chunk.chunk_id == "c2"
    assert results[0].paper.pmid == "2"
    assert results[0].method is RetrievalMethod.BM25


def test_results_are_ranked_by_score_and_capped_at_top_k(corpus) -> None:
    chunks, papers = corpus
    results = BM25Retriever(chunks, papers).search("semaglutide MACE placebo", top_k=1)
    assert [r.chunk.chunk_id for r in results] == ["c2"]

    results = BM25Retriever(chunks, papers).search("semaglutide MACE placebo", top_k=5)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_chunks_with_no_matching_term_are_not_returned(corpus) -> None:
    chunks, papers = corpus
    results = BM25Retriever(chunks, papers).search("liraglutide", top_k=10)
    assert [r.chunk.chunk_id for r in results] == ["c3"]


def test_hit_without_a_paper_is_skipped_and_ranks_stay_contiguous(corpus) -> None:
    chunks, papers = corpus
    del papers["2"]
    results = BM25Retriever(chunks, papers).search("semaglutide", top_k=10)
    assert [r.chunk.chunk_id for r in results] == ["c1"]
    assert results[0].rank == 1


@pytest.mark.parametrize("query", ["", "the of and", "!!!"])
def test_query_with_no_usable_tokens_returns_nothing(corpus, query: str) -> None:
    chunks, papers = corpus
    assert BM25Retriever(chunks, papers).search(query, top_k=5) == []


def test_empty_corpus_returns_nothing() -> None:
    retriever = BM25Retriever([], {})
    assert len(retriever) == 0
    assert retriever.search("semaglutide", top_k=5) == []


def test_from_store_loads_the_whole_corpus(tmp_path, corpus) -> None:
    chunks, papers = corpus
    with Store(tmp_path / "medrag.db") as store:
        store.save_papers(papers.values())
        store.save_chunks(chunks)
        retriever = BM25Retriever.from_store(store)
    assert len(retriever) == 5
    assert retriever.search("LEADER", top_k=1)[0].chunk.chunk_id == "c3"
