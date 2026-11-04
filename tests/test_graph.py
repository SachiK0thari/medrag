"""Unit tests for the LangGraph agent loop.

The LLM is replaced with canned responses and the retrieval stack with
in-memory fakes, so these run with no network, no models, and no Pinecone.
"""

from __future__ import annotations

import pytest

from medrag.agent import graph as graph_mod
from medrag.agent import llm
from medrag.agent.graph import (
    MedRagAgent,
    _dedupe_by_chunk_id,
    _extract_citations,
    _extract_json,
    _parse_reflection,
    _parse_sub_queries,
    _unique_pmids,
)
from medrag.agent.llm import LLMResponse
from medrag.agent.prompts import DECOMPOSE_SYSTEM, DECOMPOSE_SYSTEM_FAITHFUL
from medrag.models import (
    Chunk,
    Paper,
    PaperSource,
    RetrievalMethod,
    RetrievedChunk,
    SubQuery,
)

# --------------------------------------------------------------------------- #
# Fixtures / fakes                                                            #
# --------------------------------------------------------------------------- #


def _rc(pmid: str, idx: int = 0, score: float = 1.0, rank: int = 1) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{pmid}-{idx}",
        pmid=pmid,
        text=f"text of {pmid} chunk {idx}",
        section="abstract",
        chunk_index=idx,
        n_tokens=5,
    )
    paper = Paper(
        pmid=pmid,
        title=f"Paper {pmid}",
        authors=["Ada Lovelace"],
        journal="J Test",
        year=2024,
        source=PaperSource.PUBMED_ABSTRACT,
    )
    return RetrievedChunk(
        chunk=chunk, paper=paper, score=score, rank=rank, method=RetrievalMethod.HYBRID_RERANKED
    )


class FakeHybrid:
    """Returns a fixed candidate list per query; records every query it saw."""

    def __init__(self, results: dict[str, list[RetrievedChunk]]) -> None:
        self.results = results
        self.calls: list[str] = []

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.calls.append(query)
        return list(self.results.get(query, []))


class FakeReranker:
    """Keeps input order, assigns descending scores from `base`; records calls.

    `base` per query lets a test make one sub-query's scores dominate another's,
    reproducing the cross-sub-query score-comparability problem.
    """

    def __init__(self, base: dict[str, float] | None = None) -> None:
        self.base = base or {}
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, len(chunks)))
        start = self.base.get(query, 1.0)
        return [
            RetrievedChunk(
                chunk=rc.chunk,
                paper=rc.paper,
                score=start - i * 0.01,
                rank=i + 1,
                method=RetrievalMethod.HYBRID_RERANKED,
            )
            for i, rc in enumerate(chunks[:top_k])
        ]


class ScriptedLLM:
    """Dispatches on the system prompt so each node gets a tailored response."""

    def __init__(
        self,
        *,
        decompose: str = '[{"text": "sq1"}]',
        reflect: str | list[str] = '{"sufficient": true, "missing": ""}',
        answer: str = "Fact one [1]. Fact two [2].",
    ) -> None:
        self.decompose = decompose
        self.reflect = [reflect] if isinstance(reflect, str) else list(reflect)
        self.answer = answer
        self.calls: list[str] = []

    def __call__(self, system: str, user: str, **kwargs: object) -> LLMResponse:
        if system in (DECOMPOSE_SYSTEM, DECOMPOSE_SYSTEM_FAITHFUL):
            self.calls.append("decompose")
            text = self.decompose
        elif system.startswith("You are grading"):
            self.calls.append("reflect")
            text = self.reflect.pop(0) if len(self.reflect) > 1 else self.reflect[0]
        else:
            self.calls.append("answer")
            text = self.answer
        return LLMResponse(text=text, input_tokens=10, output_tokens=5, model="fake")


@pytest.fixture
def scripted_llm(monkeypatch: pytest.MonkeyPatch):
    def _install(**kwargs: object) -> ScriptedLLM:
        fake = ScriptedLLM(**kwargs)  # type: ignore[arg-type]
        monkeypatch.setattr(llm, "complete", fake)
        return fake

    return _install


def _agent(hybrid: FakeHybrid, reranker: FakeReranker, **knobs: object) -> MedRagAgent:
    return MedRagAgent(hybrid=hybrid, reranker=reranker, **knobs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Pure parsing helpers                                                         #
# --------------------------------------------------------------------------- #


def test_extract_json_finds_first_balanced_block():
    assert _extract_json("noise [1, [2, 3]] trailing [4]", "[", "]") == "[1, [2, 3]]"
    assert _extract_json("no json here", "{", "}") is None
    assert _extract_json("unbalanced {", "{", "}") is None


def test_parse_sub_queries_objects_and_strings():
    text = '[{"text": "a", "rationale": "r"}, "b", {"nope": 1}]'
    out = _parse_sub_queries(text, fallback="q", limit=4)
    assert [sq.text for sq in out] == ["a", "b"]
    assert out[0].rationale == "r"
    assert out[1].rationale is None


def test_parse_sub_queries_respects_limit():
    text = '["a", "b", "c", "d", "e"]'
    assert len(_parse_sub_queries(text, fallback="q", limit=2)) == 2


@pytest.mark.parametrize("bad", ["not json", "[]", "[1, 2]", '{"text": "obj not list"}'])
def test_parse_sub_queries_falls_back_to_question(bad: str):
    out = _parse_sub_queries(bad, fallback="the question", limit=4)
    assert len(out) == 1
    assert out[0].text == "the question"
    assert out[0].rationale and "fallback" in out[0].rationale


def test_parse_reflection():
    assert _parse_reflection('{"sufficient": false, "missing": "dose data"}') == (
        False,
        "dose data",
    )
    assert _parse_reflection('{"sufficient": true}') == (True, "")
    # Unparseable -> assume sufficient so the loop terminates.
    assert _parse_reflection("garbage") == (True, "")
    # Missing key -> insufficient but with no follow-up text.
    assert _parse_reflection('{"missing": "x"}') == (False, "x")


def test_extract_citations_filters_and_dedupes_in_order():
    sources = [_rc("1"), _rc("2"), _rc("3")]
    text = "A [2]. B [9]. C [2]. D [1]. E [0]."
    cits = _extract_citations(text, sources)
    assert [c.marker for c in cits] == ["[2]", "[1]"]
    assert cits[0].pmid == "2"
    assert cits[0].chunk_id == "2-0"


def test_dedupe_by_chunk_id_keeps_first_occurrence():
    a, b = _rc("1", score=0.9), _rc("1", score=0.1)
    out = _dedupe_by_chunk_id([a, _rc("2"), b])
    assert [rc.chunk.chunk_id for rc in out] == ["1-0", "2-0"]
    assert out[0].score == 0.9


def test_unique_pmids_dedupes_and_limits():
    chunks = [_rc("1", 0), _rc("1", 1), _rc("2"), _rc("3")]
    assert _unique_pmids(chunks) == ["1", "2", "3"]
    assert _unique_pmids(chunks, limit=2) == ["1", "2"]


# --------------------------------------------------------------------------- #
# Agent loop                                                                   #
# --------------------------------------------------------------------------- #


def test_happy_path_produces_cited_answer_and_trace(scripted_llm):
    scripted_llm(decompose='[{"text": "sq1"}, {"text": "sq2"}]')
    hybrid = FakeHybrid({"sq1": [_rc("11"), _rc("12")], "sq2": [_rc("21")]})
    agent = _agent(hybrid, FakeReranker())

    retrieved, answer, trace = agent.run_with_trace("Q?")

    assert hybrid.calls == ["sq1", "sq2"]
    assert {rc.paper.pmid for rc in retrieved} == {"11", "12", "21"}
    assert answer.question == "Q?"
    assert [c.marker for c in answer.citations] == ["[1]", "[2]"]
    assert answer.n_retrieval_calls == 2
    assert answer.total_chunks_considered == 3
    assert [sq.text for sq in answer.sub_queries] == ["sq1", "sq2"]
    steps = [e["step"] for e in trace]
    assert steps == ["decompose", "retrieve", "retrieve", "merge", "reflect"]


def test_run_matches_run_with_trace_minus_trace(scripted_llm):
    scripted_llm()
    hybrid = FakeHybrid({"sq1": [_rc("1")]})
    agent = _agent(hybrid, FakeReranker())
    retrieved, answer = agent.run("Q?")
    assert [rc.paper.pmid for rc in retrieved] == ["1"]
    assert answer.text.startswith("Fact one")
    assert agent.answer("Q?").text == answer.text


def test_decompose_failure_falls_back_to_original_question(scripted_llm):
    scripted_llm(decompose="I cannot help with that.")
    hybrid = FakeHybrid({"Q?": [_rc("1")]})
    _, answer = _agent(hybrid, FakeReranker()).run("Q?")
    assert hybrid.calls == ["Q?"]
    assert answer.sub_queries[0].text == "Q?"


def test_reflection_loop_runs_followup_only_for_pending_subqueries(scripted_llm):
    scripted_llm(
        reflect=[
            '{"sufficient": false, "missing": "follow-up query"}',
            '{"sufficient": true, "missing": ""}',
        ]
    )
    hybrid = FakeHybrid({"sq1": [_rc("1")], "follow-up query": [_rc("2")]})
    agent = _agent(hybrid, FakeReranker(), max_reflection_loops=2)

    retrieved, answer, trace = agent.run_with_trace("Q?")

    # sq1 must not be re-run when the follow-up is appended.
    assert hybrid.calls == ["sq1", "follow-up query"]
    assert answer.n_retrieval_calls == 2
    assert {rc.paper.pmid for rc in retrieved} == {"1", "2"}
    assert [sq.text for sq in answer.sub_queries] == ["sq1", "follow-up query"]
    assert sum(1 for e in trace if e["step"] == "reflect") == 2


def test_reflection_loop_terminates_at_max_loops(scripted_llm):
    fake = scripted_llm(reflect='{"sufficient": false, "missing": "more"}')
    hybrid = FakeHybrid({"sq1": [_rc("1")], "more": [_rc("2")]})
    agent = _agent(hybrid, FakeReranker(), max_reflection_loops=2)

    _, answer = agent.run("Q?")

    # 2 reflections that ask for more, then the cap short-circuits without an LLM call.
    assert fake.calls.count("reflect") == 2
    assert fake.calls[-1] == "answer"
    assert hybrid.calls == ["sq1", "more", "more"]
    assert isinstance(answer.text, str)


def test_zero_reflection_loops_skips_reflect_llm_call(scripted_llm):
    fake = scripted_llm()
    hybrid = FakeHybrid({"sq1": [_rc("1")]})
    _agent(hybrid, FakeReranker(), max_reflection_loops=0).run("Q?")
    assert fake.calls == ["decompose", "answer"]


def test_reflect_with_no_retrieved_chunks_goes_straight_to_answer(scripted_llm):
    fake = scripted_llm()
    hybrid = FakeHybrid({})  # every search returns nothing
    _, answer = _agent(hybrid, FakeReranker()).run("Q?")
    assert "reflect" not in fake.calls
    assert answer.citations == []
    assert answer.confidence == 0.0
    assert answer.total_chunks_considered == 0


def test_keep_original_query_prepends_question_and_dedupes(scripted_llm):
    scripted_llm(decompose='[{"text": "q?"}, {"text": "sq2"}, {"text": "sq3"}, {"text": "sq4"}]')
    hybrid = FakeHybrid({})
    agent = _agent(hybrid, FakeReranker(), keep_original_query=True)

    _, answer = agent.run("Q?")

    texts = [sq.text for sq in answer.sub_queries]
    assert texts[0] == "Q?"
    assert "q?" not in texts  # case-insensitive duplicate of the question removed
    assert len(texts) <= agent._settings.max_subqueries
    assert hybrid.calls[0] == "Q?"


def test_v1_merge_sorts_by_incomparable_per_subquery_scores(scripted_llm):
    """Documents the v1 failure mode: the tangent's chunks outrank the real hit."""
    scripted_llm(decompose='[{"text": "real"}, {"text": "tangent"}]')
    hybrid = FakeHybrid({"real": [_rc("gold")], "tangent": [_rc("t1"), _rc("t2")]})
    reranker = FakeReranker(base={"real": 0.5, "tangent": 0.9})

    retrieved, _ = _agent(hybrid, reranker).run("Q?")

    assert [rc.paper.pmid for rc in retrieved] == ["t1", "t2", "gold"]
    # No rerank against the original question happened.
    assert all(q != "Q?" for q, _ in reranker.calls)


def test_rerank_against_original_rescores_merged_pool(scripted_llm):
    scripted_llm(decompose='[{"text": "real"}, {"text": "tangent"}]')
    hybrid = FakeHybrid({"real": [_rc("gold")], "tangent": [_rc("t1"), _rc("t2")]})
    reranker = FakeReranker(base={"real": 0.5, "tangent": 0.9})

    retrieved, _, trace = _agent(hybrid, reranker, rerank_against_original=True).run_with_trace(
        "Q?"
    )

    # Final call reranks the whole 3-chunk pool against the user's question.
    assert reranker.calls[-1] == ("Q?", 3)
    assert len(retrieved) == 3
    merge = next(e for e in trace if e["step"] == "merge")
    assert merge["rerank_against_original"] is True


def test_faithful_decomposition_uses_v2_prompt(scripted_llm, monkeypatch):
    seen: list[str] = []
    real = ScriptedLLM()

    def spy(system: str, user: str, **kw: object) -> LLMResponse:
        seen.append(system)
        return real(system, user, **kw)

    monkeypatch.setattr(llm, "complete", spy)
    hybrid = FakeHybrid({"sq1": [_rc("1")]})
    _agent(hybrid, FakeReranker(), faithful_decomposition=True).run("Q?")
    assert seen[0] == DECOMPOSE_SYSTEM_FAITHFUL
    seen.clear()
    _agent(hybrid, FakeReranker()).run("Q?")
    assert seen[0] == DECOMPOSE_SYSTEM


def test_generate_answer_slices_top_k(scripted_llm):
    scripted_llm(answer="Only [1] and [5].")
    retrieved = [_rc(str(i)) for i in range(1, 7)]
    answer = graph_mod.generate_answer("Q?", retrieved, top_k=3)
    # [5] is outside the top-3 the model was shown -> not a valid citation.
    assert [c.pmid for c in answer.citations] == ["1"]
    assert answer.total_chunks_considered == 6
    assert answer.n_retrieval_calls == 1
    assert answer.sub_queries == []


def test_invoke_raises_if_graph_yields_no_answer(scripted_llm, monkeypatch):
    scripted_llm()
    agent = _agent(FakeHybrid({}), FakeReranker())

    class NoAnswerGraph:
        def invoke(self, state: dict) -> dict:
            return {"retrieved": []}

    monkeypatch.setattr(agent, "_get_graph", lambda: NoAnswerGraph())
    with pytest.raises(RuntimeError, match="without producing an Answer"):
        agent.run("Q?")


def test_subquery_model_roundtrip():
    sq = SubQuery(text="x")
    assert sq.rationale is None
