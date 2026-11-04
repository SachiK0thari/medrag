"""LangGraph agent loop.

State machine:
    decompose -> retrieve -> reflect -> (retrieve | answer) -> end

`retrieve` only runs the sub-queries that haven't been executed yet, so the
reflect node can append a follow-up sub-query and loop without re-running
anything. Accumulated `RetrievedChunk`s are deduped by chunk_id and kept in
rerank-score order so the answer node can slice the top-N directly.

Ablation knobs (all default to the original v1 behaviour so the ``agentic``
eval row stays a true baseline):

- ``keep_original_query``: always search the user's question verbatim as the
  first sub-query. v1 only searched the LLM's paraphrased sub-queries, which
  drops exact tokens (trial acronyms, drug names) that BM25 relies on.
- ``rerank_against_original``: after merging sub-query results, re-score the
  whole pool with the cross-encoder against the *original question*. v1 sorted
  the merged pool by per-sub-query rerank scores, which are not comparable
  across sub-queries, so chunks matching a tangential sub-query could outrank
  chunks matching the actual question.
- ``max_reflection_loops``: override the settings value (0 disables reflect).
- ``faithful_decomposition``: use the v2 planner prompt that keeps named
  entities verbatim and returns a single sub-query for atomic questions.

Every run records a per-sub-query ``trace`` (which PMIDs each sub-query
surfaced) so regressions can be attributed to a specific planner output.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, START, StateGraph

from medrag.agent import llm
from medrag.agent.prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER_TEMPLATE,
    DECOMPOSE_SYSTEM,
    DECOMPOSE_SYSTEM_FAITHFUL,
    REFLECT_SYSTEM,
)
from medrag.config import get_settings
from medrag.models import (
    Answer,
    Citation,
    RetrievedChunk,
    SubQuery,
)
from medrag.retrieval.bm25 import BM25Retriever
from medrag.retrieval.dense import DenseRetriever
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import Reranker
from medrag.store import Store

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


TraceEntry = dict[str, object]


class AgentState(TypedDict, total=False):
    question: str
    sub_queries: list[SubQuery]
    retrieved: list[RetrievedChunk]
    n_retrieval_calls: int
    reflection_loops: int
    needs_more: bool
    missing: str
    answer: Answer
    trace: list[TraceEntry]


# --------------------------------------------------------------------------- #
# Parsing helpers                                                              #
# --------------------------------------------------------------------------- #


def _extract_json(text: str, opener: str, closer: str) -> str | None:
    """Return the first top-level JSON substring bounded by opener/closer."""
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_sub_queries(text: str, fallback: str, limit: int) -> list[SubQuery]:
    raw = _extract_json(text, "[", "]")
    if raw is not None:
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            items = None
        if isinstance(items, list):
            parsed: list[SubQuery] = []
            for item in items[:limit]:
                if isinstance(item, dict) and "text" in item:
                    parsed.append(
                        SubQuery(
                            text=str(item["text"]).strip(),
                            rationale=(
                                str(item["rationale"]).strip() if item.get("rationale") else None
                            ),
                        )
                    )
                elif isinstance(item, str):
                    parsed.append(SubQuery(text=item.strip()))
            if parsed:
                return parsed
    return [SubQuery(text=fallback, rationale="fallback: decomposition failed")]


def _parse_reflection(text: str) -> tuple[bool, str]:
    raw = _extract_json(text, "{", "}")
    if raw is None:
        return True, ""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return True, ""
    sufficient = bool(obj.get("sufficient", False))
    missing = str(obj.get("missing", "")).strip()
    return sufficient, missing


# --------------------------------------------------------------------------- #
# Source rendering + citation extraction                                       #
# --------------------------------------------------------------------------- #


def _render_sources(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, rc in enumerate(chunks, start=1):
        header = f"[{i}] {rc.paper.display_citation} (PMID {rc.paper.pmid})"
        snippet = rc.chunk.text.strip().replace("\n", " ")
        if len(snippet) > 600:
            snippet = snippet[:600] + "..."
        lines.append(f"{header}\n    {snippet}")
    return "\n\n".join(lines)


def _extract_citations(answer_text: str, sources: list[RetrievedChunk]) -> list[Citation]:
    used_markers: list[int] = []
    seen: set[int] = set()
    for match in _CITATION_MARKER_RE.finditer(answer_text):
        n = int(match.group(1))
        if 1 <= n <= len(sources) and n not in seen:
            seen.add(n)
            used_markers.append(n)

    citations: list[Citation] = []
    for n in used_markers:
        rc = sources[n - 1]
        snippet = rc.chunk.text.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        citations.append(
            Citation(
                marker=f"[{n}]",
                pmid=rc.paper.pmid,
                chunk_id=rc.chunk.chunk_id,
                display_citation=rc.paper.display_citation,
                snippet=snippet,
            )
        )
    return citations


def _confidence(citations: list[Citation], sources: list[RetrievedChunk]) -> float:
    if not sources:
        return 0.0
    if not citations:
        return 0.2
    base = 0.5 + 0.1 * len(citations)
    return max(0.0, min(1.0, base))


def _dedupe_by_chunk_id(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for rc in chunks:
        cid = rc.chunk.chunk_id
        if cid in seen:
            continue
        seen.add(cid)
        out.append(rc)
    return out


def _unique_pmids(chunks: list[RetrievedChunk], limit: int = 10) -> list[str]:
    """First `limit` distinct PMIDs in rank order (mirrors the eval harness)."""
    seen: set[str] = set()
    out: list[str] = []
    for rc in chunks:
        pmid = rc.paper.pmid
        if pmid in seen:
            continue
        seen.add(pmid)
        out.append(pmid)
        if len(out) >= limit:
            break
    return out


def generate_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    *,
    top_k: int | None = None,
    sub_queries: list[SubQuery] | None = None,
    n_retrieval_calls: int = 1,
) -> Answer:
    """Single-pass answer generation over a fixed retrieval.

    Shared by the agent's `_answer_node` and the eval harness: using the same
    prompt + citation logic for every config isolates retrieval quality as the
    only independent variable when comparing faithfulness across baselines.
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.rerank_top_k
    top = retrieved[:k]
    sources_str = _render_sources(top)
    user = ANSWER_USER_TEMPLATE.format(question=question, sources=sources_str)
    resp = llm.complete(ANSWER_SYSTEM, user)
    text = resp.text.strip()
    citations = _extract_citations(text, top)
    return Answer(
        question=question,
        text=text,
        citations=citations,
        sub_queries=list(sub_queries) if sub_queries else [],
        confidence=_confidence(citations, top),
        n_retrieval_calls=max(n_retrieval_calls, 1),
        total_chunks_considered=len(retrieved),
    )


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #


class MedRagAgent:
    """LangGraph-backed agentic RAG loop over the hybrid retriever."""

    def __init__(
        self,
        *,
        hybrid: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        store: Store | None = None,
        keep_original_query: bool = False,
        rerank_against_original: bool = False,
        max_reflection_loops: int | None = None,
        faithful_decomposition: bool = False,
    ) -> None:
        self._settings = get_settings()
        self._store = store
        self._hybrid = hybrid
        self._reranker = reranker
        self._graph: CompiledStateGraph | None = None
        self._keep_original_query = keep_original_query
        self._rerank_against_original = rerank_against_original
        self._max_reflection_loops = (
            max_reflection_loops
            if max_reflection_loops is not None
            else self._settings.max_reflection_loops
        )
        self._decompose_system = (
            DECOMPOSE_SYSTEM_FAITHFUL if faithful_decomposition else DECOMPOSE_SYSTEM
        )

    # ---- lazy component wiring ------------------------------------------ #

    def _get_hybrid(self) -> HybridRetriever:
        if self._hybrid is None:
            if self._store is None:
                self._store = Store()
            bm25 = BM25Retriever.from_store(self._store)
            dense = DenseRetriever(store=self._store)
            self._hybrid = HybridRetriever(bm25=bm25, dense=dense)
        return self._hybrid

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def _get_graph(self) -> CompiledStateGraph:
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    # ---- nodes ----------------------------------------------------------- #

    def _decompose_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        limit = self._settings.max_subqueries
        resp = llm.complete(self._decompose_system, question)
        sub_queries = _parse_sub_queries(resp.text, fallback=question, limit=limit)
        if self._keep_original_query:
            original = SubQuery(text=question, rationale="original question, searched verbatim")
            rest = [sq for sq in sub_queries if sq.text.strip().lower() != question.strip().lower()]
            sub_queries = [original, *rest][:limit]
        trace: list[TraceEntry] = [
            {
                "step": "decompose",
                "prompt": "faithful"
                if self._decompose_system is DECOMPOSE_SYSTEM_FAITHFUL
                else "v1",
                "sub_queries": [sq.text for sq in sub_queries],
            }
        ]
        return {"sub_queries": sub_queries, "trace": trace}

    def _retrieve_node(self, state: AgentState) -> AgentState:
        hybrid = self._get_hybrid()
        reranker = self._get_reranker()
        already = state.get("n_retrieval_calls", 0)
        pending = state["sub_queries"][already:]

        trace = list(state.get("trace", []))
        new_chunks: list[RetrievedChunk] = []
        for sq in pending:
            fused = hybrid.search(sq.text)
            if not fused:
                trace.append({"step": "retrieve", "sub_query": sq.text, "top_pmids": []})
                continue
            reranked = reranker.rerank(sq.text, fused, self._settings.rerank_top_k)
            new_chunks.extend(reranked)
            trace.append(
                {
                    "step": "retrieve",
                    "sub_query": sq.text,
                    "top_pmids": _unique_pmids(reranked),
                }
            )

        merged = _dedupe_by_chunk_id(state.get("retrieved", []) + new_chunks)
        if self._rerank_against_original and merged:
            # Re-score the whole accumulated pool against the user's question so
            # chunks from different sub-queries are ranked on a common scale.
            merged = reranker.rerank(state["question"], merged, len(merged))
        else:
            merged.sort(key=lambda rc: rc.score, reverse=True)
        trace.append(
            {
                "step": "merge",
                "n_chunks": len(merged),
                "rerank_against_original": self._rerank_against_original,
                "top_pmids": _unique_pmids(merged),
            }
        )
        return {
            "retrieved": merged,
            "n_retrieval_calls": already + len(pending),
            "trace": trace,
        }

    def _reflect_node(self, state: AgentState) -> AgentState:
        loops = state.get("reflection_loops", 0)
        if loops >= self._max_reflection_loops:
            return {"needs_more": False, "missing": "", "reflection_loops": loops + 1}

        top = state.get("retrieved", [])[: self._settings.rerank_top_k]
        if not top:
            return {"needs_more": False, "missing": "", "reflection_loops": loops + 1}

        user = f"Question: {state['question']}\n\nSources:\n{_render_sources(top)}"
        resp = llm.complete(REFLECT_SYSTEM, user, max_tokens=256)
        sufficient, missing = _parse_reflection(resp.text)
        trace = [
            *state.get("trace", []),
            {"step": "reflect", "sufficient": sufficient, "missing": missing},
        ]
        return {
            "needs_more": (not sufficient) and bool(missing),
            "missing": missing,
            "reflection_loops": loops + 1,
            "trace": trace,
        }

    def _followup_node(self, state: AgentState) -> AgentState:
        missing = state.get("missing", "").strip()
        if not missing:
            return {}
        follow_up = SubQuery(text=missing, rationale="follow-up from reflection")
        return {"sub_queries": state["sub_queries"] + [follow_up]}

    def _answer_node(self, state: AgentState) -> AgentState:
        answer = generate_answer(
            question=state["question"],
            retrieved=state.get("retrieved", []),
            top_k=self._settings.rerank_top_k,
            sub_queries=state.get("sub_queries", []),
            n_retrieval_calls=state.get("n_retrieval_calls", 1),
        )
        return {"answer": answer}

    # ---- graph ----------------------------------------------------------- #

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("decompose", self._decompose_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("followup", self._followup_node)
        graph.add_node("generate", self._answer_node)

        graph.add_edge(START, "decompose")
        graph.add_edge("decompose", "retrieve")
        graph.add_edge("retrieve", "reflect")
        graph.add_conditional_edges(
            "reflect",
            lambda s: "followup" if s.get("needs_more") else "generate",
            {"followup": "followup", "generate": "generate"},
        )
        graph.add_edge("followup", "retrieve")
        graph.add_edge("generate", END)
        return graph.compile()

    # ---- public API ------------------------------------------------------ #

    def answer(self, question: str) -> Answer:
        _, answer, _ = self._invoke(question)
        return answer

    def retrieve_for_eval(self, question: str) -> list[RetrievedChunk]:
        """Run the full agent loop and return the retrieved chunks only.

        Used by the eval harness to score the agentic config on the same
        retrieval metrics (recall@k, MRR, nDCG) as the non-agentic baselines.
        """
        retrieved, _, _ = self._invoke(question)
        return retrieved

    def run(self, question: str) -> tuple[list[RetrievedChunk], Answer]:
        """Run the full agent loop and return both retrieval and answer.

        Single public entry point for the eval harness, which needs both
        halves to compute retrieval metrics AND faithfulness / cost.
        """
        retrieved, answer, _ = self._invoke(question)
        return retrieved, answer

    def run_with_trace(
        self, question: str
    ) -> tuple[list[RetrievedChunk], Answer, list[TraceEntry]]:
        """Like `run`, plus the per-step trace (sub-queries and PMIDs each found)."""
        return self._invoke(question)

    def _invoke(self, question: str) -> tuple[list[RetrievedChunk], Answer, list[TraceEntry]]:
        initial: AgentState = {
            "question": question,
            "sub_queries": [],
            "retrieved": [],
            "n_retrieval_calls": 0,
            "reflection_loops": 0,
            "needs_more": False,
            "missing": "",
            "trace": [],
        }
        final = self._get_graph().invoke(initial)
        answer = final.get("answer")
        if not isinstance(answer, Answer):
            raise RuntimeError("agent graph terminated without producing an Answer")
        return list(final.get("retrieved", [])), answer, list(final.get("trace", []))
