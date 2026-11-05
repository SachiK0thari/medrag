"""EvalConfig implementations for the eval harness.

Each class wraps one retrieval strategy behind the `EvalConfig` protocol
(name + run(question) -> EvalResult) so `run_all` can benchmark them
uniformly. `EvalResult` carries both the retrieval and an answer, because
the harness scores retrieval metrics (recall / nDCG) on the chunks and
faithfulness + cost on the answer.

The four retrieval-only configs all delegate answer generation to the
shared `generate_answer` helper in `medrag.agent.graph`. Using the same
prompt + citation logic for every config isolates retrieval quality as
the only independent variable — a BM25-only vs hybrid-rerank faithfulness
delta then cleanly attributes to the retriever, not a different prompt.

Components (store, BM25, dense, reranker, agent) are built lazily and
shared across configs via `_SharedComponents` so a full eval run only
loads the corpus and the cross-encoder once.

Agentic ablations: `agentic` is the original v1 loop, reproduced by passing
the legacy knob values explicitly (`MedRagAgent`'s own defaults are v2). Each
`agentic_*` variant flips exactly one knob relative to v1 so the eval table
isolates the effect of that change; `agentic_v2` is the combination that
ships as the default in the API and UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from medrag.agent.graph import MedRagAgent, TraceEntry, generate_answer
from medrag.config import get_settings
from medrag.models import Answer, RetrievedChunk
from medrag.retrieval.bm25 import BM25Retriever
from medrag.retrieval.dense import DenseRetriever
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import Reranker
from medrag.store import Store


@dataclass(frozen=True)
class EvalResult:
    """What one config produces for one question."""

    retrieved: list[RetrievedChunk]
    answer: Answer
    # Per-step agent trace (empty for the non-agentic configs).
    trace: list[TraceEntry] = field(default_factory=list)


class EvalConfig(Protocol):
    """Structural protocol every config class must satisfy.

    Kept minimal on purpose: one public `run` method per config. The
    harness doesn't know (or care) whether a config is retrieval-only or
    an agentic loop — it just asks for an `EvalResult`.
    """

    name: str

    def run(self, question: str) -> EvalResult: ...


class _SharedComponents:
    """Lazily-constructed retrieval components reused across configs."""

    def __init__(self) -> None:
        self._store: Store | None = None
        self._bm25: BM25Retriever | None = None
        self._dense: DenseRetriever | None = None
        self._hybrid: HybridRetriever | None = None
        self._reranker: Reranker | None = None
        self._agents: dict[tuple[bool, bool, int | None, bool], MedRagAgent] = {}

    def store(self) -> Store:
        if self._store is None:
            self._store = Store()
        return self._store

    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever.from_store(self.store())
        return self._bm25

    def dense(self) -> DenseRetriever:
        if self._dense is None:
            self._dense = DenseRetriever(store=self.store())
        return self._dense

    def hybrid(self) -> HybridRetriever:
        if self._hybrid is None:
            self._hybrid = HybridRetriever(bm25=self.bm25(), dense=self.dense())
        return self._hybrid

    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def agent(
        self,
        *,
        keep_original_query: bool = False,
        rerank_against_original: bool = False,
        max_reflection_loops: int | None = None,
        faithful_decomposition: bool = False,
    ) -> MedRagAgent:
        """One agent per knob combination, all sharing the retrieval stack."""
        key = (
            keep_original_query,
            rerank_against_original,
            max_reflection_loops,
            faithful_decomposition,
        )
        if key not in self._agents:
            self._agents[key] = MedRagAgent(
                hybrid=self.hybrid(),
                reranker=self.reranker(),
                store=self.store(),
                keep_original_query=keep_original_query,
                rerank_against_original=rerank_against_original,
                max_reflection_loops=max_reflection_loops,
                faithful_decomposition=faithful_decomposition,
            )
        return self._agents[key]


class DenseOnlyConfig:
    name = "dense_only"

    def __init__(self, components: _SharedComponents) -> None:
        self._c = components
        self._settings = get_settings()

    def run(self, question: str) -> EvalResult:
        retrieved = self._c.dense().search(question, self._settings.dense_top_k)
        answer = generate_answer(question, retrieved, top_k=self._settings.rerank_top_k)
        return EvalResult(retrieved=retrieved, answer=answer)


class BM25OnlyConfig:
    name = "bm25_only"

    def __init__(self, components: _SharedComponents) -> None:
        self._c = components
        self._settings = get_settings()

    def run(self, question: str) -> EvalResult:
        retrieved = self._c.bm25().search(question, self._settings.bm25_top_k)
        answer = generate_answer(question, retrieved, top_k=self._settings.rerank_top_k)
        return EvalResult(retrieved=retrieved, answer=answer)


class HybridRRFConfig:
    name = "hybrid_rrf"

    def __init__(self, components: _SharedComponents) -> None:
        self._c = components
        self._settings = get_settings()

    def run(self, question: str) -> EvalResult:
        retrieved = self._c.hybrid().search(question)
        answer = generate_answer(question, retrieved, top_k=self._settings.rerank_top_k)
        return EvalResult(retrieved=retrieved, answer=answer)


class HybridRerankedConfig:
    name = "hybrid_reranked"

    def __init__(self, components: _SharedComponents) -> None:
        self._c = components
        self._settings = get_settings()

    def run(self, question: str) -> EvalResult:
        fused = self._c.hybrid().search(question)
        retrieved = self._c.reranker().rerank(question, fused, self._settings.rerank_top_k)
        answer = generate_answer(question, retrieved, top_k=self._settings.rerank_top_k)
        return EvalResult(retrieved=retrieved, answer=answer)


class AgenticConfig:
    """Full LangGraph loop. Knobs select an ablation variant (see module doc)."""

    def __init__(
        self,
        components: _SharedComponents,
        *,
        name: str = "agentic",
        keep_original_query: bool = False,
        rerank_against_original: bool = False,
        max_reflection_loops: int | None = None,
        faithful_decomposition: bool = False,
    ) -> None:
        self.name = name
        self._c = components
        self._keep_original_query = keep_original_query
        self._rerank_against_original = rerank_against_original
        self._max_reflection_loops = max_reflection_loops
        self._faithful_decomposition = faithful_decomposition

    def agent(self) -> MedRagAgent:
        return self._c.agent(
            keep_original_query=self._keep_original_query,
            rerank_against_original=self._rerank_against_original,
            max_reflection_loops=self._max_reflection_loops,
            faithful_decomposition=self._faithful_decomposition,
        )

    def run(self, question: str) -> EvalResult:
        retrieved, answer, trace = self.agent().run_with_trace(question)
        return EvalResult(retrieved=retrieved, answer=answer, trace=trace)


def build_all_configs() -> list[EvalConfig]:
    """Instantiate every EvalConfig, sharing one component cache."""
    components = _SharedComponents()
    configs: list[EvalConfig] = [
        DenseOnlyConfig(components),
        BM25OnlyConfig(components),
        HybridRRFConfig(components),
        HybridRerankedConfig(components),
        # v1 baseline — unmodified loop.
        AgenticConfig(components),
        # One-knob ablations.
        AgenticConfig(components, name="agentic_keep_orig", keep_original_query=True),
        AgenticConfig(components, name="agentic_rerank_orig", rerank_against_original=True),
        AgenticConfig(components, name="agentic_no_reflect", max_reflection_loops=0),
        AgenticConfig(components, name="agentic_faithful", faithful_decomposition=True),
        # Shipping candidate.
        AgenticConfig(
            components,
            name="agentic_v2",
            keep_original_query=True,
            rerank_against_original=True,
            faithful_decomposition=True,
        ),
    ]
    return configs


def config_names() -> list[str]:
    return [c.name for c in build_all_configs()]


def build_configs(names: list[str] | None = None) -> list[EvalConfig]:
    """Instantiate the named configs (all when `names` is None).

    Raises ValueError on an unknown name so a typo in `--configs` fails fast
    instead of silently running nothing.
    """
    all_configs = build_all_configs()
    if names is None:
        return all_configs
    by_name = {c.name: c for c in all_configs}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"unknown config(s): {unknown}; choose from {sorted(by_name)}")
    return [by_name[n] for n in names]
