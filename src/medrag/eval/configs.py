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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from medrag.agent.graph import MedRagAgent, generate_answer
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
        self._agent: MedRagAgent | None = None

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

    def agent(self) -> MedRagAgent:
        if self._agent is None:
            self._agent = MedRagAgent(
                hybrid=self.hybrid(),
                reranker=self.reranker(),
                store=self.store(),
            )
        return self._agent


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
    name = "agentic"

    def __init__(self, components: _SharedComponents) -> None:
        self._c = components

    def run(self, question: str) -> EvalResult:
        retrieved, answer = self._c.agent().run(question)
        return EvalResult(retrieved=retrieved, answer=answer)


def build_all_configs() -> list[EvalConfig]:
    """Instantiate every EvalConfig, sharing one component cache."""
    components = _SharedComponents()
    configs: list[EvalConfig] = [
        DenseOnlyConfig(components),
        BM25OnlyConfig(components),
        HybridRRFConfig(components),
        HybridRerankedConfig(components),
        AgenticConfig(components),
    ]
    return configs
