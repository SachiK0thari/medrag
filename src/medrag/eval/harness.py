"""Eval harness: runs the question set under multiple retrieval configs
and writes a CSV results table.

Configs benchmarked:
    - dense_only       (baseline: Pinecone only, no rerank)
    - bm25_only        (sparse baseline)
    - hybrid_rrf       (BM25 + dense, RRF fused)
    - hybrid_reranked  (RRF + bge cross-encoder rerank)
    - agentic          (full LangGraph loop with decompose + reflect)

Output: data/eval_results.csv with one row per (config, question).
Plus a summary table aggregated by config.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from medrag.config import get_settings
from medrag.eval.metrics import (
    aggregate,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from medrag.eval.questions import QUESTIONS, EvalQuestion
from medrag.models import RetrievedChunk


class RetrievalConfig(Protocol):
    name: str

    def retrieve(self, question: str) -> list[RetrievedChunk]: ...


@dataclass
class EvalRow:
    config: str
    qid: str
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    latency_ms: float


def evaluate_config(
    config: RetrievalConfig,
    questions: list[EvalQuestion],
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for q in questions:
        if not q.gold_pmids:
            continue  # skip uncurated questions
        t0 = time.perf_counter()
        retrieved = config.retrieve(q.question)
        latency_ms = (time.perf_counter() - t0) * 1000
        retrieved_pmids = [rc.paper.pmid for rc in retrieved]
        gold = set(q.gold_pmids)
        rows.append(
            EvalRow(
                config=config.name,
                qid=q.qid,
                recall_at_10=recall_at_k(retrieved_pmids, gold, k=10),
                mrr=mean_reciprocal_rank(retrieved_pmids, gold),
                ndcg_at_10=ndcg_at_k(retrieved_pmids, gold, k=10),
                latency_ms=latency_ms,
            )
        )
    return rows


def write_results(rows: list[EvalRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["config", "qid", "recall_at_10", "mrr", "ndcg_at_10", "latency_ms"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)


def summarize(rows: list[EvalRow]) -> dict[str, dict[str, float]]:
    """Aggregate per-config means for the README results table."""
    by_config: dict[str, list[EvalRow]] = {}
    for r in rows:
        by_config.setdefault(r.config, []).append(r)
    return {
        cfg: {
            "recall_at_10": aggregate([r.recall_at_10 for r in cfg_rows]),
            "mrr": aggregate([r.mrr for r in cfg_rows]),
            "ndcg_at_10": aggregate([r.ndcg_at_10 for r in cfg_rows]),
            "latency_ms_p50": _percentile([r.latency_ms for r in cfg_rows], 50),
            "latency_ms_p95": _percentile([r.latency_ms for r in cfg_rows], 95),
            "n": len(cfg_rows),
        }
        for cfg, cfg_rows in by_config.items()
    }


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def run_all(configs: list[RetrievalConfig]) -> None:
    settings = get_settings()
    all_rows: list[EvalRow] = []
    for cfg in configs:
        print(f"Evaluating {cfg.name}...")
        all_rows.extend(evaluate_config(cfg, QUESTIONS))
    out_path = settings.data_dir / "eval_results.csv"
    write_results(all_rows, out_path)
    summary = summarize(all_rows)
    print("\nSummary:")
    for cfg_name, metrics in summary.items():
        print(f"  {cfg_name}: {metrics}")
