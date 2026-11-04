"""Eval harness: runs the question set under multiple retrieval configs
and writes a CSV results table.

Configs benchmarked (see `medrag.eval.configs`):
    - dense_only       (baseline: Pinecone only, no rerank)
    - bm25_only        (sparse baseline)
    - hybrid_rrf       (BM25 + dense, RRF fused)
    - hybrid_reranked  (RRF + bge cross-encoder rerank)
    - agentic          (v1 LangGraph loop: decompose + reflect, unmodified)
    - agentic_*        (one-knob ablations of the loop)
    - agentic_v2       (shipping combination)

Per question and per config we record:
    - recall@10, MRR, nDCG@10 — retrieval quality vs gold PMIDs
    - faithfulness — claim-level grounding of the generated answer
    - input_tokens, output_tokens, cost_usd — LLM spend for this pass
    - latency_ms — wall-clock time for run()

Output:
    - data/eval_results.csv with one row per (config, question)
    - printed summary table aggregated by config (means + p50 latency)
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from medrag.agent import llm
from medrag.config import get_settings
from medrag.eval.configs import EvalConfig
from medrag.eval.metrics import (
    aggregate,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from medrag.eval.questions import QUESTIONS, EvalQuestion

# --------------------------------------------------------------------------- #
# Cost model                                                                  #
# --------------------------------------------------------------------------- #

# USD per 1M tokens, keyed by OpenRouter model id. Only used as a fallback:
# OpenRouter reports the actual cost on every response (``LLMResponse.cost_usd``)
# and the harness prefers that. Rough list prices as of 2026-11 — update as
# they change. Unknown models fall back to a conservative estimate so
# `cost_usd` is never silently zero.
_PRICING_USD_PER_M: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-v4.1-flash": (0.15, 0.60),
    "meta-llama/llama-3.3-70b-instruct": (0.59, 0.79),
    "meta-llama/llama-3.1-8b-instruct": (0.05, 0.08),
}
_FALLBACK_PRICING: tuple[float, float] = (0.60, 0.80)


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate $ cost of a single LLM call from token counts + model name."""
    in_rate, out_rate = _PRICING_USD_PER_M.get(model, _FALLBACK_PRICING)
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


# --------------------------------------------------------------------------- #
# Result row                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class EvalRow:
    config: str
    qid: str
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    faithfulness: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


def _unique_preserving_order(pmids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for pmid in pmids:
        if pmid in seen:
            continue
        seen.add(pmid)
        out.append(pmid)
    return out


_CSV_FIELDS: list[str] = [
    "config",
    "qid",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
    "faithfulness",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_ms",
]


# --------------------------------------------------------------------------- #
# Per-config loop                                                             #
# --------------------------------------------------------------------------- #


def evaluate_config(
    config: EvalConfig,
    questions: list[EvalQuestion],
) -> list[EvalRow]:
    # Imported here so the harness (and its unit tests) load without the
    # optional `eval` extra; ragas is only needed when a question is scored.
    from medrag.eval.faithfulness import compute_faithfulness

    rows: list[EvalRow] = []
    skipped = 0

    for q in questions:
        if not q.gold_pmids:
            skipped += 1
            continue

        t0 = time.perf_counter()
        with llm.track_usage() as usage:
            result = config.run(q.question)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Dedupe PMIDs while preserving rank order: retrievers return chunks,
        # not papers, and a paper with several relevant chunks would otherwise
        # crowd out unique papers in the top-10 and depress recall@10.
        retrieved_pmids = _unique_preserving_order([rc.paper.pmid for rc in result.retrieved])
        gold = set(q.gold_pmids)

        # Faithfulness is scored over the same top-k the answerer saw.
        settings = get_settings()
        top_k_contexts = result.retrieved[: settings.rerank_top_k]
        faithfulness = compute_faithfulness(
            question=q.question,
            answer_text=result.answer.text,
            contexts=top_k_contexts,
        )

        input_tokens = sum(u.input_tokens for u in usage)
        output_tokens = sum(u.output_tokens for u in usage)
        # Every LLM call in this question was made against a single model;
        # grab it off the last response (or fall back to configured default).
        model = usage[-1].model if usage else get_settings().llm_model
        # Prefer the provider-reported cost; estimate only if any call lacked it.
        if usage and all(u.cost_usd is not None for u in usage):
            cost_usd = sum(u.cost_usd or 0.0 for u in usage)
        else:
            cost_usd = estimate_cost_usd(input_tokens, output_tokens, model)

        rows.append(
            EvalRow(
                config=config.name,
                qid=q.qid,
                recall_at_10=recall_at_k(retrieved_pmids, gold, k=10),
                mrr=mean_reciprocal_rank(retrieved_pmids, gold),
                ndcg_at_10=ndcg_at_k(retrieved_pmids, gold, k=10),
                faithfulness=faithfulness,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
            )
        )

    if skipped:
        print(
            f"  [{config.name}] skipped {skipped} question(s) with no gold_pmids — "
            "curate questions.py before trusting these numbers."
        )
    return rows


# --------------------------------------------------------------------------- #
# I/O + summary                                                               #
# --------------------------------------------------------------------------- #


def write_results(rows: list[EvalRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            row_dict = asdict(r)
            # CSV: blank cell for "scorer failed" (None), not 0.0
            if row_dict["faithfulness"] is None:
                row_dict["faithfulness"] = ""
            writer.writerow(row_dict)


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(rows: list[EvalRow]) -> dict[str, dict[str, float]]:
    """Aggregate per-config means for the README results table."""
    by_config: dict[str, list[EvalRow]] = {}
    for r in rows:
        by_config.setdefault(r.config, []).append(r)

    summary: dict[str, dict[str, float]] = {}
    for cfg, cfg_rows in by_config.items():
        faith_vals = [r.faithfulness for r in cfg_rows if r.faithfulness is not None]
        summary[cfg] = {
            "recall_at_10": aggregate([r.recall_at_10 for r in cfg_rows]),
            "mrr": aggregate([r.mrr for r in cfg_rows]),
            "ndcg_at_10": aggregate([r.ndcg_at_10 for r in cfg_rows]),
            "faithfulness": aggregate(faith_vals) if faith_vals else 0.0,
            "cost_usd_mean": aggregate([r.cost_usd for r in cfg_rows]),
            "latency_ms_p50": _percentile([r.latency_ms for r in cfg_rows], 50),
            "latency_ms_p95": _percentile([r.latency_ms for r in cfg_rows], 95),
            "n": float(len(cfg_rows)),
        }
    return summary


def run_all(configs: list[EvalConfig], out_path: Path | None = None) -> None:
    settings = get_settings()
    all_rows: list[EvalRow] = []
    for cfg in configs:
        print(f"Evaluating {cfg.name}...")
        all_rows.extend(evaluate_config(cfg, QUESTIONS))

    out_path = out_path or settings.data_dir / "eval_results.csv"
    write_results(all_rows, out_path)
    print(f"\nWrote {len(all_rows)} rows to {out_path}")

    summary = summarize(all_rows)
    print("\nSummary (means unless noted):")
    for cfg_name, metrics in summary.items():
        metric_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"  {cfg_name}: {metric_str}")
