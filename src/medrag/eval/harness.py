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
    - recall@10, MRR, nDCG@10 — retrieval quality vs gold PMIDs, scored on the
      answer's context: the papers behind the top-k chunks the answer was
      written from (k = `rerank_top_k`, 10)
    - faithfulness — claim-level grounding of the generated answer
    - input_tokens, output_tokens, cost_usd — LLM spend for this pass
    - latency_ms — wall-clock time for run()

Output:
    - data/eval_results.csv with one row per (config, question)
    - data/eval_traces.jsonl with, per (config, question), the PMIDs and chunk
      IDs of the answer's context, the config's whole ranking as distinct
      PMIDs, the agent's sub-queries and per-step trace, and the answer text —
      the raw material for `scripts/analyze_eval.py`
    - printed summary table aggregated by config (means + p50 latency)

Both files are checkpointed after every config, so a crash late in a long
run keeps everything finished so far. `run_all(..., keep_existing=True)` then
re-runs only the named configs and keeps the other configs' rows and traces.
"""

from __future__ import annotations

import csv
import json
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
    llm_calls: int = 0
    # Calls whose output hit max_tokens. Should be 0; anything else means an
    # answer or a planner reply was cut off and the row is suspect.
    truncated_calls: int = 0


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
    "llm_calls",
    "truncated_calls",
]


# --------------------------------------------------------------------------- #
# Per-config loop                                                             #
# --------------------------------------------------------------------------- #


def evaluate_config(
    config: EvalConfig,
    questions: list[EvalQuestion],
    trace_path: Path | None = None,
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

        gold = set(q.gold_pmids)

        # Retrieval is scored on what the answerer saw: the top-k chunks. A
        # gold paper ranked below them cannot be cited, however the config
        # ranked it. Scoring the first 10 distinct papers of each config's
        # whole ranking instead compared lists of different lengths (10 chunks
        # are about 8 papers on the full-text corpus, and some configs return
        # 10 chunks while others return 30 or more).
        settings = get_settings()
        top_k_contexts = result.retrieved[: settings.rerank_top_k]
        context_pmids = _unique_preserving_order([rc.paper.pmid for rc in top_k_contexts])
        # The whole ranking, as distinct papers, kept in the trace for analysis.
        retrieved_pmids = _unique_preserving_order([rc.paper.pmid for rc in result.retrieved])

        # Faithfulness is scored over the same top-k.
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
                recall_at_10=recall_at_k(context_pmids, gold, k=10),
                mrr=mean_reciprocal_rank(context_pmids, gold),
                ndcg_at_10=ndcg_at_k(context_pmids, gold, k=10),
                faithfulness=faithfulness,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                llm_calls=len(usage),
                truncated_calls=sum(1 for u in usage if u.truncated),
            )
        )
        if trace_path is not None:
            record = {
                "config": config.name,
                "qid": q.qid,
                "question": q.question,
                "gold": sorted(gold),
                "context_pmids": context_pmids,
                "context_chunk_ids": [rc.chunk.chunk_id for rc in top_k_contexts],
                "ranked_pmids": retrieved_pmids,
                "trace": result.trace,
                "answer": result.answer.text,
                "n_citations": len(result.answer.citations),
                "faithfulness": faithfulness,
            }
            with trace_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

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


def read_results(path: Path) -> list[EvalRow]:
    """Load rows written by `write_results` (blank faithfulness -> None)."""
    with path.open(newline="") as f:
        return [
            EvalRow(
                config=r["config"],
                qid=r["qid"],
                recall_at_10=float(r["recall_at_10"]),
                mrr=float(r["mrr"]),
                ndcg_at_10=float(r["ndcg_at_10"]),
                faithfulness=float(r["faithfulness"]) if r.get("faithfulness") else None,
                input_tokens=int(r["input_tokens"]),
                output_tokens=int(r["output_tokens"]),
                cost_usd=float(r["cost_usd"]),
                latency_ms=float(r["latency_ms"]),
                llm_calls=int(r.get("llm_calls") or 0),
                truncated_calls=int(r.get("truncated_calls") or 0),
            )
            for r in csv.DictReader(f)
        ]


def _load_kept(out_path: Path, trace_path: Path, rerun: set[str]) -> tuple[list[EvalRow], str]:
    """Rows and trace lines from a previous run for configs NOT being re-run."""
    rows = [r for r in read_results(out_path) if r.config not in rerun] if out_path.exists() else []
    kept_lines: list[str] = []
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            if line.strip() and json.loads(line)["config"] not in rerun:
                kept_lines.append(line)
    return rows, "".join(line + "\n" for line in kept_lines)


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


def run_all(
    configs: list[EvalConfig],
    out_path: Path | None = None,
    *,
    keep_existing: bool = False,
) -> None:
    """Evaluate `configs` and write the CSV + traces.

    With `keep_existing`, rows and traces already on disk for configs that are
    not in `configs` are preserved, so a subset can be re-run (after a crash or
    a fix) without repeating the whole benchmark.
    """
    settings = get_settings()
    out_path = out_path or settings.data_dir / "eval_results.csv"
    trace_path = out_path.with_name(out_path.stem.replace("results", "traces") + ".jsonl")
    if trace_path == out_path:  # --out without "results" in the name
        trace_path = out_path.with_suffix(".traces.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[EvalRow] = []
    kept_traces = ""
    if keep_existing:
        all_rows, kept_traces = _load_kept(out_path, trace_path, {c.name for c in configs})
        kept = sorted({r.config for r in all_rows})
        print(f"Keeping {len(all_rows)} existing rows for: {', '.join(kept) or 'none'}", flush=True)
    trace_path.write_text(kept_traces)  # evaluate_config appends

    for cfg in configs:
        print(f"Evaluating {cfg.name}...", flush=True)
        t0 = time.perf_counter()
        all_rows.extend(evaluate_config(cfg, QUESTIONS, trace_path=trace_path))
        write_results(all_rows, out_path)  # checkpoint after every config
        print(f"  done in {time.perf_counter() - t0:.0f}s; checkpointed", flush=True)

    print(f"\nWrote {len(all_rows)} rows to {out_path}")
    print(f"Wrote traces to {trace_path}")

    summary = summarize(all_rows)
    print("\nSummary (means unless noted):")
    for cfg_name, metrics in summary.items():
        metric_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"  {cfg_name}: {metric_str}")
