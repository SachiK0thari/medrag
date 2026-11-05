"""Turn an eval run into the tables quoted in the README and findings doc.

Reads the two files `scripts/run_eval.py` writes:

    data/eval_results.csv    one row per (config, question): metrics, cost, latency
    data/eval_traces.jsonl   per (config, question): ranked PMIDs + agent trace

and prints, as GitHub-flavoured markdown:

    1. The per-config summary table (recall, nDCG, faithfulness, cost, latency).
    2. The same retrieval metrics against a STRICT gold set (papers graded
       "directly answers" by `scripts/grade_gold.py`), with a paired bootstrap
       confidence interval on each config's recall difference from the
       baseline. With ~30 questions most differences are inside the noise;
       the interval says which ones are not.
    3. For each agentic config, how it differs from the hybrid_reranked
       baseline question by question, and WHERE gold papers were lost:
         - never surfaced: no sub-query returned the paper at all
           (the planner asked the wrong questions)
         - lost at merge: some sub-query returned it, but it fell out of the
           merged top-10 (the ranking across sub-queries is wrong)

No network, no models: this is pure post-processing, so every number in the
docs can be regenerated from the committed CSV in under a second.

Usage:
    python scripts/analyze_eval.py
    python scripts/analyze_eval.py --results data/eval_results.csv --baseline hybrid_reranked
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from medrag.config import DATA_DIR
from medrag.eval.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    paired_bootstrap_ci,
    recall_at_k,
)

K = 10


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def load_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            by_config[row["config"]].append(row)
    return by_config


def load_traces(path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["config"]][rec["qid"]] = rec
    return out


def summary_table(by_config: dict[str, list[dict[str, str]]]) -> str:
    lines = [
        "| Config | Recall@10 | MRR | nDCG@10 | Faithfulness | LLM calls | $/query | p50 latency | p95 latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg, rows in by_config.items():
        faith = [float(r["faithfulness"]) for r in rows if r.get("faithfulness")]
        faith_cell = f"{_mean(faith):.3f}" if faith else "—"
        if faith and len(faith) < len(rows):
            faith_cell += f" (n={len(faith)})"
        lat = [float(r["latency_ms"]) / 1000 for r in rows]
        calls = [float(r.get("llm_calls") or 0) for r in rows]
        lines.append(
            f"| {cfg} "
            f"| {_mean([float(r['recall_at_10']) for r in rows]):.3f} "
            f"| {_mean([float(r['mrr']) for r in rows]):.3f} "
            f"| {_mean([float(r['ndcg_at_10']) for r in rows]):.3f} "
            f"| {faith_cell} "
            f"| {_mean(calls):.1f} "
            f"| ${_mean([float(r['cost_usd']) for r in rows]):.5f} "
            f"| {_percentile(lat, 50):.1f}s "
            f"| {_percentile(lat, 95):.1f}s |"
        )
    return "\n".join(lines)


def health_checks(by_config: dict[str, list[dict[str, str]]]) -> list[str]:
    """Things that would make the numbers above untrustworthy."""
    problems: list[str] = []
    counts = {cfg: len(rows) for cfg, rows in by_config.items()}
    if len(set(counts.values())) > 1:
        problems.append(f"configs cover different numbers of questions: {counts}")
    for cfg, rows in by_config.items():
        trunc = sum(int(r.get("truncated_calls") or 0) for r in rows)
        if trunc:
            problems.append(f"{cfg}: {trunc} LLM call(s) hit max_tokens")
        blank = sum(1 for r in rows if not r.get("faithfulness"))
        if blank:
            problems.append(f"{cfg}: faithfulness missing on {blank}/{len(rows)} questions")
    return problems


def load_grades() -> dict[str, dict[str, int]]:
    """qid -> pmid -> grade (2 direct, 1 related, 0 irrelevant); {} if not graded yet."""
    try:
        from medrag.eval.gold_grades import GOLD_GRADES
    except ImportError:
        return {}
    return GOLD_GRADES


def gold_sets(
    traces: dict[str, dict[str, dict]], grades: dict[str, dict[str, int]], min_grade: int
) -> dict[str, set[str]]:
    """qid -> gold PMIDs with grade >= min_grade; questions left with none are dropped."""
    any_config = next(iter(traces.values()))
    out: dict[str, set[str]] = {}
    for qid, rec in any_config.items():
        gold = {p for p in rec["gold"] if grades.get(qid, {}).get(p, 1) >= min_grade}
        if gold:
            out[qid] = gold
    return out


def retrieval_table(
    traces: dict[str, dict[str, dict]], gold: dict[str, set[str]], baseline: str
) -> str:
    """Recall / MRR / nDCG recomputed from saved rankings against `gold`."""
    qids = sorted(gold)

    def recalls(cfg: str) -> list[float]:
        return [recall_at_k(traces[cfg][q]["ranked_pmids"], gold[q], K) for q in qids]

    base = recalls(baseline) if baseline in traces else None
    lines = [
        f"| Config | Recall@10 | MRR | nDCG@10 | Recall vs {baseline} (95% CI) |",
        "|---|---|---|---|---|",
    ]
    for cfg, recs in traces.items():
        rec = recalls(cfg)
        mrr = _mean([mean_reciprocal_rank(recs[q]["ranked_pmids"], gold[q]) for q in qids])
        ndcg = _mean([ndcg_at_k(recs[q]["ranked_pmids"], gold[q], K) for q in qids])
        if base is None or cfg == baseline:
            delta_cell = "—"
        else:
            deltas = [a - b for a, b in zip(rec, base, strict=True)]
            lo, hi = paired_bootstrap_ci(deltas)
            verdict = "" if lo <= 0.0 <= hi else " **significant**"
            delta_cell = f"{_mean(deltas):+.3f} ({lo:+.3f} to {hi:+.3f}){verdict}"
        lines.append(f"| {cfg} | {_mean(rec):.3f} | {mrr:.3f} | {ndcg:.3f} | {delta_cell} |")
    return "\n".join(lines)


def attribution(traces: dict[str, dict[str, dict]], baseline: str) -> str:
    base = traces.get(baseline, {})
    lines = [
        "| Config | Mean sub-queries | Single-query questions | Follow-ups | Worse / better than baseline | Gold never surfaced | Gold lost at merge |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg, recs in traces.items():
        if not any(r["trace"] for r in recs.values()):
            continue  # non-agentic config
        n_sub: list[float] = []
        single = followups = worse = better = never = lost_merge = total_gold = 0
        empty_reflect = 0
        for qid, rec in recs.items():
            gold = set(rec["gold"])
            total_gold += len(gold)
            final = set(rec["ranked_pmids"][:K]) & gold
            retrieve_steps = [e for e in rec["trace"] if e["step"] == "retrieve"]
            decompose = next((e for e in rec["trace"] if e["step"] == "decompose"), None)
            planned = len(decompose["sub_queries"]) if decompose else len(retrieve_steps)
            n_sub.append(planned)
            single += planned == 1
            reflects = [e for e in rec["trace"] if e["step"] == "reflect"]
            followups += sum(1 for e in reflects if not e["sufficient"] and e["missing"])
            empty_reflect += sum(1 for e in reflects if e.get("empty_reply"))
            surfaced: set[str] = set()
            for e in retrieve_steps:
                surfaced |= gold & set(e["top_pmids"])
            never += len(gold - surfaced)
            lost_merge += len(surfaced - final)
            if qid in base:
                b = set(base[qid]["ranked_pmids"][:K]) & gold
                worse += len(final) < len(b)
                better += len(final) > len(b)
        note = f" ({empty_reflect} empty reflect replies)" if empty_reflect else ""
        lines.append(
            f"| {cfg} | {_mean(n_sub):.2f} | {single}/{len(recs)} | {followups}{note} "
            f"| {worse} / {better} | {never}/{total_gold} | {lost_merge}/{total_gold} |"
        )
    return "\n".join(lines)


def per_question_diffs(traces: dict[str, dict[str, dict]], baseline: str, config: str) -> str:
    base, recs = traces.get(baseline, {}), traces.get(config, {})
    out: list[str] = []
    for qid in sorted(recs):
        if qid not in base:
            continue
        gold = set(recs[qid]["gold"])
        b = set(base[qid]["ranked_pmids"][:K]) & gold
        a = set(recs[qid]["ranked_pmids"][:K]) & gold
        if a != b:
            decompose = next((e for e in recs[qid]["trace"] if e["step"] == "decompose"), None)
            subs = decompose["sub_queries"] if decompose else []
            out.append(
                f"- {qid}: lost {sorted(b - a)}, gained {sorted(a - b)}; sub-queries: {subs}"
            )
    return "\n".join(out) if out else "- identical gold hits on every question"


def main() -> None:
    # Paths come from the module constant, not Settings: Settings requires API
    # keys, and this script has to run from a fresh clone that has none.
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=DATA_DIR / "eval_results.csv")
    parser.add_argument("--traces", type=Path, default=DATA_DIR / "eval_traces.jsonl")
    parser.add_argument("--baseline", default="hybrid_reranked")
    parser.add_argument("--diff", default="agentic_v2", help="Config to diff against baseline.")
    args = parser.parse_args()

    by_config = load_rows(args.results)
    traces = load_traces(args.traces)
    n = len(next(iter(by_config.values())))

    print(f"## Summary ({n} questions per config)\n")
    print(summary_table(by_config))

    problems = health_checks(by_config)
    print("\n## Health checks\n")
    print("\n".join(f"- {p}" for p in problems) if problems else "- all clear")

    grades = load_grades()
    if traces:
        lenient = gold_sets(
            traces, grades, min_grade=0
        )  # every curated label, as the harness scores
        n_lenient = sum(len(v) for v in lenient.values())
        print(
            f"\n## Retrieval vs lenient gold: all curated labels "
            f"({len(lenient)} questions, {n_lenient} papers)\n"
        )
        print(retrieval_table(traces, lenient, args.baseline))
        if grades:
            strict = gold_sets(traces, grades, min_grade=2)
            n_strict = sum(len(v) for v in strict.values())
            print(
                f"\n## Retrieval vs strict gold: directly answers "
                f"({len(strict)} questions, {n_strict} papers)\n"
            )
            print(retrieval_table(traces, strict, args.baseline))
        else:
            print("\n(no graded gold yet: run scripts/grade_gold.py for the strict table)")

    if traces:
        print(f"\n## Agentic attribution (baseline: {args.baseline})\n")
        print(attribution(traces, args.baseline))
        print(f"\n## {args.diff} vs {args.baseline}, question by question\n")
        print(per_question_diffs(traces, args.baseline, args.diff))


if __name__ == "__main__":
    main()
