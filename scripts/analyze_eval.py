"""Turn an eval run into the tables quoted in the README and findings doc.

Reads the two files `scripts/run_eval.py` writes:

    data/eval_results.csv    one row per (config, question): metrics, cost, latency
    data/eval_traces.jsonl   per (config, question): the answer's context, the
                             whole ranking, the agent trace and the answer

plus any repeat runs next to them (`eval_results_rep2.csv` and
`eval_traces_rep2.jsonl`, and so on). The agentic configs are not
deterministic, so the ones the conclusions rest on are run several times; a
config's score on a question is its mean over the runs that include it.

Retrieval is scored on the answer's context: the distinct papers behind the
top-10 chunks the answer was written from. Traces from before that was
recorded fall back to the first 10 distinct papers of the config's ranking.

Prints, as GitHub-flavoured markdown:

    1. The per-config summary table (recall, nDCG, faithfulness, cost, latency).
    2. The retrieval metrics against the curated gold labels and against the
       STRICT subset (papers graded "directly answers" by
       `scripts/grade_gold.py`), with a paired bootstrap confidence interval on
       each config's recall difference from the baseline. With ~30 questions
       most differences are inside the noise; the interval says which are not.
    3. If `scripts/extend_gold.py` has run: what the papers in each config's
       context are (gold, judged and passed over, never judged), how the judge
       graded the ones without a gold label, and the retrieval metrics again
       with every paper graded direct counted as gold.
    4. How many answers quote a 95% confidence interval and how many say the
       sources lack the information (pattern counts over the answer text).
    5. For each agentic config, how it differs from the hybrid_reranked
       baseline question by question, and WHERE gold papers were lost:
         - never surfaced: no sub-query returned the paper at all
           (the planner asked the wrong questions)
         - lost at merge: some sub-query returned it, but it fell out of the
           answer's context (the ranking across sub-queries is wrong)

Sections 3 (the first table), 4 and 5 describe the main run only.

No network, no models: this is pure post-processing, so every number in the
docs can be regenerated from the committed files in under a second.

Usage:
    python scripts/analyze_eval.py
    python scripts/analyze_eval.py --baseline agentic        # deltas against another config

    # The earlier run on the abstracts-only corpus, as analysed in docs/eval_findings.md:
    python scripts/analyze_eval.py --results data/eval_results_abstracts.csv \
        --traces data/eval_traces_abstracts.jsonl --no-extension
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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

# Answer-text patterns. Crude on purpose: they count strings, not correctness.
_QUOTES_CI_RE = re.compile(r"95\s?%\s?(?:CI|confidence interval)", re.IGNORECASE)
_SOURCES_LACK_RE = re.compile(
    r"(?:do|does) not (?:contain|report|provide|state|include|specify)"
    r"|not (?:reported|provided|stated|given)"
    r"|cannot be (?:given|determined|answered)",
    re.IGNORECASE,
)


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


Traces = dict[str, dict[str, dict]]  # config -> qid -> trace record


def load_rows(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    """Rows of every run, pooled by config."""
    by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                by_config[row["config"]].append(row)
    return by_config


def load_traces(path: Path) -> Traces:
    out: Traces = defaultdict(dict)
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["config"]][rec["qid"]] = rec
    return out


def repeat_paths(path: Path) -> list[Path]:
    """Repeat-run files that sit next to `path`: <stem>_rep2<suffix>, <stem>_rep3<suffix>, ..."""
    return sorted(path.parent.glob(f"{path.stem}_rep*{path.suffix}"))


def scored(rec: dict) -> list[str]:
    """The distinct papers a config is scored on for one question, in rank order."""
    if "context_pmids" in rec:
        return list(rec["context_pmids"])
    return list(rec["ranked_pmids"][:K])  # traces older than the context field


def runs_of(runs: list[Traces], cfg: str) -> list[dict[str, dict]]:
    return [run[cfg] for run in runs if cfg in run]


def per_run(runs: list[Traces], cfg: str, gold: dict[str, set[str]]) -> list[float]:
    """`cfg`'s mean recall over the questions of `gold`, one value per run."""
    return [
        _mean([recall_at_k(scored(recs[qid]), gold[qid], K) for qid in sorted(gold)])
        for recs in runs_of(runs, cfg)
    ]


def per_question(
    runs: list[Traces], cfg: str, gold: dict[str, set[str]], metric: str
) -> list[float]:
    """`cfg`'s score on each question of `gold` (sorted by qid), averaged over its runs."""
    fn = {
        "recall": lambda ranked, g: recall_at_k(ranked, g, K),
        "mrr": mean_reciprocal_rank,
        "ndcg": lambda ranked, g: ndcg_at_k(ranked, g, K),
        "n_papers": lambda ranked, g: float(len(ranked)),
        "hits": lambda ranked, g: float(len(set(ranked) & g)),
    }[metric]
    return [
        _mean([fn(scored(recs[qid]), gold[qid]) for recs in runs_of(runs, cfg)])
        for qid in sorted(gold)
    ]


def summary_table(by_config: dict[str, list[dict[str, str]]]) -> str:
    lines = [
        "| Config | Runs | Recall@10 | MRR | nDCG@10 | Faithfulness | LLM calls | $/query | p50 latency | p95 latency |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    n_questions = len({r["qid"] for rows in by_config.values() for r in rows})
    for cfg, rows in by_config.items():
        faith = [float(r["faithfulness"]) for r in rows if r.get("faithfulness")]
        faith_cell = f"{_mean(faith):.3f}" if faith else "—"
        if faith and len(faith) < len(rows):
            faith_cell += f" (n={len(faith)})"
        lat = [float(r["latency_ms"]) / 1000 for r in rows]
        calls = [float(r.get("llm_calls") or 0) for r in rows]
        lines.append(
            f"| {cfg} "
            f"| {len(rows) // n_questions} "
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
    counts = {cfg: len({r["qid"] for r in rows}) for cfg, rows in by_config.items()}
    if len(set(counts.values())) > 1:
        problems.append(f"configs cover different numbers of questions: {counts}")
    for cfg, rows in by_config.items():
        trunc = sum(int(r.get("truncated_calls") or 0) for r in rows)
        if trunc:
            problems.append(f"{cfg}: {trunc} LLM call(s) hit max_tokens")
        blank = sum(1 for r in rows if not r.get("faithfulness"))
        if blank:
            problems.append(f"{cfg}: faithfulness missing on {blank}/{len(rows)} answers")
    return problems


def load_grades() -> dict[str, dict[str, int]]:
    """qid -> pmid -> grade (2 direct, 1 related, 0 irrelevant); {} if not graded yet."""
    try:
        from medrag.eval.gold_grades import GOLD_GRADES
    except ImportError:
        return {}
    return GOLD_GRADES


def load_extension() -> tuple[dict[str, dict[str, int]], dict[str, set[str]]]:
    """Grades for papers judged after the re-ingest, and the pool judged before it.

    Returns (qid -> pmid -> grade, qid -> PMIDs the original curation judged);
    both empty if `scripts/extend_gold.py` has never run.
    """
    try:
        from medrag.eval.gold_extension import EXTENSION_GRADES, ORIGINAL_POOL
    except ImportError:
        return {}, {}
    return EXTENSION_GRADES, {qid: set(pmids) for qid, pmids in ORIGINAL_POOL.items()}


def with_extension(
    gold: dict[str, set[str]], extension: dict[str, dict[str, int]]
) -> dict[str, set[str]]:
    """`gold` plus every paper graded direct since; a question can enter the set this way."""
    out = {qid: set(pmids) for qid, pmids in gold.items()}
    for qid, grades in extension.items():
        direct = {pmid for pmid, grade in grades.items() if grade == 2}
        if direct:
            out.setdefault(qid, set()).update(direct)
    return out


def extension_breakdown(
    extension: dict[str, dict[str, int]],
    original_pool: dict[str, set[str]],
    gold: dict[str, set[str]],
) -> str:
    """The extension's grades by what the original curation had done with each paper."""
    kinds = ("Never judged", "Judged, passed over", 'Gold, graded "related" on the abstract')
    counts = {kind: {2: 0, 1: 0, 0: 0} for kind in kinds}
    for qid, grades in extension.items():
        for pmid, grade in grades.items():
            if pmid in gold.get(qid, set()):
                kind = kinds[2]
            elif pmid in original_pool.get(qid, set()):
                kind = kinds[1]
            else:
                kind = kinds[0]
            counts[kind][grade] += 1
    lines = [
        "| Before the extension | Papers | Graded direct | related | irrelevant |",
        "|---|---|---|---|---|",
    ]
    for kind in kinds:
        c = counts[kind]
        lines.append(f"| {kind} | {sum(c.values())} | {c[2]} | {c[1]} | {c[0]} |")
    return "\n".join(lines)


def context_table(
    traces: Traces,
    extension: dict[str, dict[str, int]],
    original_pool: dict[str, set[str]],
) -> str:
    """What the papers in each config's context are, and how the unlabelled ones grade."""
    lines = [
        "| Config | Papers in context | Gold | Judged, passed over | Never judged "
        "| Non-gold graded direct | related | irrelevant | not graded |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg, recs in traces.items():
        n = len(recs)
        total = gold_n = passed = never = 0
        counts = {2: 0, 1: 0, 0: 0, None: 0}
        for qid, rec in recs.items():
            gold = set(rec["gold"])
            for pmid in scored(rec):
                total += 1
                if pmid in gold:
                    gold_n += 1
                    continue
                if pmid in original_pool.get(qid, set()):
                    passed += 1
                else:
                    never += 1
                counts[extension.get(qid, {}).get(pmid)] += 1
        lines.append(
            f"| {cfg} | {total / n:.1f} | {gold_n / n:.1f} | {passed / n:.1f} | {never / n:.1f} "
            f"| {counts[2]} | {counts[1]} | {counts[0]} | {counts[None]} |"
        )
    return "\n".join(lines)


def gold_sets(
    traces: Traces, grades: dict[str, dict[str, int]], min_grade: int
) -> dict[str, set[str]]:
    """qid -> gold PMIDs with grade >= min_grade; questions left with none are dropped."""
    any_config = next(iter(traces.values()))
    out: dict[str, set[str]] = {}
    for qid, rec in any_config.items():
        gold = {p for p in rec["gold"] if grades.get(qid, {}).get(p, 1) >= min_grade}
        if gold:
            out[qid] = gold
    return out


def retrieval_table(runs: list[Traces], gold: dict[str, set[str]], baseline: str) -> str:
    """Recall / MRR / nDCG recomputed from the saved contexts against `gold`.

    A config with several runs is scored per question on the mean of its runs,
    and the last column lists its mean recall in each run, which is the
    run-to-run noise in plain sight.
    """
    configs = list(dict.fromkeys(cfg for run in runs for cfg in run))
    base = per_question(runs, baseline, gold, "recall") if baseline in configs else None
    lines = [
        f"| Config | Runs | Papers in context | Gold among them | Recall@10 | MRR | nDCG@10 "
        f"| Recall vs {baseline} (95% CI) | Recall, each run |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg in configs:
        rec = per_question(runs, cfg, gold, "recall")
        if base is None or cfg == baseline:
            delta_cell = "—"
        else:
            deltas = [a - b for a, b in zip(rec, base, strict=True)]
            lo, hi = paired_bootstrap_ci(deltas)
            verdict = "" if lo <= 0.0 <= hi else " **significant**"
            delta_cell = f"{_mean(deltas):+.3f} ({lo:+.3f} to {hi:+.3f}){verdict}"
        each_run = per_run(runs, cfg, gold)
        lines.append(
            f"| {cfg} | {len(each_run)} "
            f"| {_mean(per_question(runs, cfg, gold, 'n_papers')):.1f} "
            f"| {_mean(per_question(runs, cfg, gold, 'hits')):.2f} "
            f"| {_mean(rec):.3f} "
            f"| {_mean(per_question(runs, cfg, gold, 'mrr')):.3f} "
            f"| {_mean(per_question(runs, cfg, gold, 'ndcg')):.3f} "
            f"| {delta_cell} "
            f"| {', '.join(f'{r:.3f}' for r in each_run) if len(each_run) > 1 else '—'} |"
        )
    return "\n".join(lines)


def answers_table(traces: Traces) -> str:
    """How often an answer quotes an effect estimate, and how often it gives up."""
    lines = [
        "| Config | Answers quoting a 95% CI | Answers saying the sources lack something "
        "| Mean words | Mean citations |",
        "|---|---|---|---|---|",
    ]
    for cfg, recs in traces.items():
        answers = [rec["answer"] for rec in recs.values()]
        n = len(answers)
        lines.append(
            f"| {cfg} | {sum(bool(_QUOTES_CI_RE.search(a)) for a in answers)}/{n} "
            f"| {sum(bool(_SOURCES_LACK_RE.search(a)) for a in answers)}/{n} "
            f"| {_mean([float(len(a.split())) for a in answers]):.0f} "
            f"| {_mean([float(rec['n_citations']) for rec in recs.values()]):.1f} |"
        )
    return "\n".join(lines)


def attribution(traces: Traces, baseline: str) -> str:
    base = traces.get(baseline, {})
    total = sum(len(rec["gold"]) for rec in base.values())
    in_top = sum(len(set(scored(rec)) & set(rec["gold"])) for rec in base.values())
    lines = [
        "| Config | Mean sub-queries | Single-query questions | Follow-ups | Worse / better than baseline | Gold never surfaced | Gold lost at merge | Gold in the answer's context |",
        "|---|---|---|---|---|---|---|---|",
        f"| {baseline} (no agent) | — | — | — | — | — | — | {in_top}/{total} |",
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
            final = set(scored(rec)) & gold
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
                b = set(scored(base[qid])) & gold
                worse += len(final) < len(b)
                better += len(final) > len(b)
        note = f" ({empty_reflect} empty reflect replies)" if empty_reflect else ""
        lines.append(
            f"| {cfg} | {_mean(n_sub):.2f} | {single}/{len(recs)} | {followups}{note} "
            f"| {worse} / {better} | {never}/{total_gold} | {lost_merge}/{total_gold} "
            f"| {total_gold - never - lost_merge}/{total_gold} |"
        )
    return "\n".join(lines)


def per_question_diffs(traces: Traces, baseline: str, config: str) -> str:
    base, recs = traces.get(baseline, {}), traces.get(config, {})
    out: list[str] = []
    for qid in sorted(recs):
        if qid not in base:
            continue
        gold = set(recs[qid]["gold"])
        b = set(scored(base[qid])) & gold
        a = set(scored(recs[qid])) & gold
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
    parser.add_argument(
        "--no-extension",
        action="store_true",
        help="Score against the original gold only (for a run that predates the full-text "
        "re-ingest: the extension grades papers on body passages that run could not retrieve).",
    )
    args = parser.parse_args()

    by_config = load_rows([args.results, *repeat_paths(args.results)])
    traces = load_traces(args.traces)
    runs = [traces, *(load_traces(path) for path in repeat_paths(args.traces))]
    n = len({r["qid"] for r in next(iter(by_config.values()))})

    print(f"## Summary ({n} questions per config; {len(runs)} run(s) on file)\n")
    print(summary_table(by_config))

    problems = health_checks(by_config)
    print("\n## Health checks\n")
    print("\n".join(f"- {p}" for p in problems) if problems else "- all clear")

    grades = load_grades()
    if traces:
        lenient = gold_sets(traces, grades, min_grade=0)  # every curated label
        n_lenient = sum(len(v) for v in lenient.values())
        print(
            f"\n## Retrieval vs lenient gold: all curated labels "
            f"({len(lenient)} questions, {n_lenient} papers)\n"
        )
        print(retrieval_table(runs, lenient, args.baseline))
        if grades:
            strict = gold_sets(traces, grades, min_grade=2)
            n_strict = sum(len(v) for v in strict.values())
            print(
                f"\n## Retrieval vs strict gold: directly answers "
                f"({len(strict)} questions, {n_strict} papers)\n"
            )
            print(retrieval_table(runs, strict, args.baseline))
        else:
            strict = {}
            print("\n(no graded gold yet: run scripts/grade_gold.py for the strict table)")

        extension, original_pool = ({}, {}) if args.no_extension else load_extension()
        if extension:
            print("\n## What is in each config's context (main run)\n")
            print(context_table(traces, extension, original_pool))
            by_grade = {
                g: sum(v == g for papers in extension.values() for v in papers.values())
                for g in (2, 1, 0)
            }
            print(
                f"\nscripts/extend_gold.py has graded {sum(by_grade.values())} papers: "
                f"{by_grade[2]} direct, {by_grade[1]} related, {by_grade[0]} irrelevant.\n"
            )
            print(extension_breakdown(extension, original_pool, lenient))
            for title, base in (("lenient", lenient), ("strict", strict)):
                extended = with_extension(base, extension)
                n_papers = sum(len(v) for v in extended.values())
                print(
                    f"\n## Retrieval vs extended gold: {title} plus every paper graded direct "
                    f"since ({len(extended)} questions, {n_papers} papers)\n"
                )
                print(retrieval_table(runs, extended, args.baseline))

    if traces:
        print("\n## Answers (main run)\n")
        print(answers_table(traces))
        print(f"\n## Agentic attribution (main run; baseline: {args.baseline})\n")
        print(attribution(traces, args.baseline))
        print(f"\n## {args.diff} vs {args.baseline}, question by question\n")
        print(per_question_diffs(traces, args.baseline, args.diff))


if __name__ == "__main__":
    main()
