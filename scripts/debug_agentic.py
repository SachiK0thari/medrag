"""Attribute the agentic loop's retrieval misses to specific planner outputs.

For every gold-labelled eval question, runs the hybrid-reranked baseline and
one or more agentic variants, then records per question:

    - gold PMIDs
    - baseline top-10 PMIDs and recall@10
    - per variant: the sub-queries the planner produced, which PMIDs each
      sub-query surfaced, the merged top-10, recall@10, and which gold PMIDs
      the baseline found but the agent lost (or vice versa)

Writes JSONL to data/agentic_debug.jsonl (gitignored) and prints a short
per-variant summary of recall deltas. Read the worst questions by hand; the
"lost" list plus the sub-query texts usually make the failure mode obvious.

Usage:
    python scripts/debug_agentic.py                              # agentic (v1) only
    python scripts/debug_agentic.py --variants agentic,agentic_v2
    python scripts/debug_agentic.py --qids q01,q05,q08,q17,q20,q21
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medrag.config import get_settings
from medrag.eval.configs import (
    AgenticConfig,
    HybridRerankedConfig,
    _SharedComponents,
    build_all_configs,
)
from medrag.eval.metrics import recall_at_k
from medrag.eval.questions import QUESTIONS

_LIMIT = 10


def _unique(pmids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in pmids:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="agentic", help="Comma-separated agentic_* names.")
    parser.add_argument("--qids", help="Comma-separated qids to restrict to.")
    parser.add_argument("--out", type=Path, default=get_settings().data_dir / "agentic_debug.jsonl")
    args = parser.parse_args()

    wanted = {n.strip() for n in args.variants.split(",")}
    variants = [c for c in build_all_configs() if isinstance(c, AgenticConfig) and c.name in wanted]
    missing = wanted - {v.name for v in variants}
    if missing:
        raise SystemExit(f"unknown agentic variant(s): {sorted(missing)}")

    # Share one retrieval stack between baseline and every variant.
    components = _SharedComponents()
    baseline = HybridRerankedConfig(components)
    for v in variants:
        v._c = components

    qids = {q.strip() for q in args.qids.split(",")} if args.qids else None
    questions = [q for q in QUESTIONS if q.gold_pmids and (qids is None or q.qid in qids)]

    deltas: dict[str, list[float]] = {v.name: [] for v in variants}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for q in questions:
            gold = set(q.gold_pmids)
            base = baseline.run(q.question)
            base_pmids = _unique([rc.paper.pmid for rc in base.retrieved])[:_LIMIT]
            base_recall = recall_at_k(base_pmids, gold, _LIMIT)
            record: dict[str, object] = {
                "qid": q.qid,
                "question": q.question,
                "gold": sorted(gold),
                "baseline": {"top_pmids": base_pmids, "recall_at_10": base_recall},
                "variants": {},
            }
            print(f"\n{q.qid}: {q.question}")
            print(f"  baseline recall@10={base_recall:.2f}  gold={sorted(gold)}")
            for v in variants:
                retrieved, _answer, trace = v.agent().run_with_trace(q.question)
                pmids = _unique([rc.paper.pmid for rc in retrieved])[:_LIMIT]
                rec = recall_at_k(pmids, gold, _LIMIT)
                deltas[v.name].append(rec - base_recall)
                found_by: dict[str, list[str]] = {}
                for entry in trace:
                    if entry.get("step") == "retrieve":
                        hits = [p for p in entry.get("top_pmids", []) if p in gold]  # type: ignore[union-attr]
                        if hits:
                            found_by[str(entry["sub_query"])] = hits
                lost = sorted((gold & set(base_pmids)) - set(pmids))
                gained = sorted((gold & set(pmids)) - set(base_pmids))
                record["variants"][v.name] = {  # type: ignore[index]
                    "sub_queries": [
                        e["sub_queries"] for e in trace if e.get("step") == "decompose"
                    ],
                    "trace": trace,
                    "top_pmids": pmids,
                    "recall_at_10": rec,
                    "gold_found_by_sub_query": found_by,
                    "lost_vs_baseline": lost,
                    "gained_vs_baseline": gained,
                }
                print(
                    f"  {v.name:22s} recall@10={rec:.2f}  delta={rec - base_recall:+.2f}"
                    f"  lost={lost} gained={gained}"
                )
                for e in trace:
                    if e.get("step") == "decompose":
                        for sq in e["sub_queries"]:  # type: ignore[attr-defined]
                            print(f"      sub-query: {sq}")
            f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(questions)} records to {args.out}")
    print("Mean recall@10 delta vs hybrid_reranked:")
    for name, ds in deltas.items():
        mean = sum(ds) / len(ds) if ds else 0.0
        worse = sum(1 for d in ds if d < 0)
        better = sum(1 for d in ds if d > 0)
        print(f"  {name:22s} {mean:+.3f}   (worse on {worse}, better on {better}, n={len(ds)})")


if __name__ == "__main__":
    main()
