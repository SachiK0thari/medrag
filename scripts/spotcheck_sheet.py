"""Build hand-check sheets for a split's labels, and read the marked sheets back.

Two sheets, each a fixed random sample:

    gold       curated gold labels; the reviewer reads the full abstract, as
               `scripts/grade_gold.py` does
    extension  papers `scripts/extend_gold.py` graded direct; the reviewer reads
               the abstract and body passages that judge read

Each paper is marked `[OK]` (the text shown answers the question), `[WEAK]`
(on topic, does not answer it) or `[WRONG]` (irrelevant). The judge's reason
sits in a collapsed block below each paper, to read after deciding.

`collect` writes the marks to the verdict files the grading scripts read
(`data/gold_spotcheck_verdicts[_test].json`,
`data/gold_extension_spotcheck_verdicts[_test].json`); rerun
`grade_gold.py --split <split>` and `extend_gold.py --split <split> --no-grade`
afterwards to fold them into the generated modules.

Usage:
    python scripts/spotcheck_sheet.py make --split test
    python scripts/spotcheck_sheet.py collect --split test --reviewer "Sachi Kothari"
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import date
from pathlib import Path

from extend_gold import PASSAGES_PER_PAPER, body_passages

from medrag.config import get_settings
from medrag.eval.splits import SPLITS, load_generated, questions_for, split_files
from medrag.models import Paper
from medrag.retrieval.bm25 import BM25Retriever
from medrag.store import Store

SAMPLE_SIZE = 20
SEED = 20261119
SCALE = {
    "OK": "the text shown answers the question",
    "WEAK": "on topic, does not answer it",
    "WRONG": "irrelevant",
}
_MARK = re.compile(r"^- \[(OK|WEAK|WRONG|\s*)\] \*\*PMID (\d+)\*\*")
_HEADING = re.compile(r"^## (\w+):")


def sheet_paths(split: str) -> dict[str, Path]:
    s = "" if split == "dev" else f"_{split}"
    data = get_settings().data_dir
    return {
        "gold": data / f"gold_spotcheck{s}.md",
        "extension": data / f"gold_extension_spotcheck{s}.md",
    }


def _sample(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted(random.Random(SEED).sample(sorted(pairs), min(SAMPLE_SIZE, len(pairs))))


def _paper_block(paper: Paper, passages: list[str], reason: str) -> list[str]:
    lines = [
        f"- [ ] **PMID {paper.pmid}** ({paper.journal}, {paper.year}): {paper.title.strip()}",
        f"  - https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}/",
        f"  - ABSTRACT: {(paper.abstract or '(none)').strip()}",
    ]
    lines += [f"  - PASSAGE {i}: {text}" for i, text in enumerate(passages, start=1)]
    lines += [
        "",
        "  <details><summary>Judge's reason (read after deciding)</summary>",
        "",
        f"  {reason}",
        "",
        "  </details>",
        "",
    ]
    return lines


def _write_sheet(
    path: Path,
    title: str,
    intro: str,
    picks: list[tuple[str, str]],
    questions: dict[str, str],
    papers: dict[str, Paper],
    passages: dict[str, dict[str, list[str]]],
    reasons: dict[tuple[str, str], str],
) -> None:
    lines = [
        f"# {title} (local, gitignored)",
        "",
        intro,
        "",
        "Replace each `[ ]` with `[OK]`, `[WEAK]` or `[WRONG]`:",
        "",
        *(f"- `[{k}]` {v}" for k, v in SCALE.items()),
        "",
        "Decide before opening the judge's reason.",
        "",
    ]
    current = None
    for qid, pmid in picks:
        if qid != current:
            lines += [f"## {qid}: {questions[qid]}", ""]
            current = qid
        lines += _paper_block(papers[pmid], passages.get(qid, {}).get(pmid, []), reasons[qid, pmid])
    path.write_text("\n".join(lines))
    print(f"wrote {path} ({len(picks)} papers)")


def make(split: str) -> None:
    files = split_files(split, get_settings().data_dir)
    questions = {q.qid: q.question for q in questions_for(split)}
    graded = json.loads(files.grades_audit.read_text())["grades"]
    extension = json.loads(files.extension_audit.read_text())["grades"]
    loaded = load_generated(files, "gold_grades", "LLM_GRADES")
    llm_grades: dict[str, dict[str, int]] = loaded[0] if loaded else {}

    gold_pairs = [(q, p) for q, grades in llm_grades.items() for p in grades]
    ext_pairs = [
        (q, p) for q, papers in extension.items() for p, g in papers.items() if int(g["grade"]) == 2
    ]
    gold_picks, ext_picks = _sample(gold_pairs), _sample(ext_pairs)

    store = Store()
    papers = store.get_papers(sorted({p for _, p in gold_picks + ext_picks}))
    bm25 = BM25Retriever.from_store(store)
    passages = {
        qid: body_passages(questions[qid], bm25) for qid in sorted({q for q, _ in ext_picks})
    }
    for qid, pmid in ext_picks:
        n = int(extension[qid][pmid].get("n_passages", PASSAGES_PER_PAPER))
        passages[qid][pmid] = passages[qid].get(pmid, [])[:n]

    paths = sheet_paths(split)
    _write_sheet(
        paths["gold"],
        f"Gold-label hand-check sheet, {split} split",
        f"{len(gold_picks)} of the {len(gold_pairs)} curated gold labels, drawn with "
        f"`random.Random({SEED}).sample` over the sorted (qid, pmid) pairs. Judge the "
        "abstract alone, as `scripts/grade_gold.py` does.",
        gold_picks,
        questions,
        papers,
        {},
        {(q, p): f"grade {graded[q][p]['grade']}: {graded[q][p]['reason']}" for q, p in gold_picks},
    )
    _write_sheet(
        paths["extension"],
        f"Extension-label hand-check sheet, {split} split",
        f"{len(ext_picks)} of the {len(ext_pairs)} papers `scripts/extend_gold.py` graded "
        f"direct, drawn with `random.Random({SEED}).sample` over the sorted (qid, pmid) "
        "pairs. The abstract and passages are exactly what the judge read.",
        ext_picks,
        questions,
        papers,
        passages,
        {(q, p): str(extension[q][p].get("reason", "")) for q, p in ext_picks},
    )


def parse_sheet(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    """qid -> pmid -> {"verdict"}; raises if any paper is unmarked."""
    verdicts: dict[str, dict[str, dict[str, str]]] = {}
    unmarked: list[str] = []
    qid = None
    for line in path.read_text().splitlines():
        if heading := _HEADING.match(line):
            qid = heading.group(1)
        elif (mark := _MARK.match(line)) and qid:
            verdict, pmid = mark.group(1).strip(), mark.group(2)
            if not verdict:
                unmarked.append(f"{qid}/{pmid}")
            else:
                verdicts.setdefault(qid, {})[pmid] = {"verdict": verdict}
    if unmarked:
        raise SystemExit(f"{path.name}: unmarked papers: {', '.join(unmarked)}")
    return verdicts


def collect(split: str, reviewer: str) -> None:
    files = split_files(split, get_settings().data_dir)
    paths = sheet_paths(split)
    for kind, out in (("gold", files.grades_human), ("extension", files.extension_human)):
        verdicts = parse_sheet(paths[kind])
        n = sum(len(v) for v in verdicts.values())
        record = {
            "reviewer": reviewer,
            "date": date.today().isoformat(),
            "sample": f"{n} {kind} labels from {paths[kind].name}, "
            f"random.Random({SEED}).sample over the sorted (qid, pmid) pairs",
            "scale": SCALE,
            "verdicts": verdicts,
        }
        out.write_text(json.dumps(record, indent=2) + "\n")
        counts = {
            k: sum(v["verdict"] == k for p in verdicts.values() for v in p.values()) for k in SCALE
        }
        print(f"wrote {out} ({n} verdicts: {counts})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="Write the two blank sheets.")
    m.add_argument("--split", choices=SPLITS, default="test")
    c = sub.add_parser("collect", help="Turn the marked sheets into verdict files.")
    c.add_argument("--split", choices=SPLITS, default="test")
    c.add_argument("--reviewer", required=True, help="Name of the person who marked the sheets.")
    args = parser.parse_args()
    if args.cmd == "make":
        make(args.split)
    else:
        collect(args.split, args.reviewer)


if __name__ == "__main__":
    main()
