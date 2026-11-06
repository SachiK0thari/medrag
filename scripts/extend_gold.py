"""Grade the papers a re-ingest brought into reach, without touching the gold set.

The gold labels in `medrag.eval.gold_pmids` were picked from a pool: the top
papers of BM25 and of dense retrieval on the corpus as it was then (abstracts
only), each shown to the curating judge as the first 500 characters of its
abstract. After a re-ingest with full text that leaves three kinds of paper
whose label no longer says what the paper can answer:

    never judged      outside the old pool; body passages now bring it into a
                      config's top 10, and the eval scores it as a miss
    judged, not gold  in the old pool and passed over on its abstract, though
                      its body may hold the answer
    gold, "related"   picked as gold, then graded "related" because its
                      abstract does not contain the answer; same question

This script grades all three on the scale `scripts/grade_gold.py` uses
(2 direct, 1 related, 0 irrelevant). For each question the candidates are

    - the same pool as `scripts/curate_gold.py`, rebuilt on the current corpus
      (top 15 papers from BM25 and top 15 from dense retrieval),
    - every paper in any config's top 10 in the eval runs given by `--traces`,
    - the question's gold papers,

minus the gold papers already graded direct, and minus gold papers with no
indexed body (the judge would read the same abstract again). The judge reads
each paper's title and abstract plus, where the body is indexed, the two body
passages that best match the question.

The original labels and grades are not changed. The result is a separate
module, `src/medrag/eval/gold_extension.py`, which `scripts/analyze_eval.py`
uses to report retrieval against the original gold and against the gold plus
every paper graded direct here.

A hand-check of a sample of the papers graded direct, if one has been saved as
`data/gold_extension_spotcheck_verdicts.json`, is written into the module as
`HUMAN_CHECK`. It sits beside the judge's grades and does not change them.

Usage:
    python scripts/extend_gold.py --traces data/eval_traces.jsonl data/eval_traces_rep2.jsonl
    python scripts/extend_gold.py --traces data/eval_traces.jsonl --force
    python scripts/extend_gold.py --no-grade    # rewrite the module from the audit file
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from curate_gold import AUDIT_PATH as CURATION_AUDIT_PATH
from curate_gold import REQUEST_SLEEP_SEC, _call_with_backoff, _dq, build_pool

from medrag.agent.graph import _extract_json
from medrag.config import PROJECT_ROOT, get_settings
from medrag.eval.gold_grades import GOLD_GRADES
from medrag.eval.questions import QUESTIONS
from medrag.models import Paper
from medrag.retrieval.bm25 import BM25Retriever
from medrag.retrieval.dense import DenseRetriever
from medrag.store import Store

OUTPUT_PATH = PROJECT_ROOT / "src" / "medrag" / "eval" / "gold_extension.py"
AUDIT_PATH = get_settings().data_dir / "gold_extension_audit.json"
HUMAN_PATH = get_settings().data_dir / "gold_extension_spotcheck_verdicts.json"
_VERDICT_TO_GRADE = {"OK": 2, "WEAK": 1, "WRONG": 0}

TOP_K = 10  # the cut-off the eval scores at
PASSAGES_PER_PAPER = 2
PAPERS_PER_CALL = 8

GRADE_SYSTEM = """You are grading relevance labels for a retrieval benchmark over
GLP-1 cardiovascular literature.

You will be given a QUESTION and several PAPERS. For each paper you see its
PMID, title and abstract and, where its full text is indexed, up to two
passages from its body. Grade each paper independently on this scale:

2 = DIRECT. The text shown contains the specific information that answers the
    question: the finding, number, recommendation or mechanism asked about. A
    reader could answer the question, at least in substantial part, from the
    text shown alone.
1 = RELATED. The paper is on the same topic (same drug class, trial or outcome)
    but the text shown does not actually answer the question. Typical cases: it
    mentions the trial or result only in passing while studying something else
    (cost-effectiveness, a subgroup, a mediation analysis); it is a broad review
    when the question asks about one specific trial or number; it covers a
    different population, drug or endpoint than the one asked about.
0 = IRRELEVANT.

Be strict about 2: topical overlap is not enough. If the question names a
specific trial, a paper earns 2 only if the text shown reports the result of
that trial that the question asks about.

Return ONLY a JSON object with this exact shape:
{"grades": [{"pmid": "PMID", "grade": 2, "reason": "short phrase"}]}

Include every paper exactly once. No prose outside the JSON."""


def top10_by_question(traces_paths: list[Path]) -> dict[str, set[str]]:
    """qid -> papers in any config's top 10 in any of the given eval runs."""
    out: dict[str, set[str]] = defaultdict(set)
    for path in traces_paths:
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["qid"]].update(rec["ranked_pmids"][:TOP_K])
    return out


def body_passages(question: str, bm25: BM25Retriever) -> dict[str, list[str]]:
    """pmid -> its body chunks that best match `question`, best first."""
    out: dict[str, list[str]] = defaultdict(list)
    for rc in bm25.search(question, len(bm25)):
        if rc.chunk.section != "abstract" and len(out[rc.paper.pmid]) < PASSAGES_PER_PAPER:
            out[rc.paper.pmid].append(rc.chunk.text.strip().replace("\n", " "))
    return out


def render_paper(paper: Paper, passages: list[str]) -> str:
    lines = [f"PMID {paper.pmid}: {paper.title.strip()}"]
    lines.append(f"ABSTRACT: {(paper.abstract or '(none)').strip()}")
    lines.extend(f"PASSAGE {i}: {text}" for i, text in enumerate(passages, start=1))
    return "\n".join(lines)


def grade_batch(
    question: str, papers: list[Paper], passages: dict[str, list[str]]
) -> dict[str, dict[str, object]]:
    """Returns pmid -> {"grade", "reason", "judge_mode", "n_passages"}."""
    blocks = [render_paper(p, passages.get(p.pmid, [])) for p in papers]
    user = f"QUESTION:\n{question}\n\nPAPERS:\n" + "\n\n".join(blocks)
    text, mode = _call_with_backoff(GRADE_SYSTEM, user)
    raw = _extract_json(text, "{", "}")
    if raw is None:
        raise RuntimeError(f"unparseable judge output: {text[:120]!r}")
    wanted = {p.pmid for p in papers}
    out: dict[str, dict[str, object]] = {}
    for item in json.loads(raw).get("grades", []):
        pmid, grade = str(item.get("pmid", "")).replace("PMID", "").strip(), item.get("grade")
        if pmid in wanted and grade in (0, 1, 2):
            out[pmid] = {
                "grade": int(grade),
                "reason": str(item.get("reason", "")),
                "judge_mode": mode,
                "n_passages": len(passages.get(pmid, [])),
            }
    missing = sorted(wanted - set(out))
    if missing:
        raise RuntimeError(f"judge did not grade {missing}")
    return out


def load_human_check() -> dict[str, dict[str, int]]:
    """The hand-check of a sample of the papers graded direct, on the judge's scale."""
    if not HUMAN_PATH.exists():
        return {}
    verdicts = json.loads(HUMAN_PATH.read_text())["verdicts"]
    return {
        qid: {pmid: _VERDICT_TO_GRADE[v["verdict"]] for pmid, v in papers.items()}
        for qid, papers in verdicts.items()
    }


def write_module(
    grades: dict[str, dict[str, int]],
    judged_before: dict[str, set[str]],
    human: dict[str, dict[str, int]],
) -> None:
    lines = [
        '"""Grades for papers judged after the full-text re-ingest (2 direct, 1 related,',
        "0 irrelevant).",
        "",
        "Generated by `scripts/extend_gold.py`; do not edit by hand.",
        "",
        "Three kinds of paper: never judged by the original curation (outside its",
        "pool), judged by it and not picked, and picked as gold but graded only",
        '"related" on the abstract. One LLM judge, reading the abstract and the',
        "best-matching body passages. The gold labels in `gold_pmids.py` and the",
        "grades in `gold_grades.py` are unchanged.",
        "",
        "`HUMAN_CHECK` is a hand-check of a random sample of the papers graded 2",
        "(empty until one is done). It is kept for comparison: `EXTENSION_GRADES`",
        "holds the judge's grades whatever the hand-check says.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "# Papers the original curation judged (its candidate pool), gold or not.",
        "ORIGINAL_POOL: dict[str, list[str]] = {",
    ]
    for qid in sorted(judged_before):
        inner = ", ".join(_dq(p) for p in sorted(judged_before[qid]))
        lines.append(f"    {_dq(qid)}: [{inner}],")
    lines += ["}", "", "EXTENSION_GRADES: dict[str, dict[str, int]] = {"]
    for qid in sorted(grades):
        inner = ", ".join(f"{_dq(p)}: {g}" for p, g in sorted(grades[qid].items()))
        lines.append(f"    {_dq(qid)}: {{{inner}}},")
    lines += ["}", "", "HUMAN_CHECK: dict[str, dict[str, int]] = {"]
    for qid in sorted(human):
        inner = ", ".join(f"{_dq(p)}: {g}" for p, g in sorted(human[qid].items()))
        lines.append(f"    {_dq(qid)}: {{{inner}}},")
    lines += ["}", ""]
    OUTPUT_PATH.write_text("\n".join(lines))
    # The pool lists are long lines; CI checks formatting over the whole package.
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--quiet", str(OUTPUT_PATH)], check=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--traces",
        type=Path,
        nargs="*",
        default=[],
        help="Eval traces (one file per run) whose top-10 papers to include.",
    )
    parser.add_argument("--force", action="store_true", help="Re-grade every candidate.")
    parser.add_argument(
        "--no-grade",
        action="store_true",
        help="Grade nothing: rewrite the module from the audit file (no API calls).",
    )
    args = parser.parse_args()

    judged_before = {
        qid: set(entry["pool"])
        for qid, entry in json.loads(CURATION_AUDIT_PATH.read_text()).items()
    }
    audit: dict[str, dict[str, dict[str, object]]] = {}
    if not args.force and AUDIT_PATH.exists():
        audit = json.loads(AUDIT_PATH.read_text())["grades"]

    if not args.no_grade:
        grade_candidates(audit, args.traces)

    grades = {
        qid: {pmid: int(g["grade"]) for pmid, g in papers.items()}  # type: ignore[call-overload]
        for qid, papers in audit.items()
        if papers
    }
    human = load_human_check()
    write_module(grades, judged_before, human)
    counts = Counter(g for papers in grades.values() for g in papers.values())
    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  graded {sum(counts.values())} papers: {dict(sorted(counts.items()))}")
    print(
        f"  questions that gain a grade-2 paper: "
        f"{sum(1 for papers in grades.values() if 2 in papers.values())}/{len(QUESTIONS)}"
    )
    if human:
        checked = Counter(g for papers in human.values() for g in papers.values())
        print(f"  hand-checked {sum(checked.values())}: {dict(sorted(checked.items()))}")


def grade_candidates(audit: dict[str, dict[str, dict[str, object]]], traces: list[Path]) -> None:
    """Grade every candidate not yet in `audit`, saving the audit file after each batch."""
    store = Store()
    bm25 = BM25Retriever.from_store(store)
    dense = DenseRetriever(store=store)
    from_traces = top10_by_question(traces)

    for q in QUESTIONS:
        pool, _ = build_pool(q.question, bm25, dense, seed=q.qid)
        passages = body_passages(q.question, bm25)
        gold_grades = GOLD_GRADES.get(q.qid, {})
        # Gold papers the judge has nothing new to read for: already direct on
        # the abstract, or no body indexed.
        settled = {
            pmid for pmid in q.gold_pmids if gold_grades.get(pmid) == 2 or not passages.get(pmid)
        }
        candidates = {p.pmid for p in pool} | from_traces[q.qid] | set(q.gold_pmids)
        done = audit.setdefault(q.qid, {})
        todo = sorted(candidates - settled - set(done))
        if not todo:
            print(f"  {q.qid}: nothing new to grade")
            continue
        print(f"  {q.qid}: {len(todo)} papers to grade; {q.question[:60]}", flush=True)
        papers = store.get_papers(todo)
        for i in range(0, len(todo), PAPERS_PER_CALL):
            batch = [papers[p] for p in todo[i : i + PAPERS_PER_CALL]]
            try:
                done.update(grade_batch(q.question, batch, passages))
            except Exception as e:  # leave ungraded; a later run retries it
                print(f"    ERROR: {e}")
                continue
            AUDIT_PATH.write_text(json.dumps({"grades": audit}, indent=2))
            time.sleep(REQUEST_SLEEP_SEC)
        print(f"    → {dict(sorted(Counter(g['grade'] for g in done.values()).items()))}")


if __name__ == "__main__":
    main()
