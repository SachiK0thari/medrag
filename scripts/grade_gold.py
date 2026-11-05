"""Grade every gold label on a three-level relevance scale.

`scripts/curate_gold.py` picks gold papers with a lenient, binary criterion
and sees only the first 500 characters of each abstract. A manual spot-check
of 28 of its labels found none irrelevant but half only topically related:
the paper is about the right drug or trial, yet its abstract does not contain
the answer. Recall against that set rewards retrieving on-topic papers, which
is a weaker claim than retrieving the answer.

This script re-reads each gold paper's FULL abstract and assigns:

    2  direct    the abstract contains the information that answers the question
    1  related   on topic, but the abstract does not answer the question
    0  irrelevant

and writes `src/medrag/eval/gold_grades.py`, which the analysis script uses to
report every config against both gold sets:

    lenient  grade >= 1   (the set the harness scores against)
    strict   grade == 2   (questions with no grade-2 paper are excluded)

The judge is zero-shot: it never sees the human verdicts. Those are held out
to measure agreement (accuracy and Cohen's kappa, printed at the end and saved
in the audit file), and they override the judge wherever both exist.

Usage:
    python scripts/grade_gold.py            # grade ungraded questions
    python scripts/grade_gold.py --force    # re-grade everything
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from curate_gold import REQUEST_SLEEP_SEC, _call_with_backoff, _dq

from medrag.agent.graph import _extract_json
from medrag.config import PROJECT_ROOT, get_settings
from medrag.eval.metrics import cohen_kappa
from medrag.eval.questions import QUESTIONS
from medrag.store import Store

OUTPUT_PATH = PROJECT_ROOT / "src" / "medrag" / "eval" / "gold_grades.py"
AUDIT_PATH = get_settings().data_dir / "gold_grades_audit.json"
HUMAN_PATH = get_settings().data_dir / "gold_spotcheck_verdicts.json"
_VERDICT_TO_GRADE = {"OK": 2, "WEAK": 1, "WRONG": 0}

GRADE_SYSTEM = """You are grading relevance labels for a retrieval benchmark over
GLP-1 cardiovascular literature.

You will be given a QUESTION and several PAPERS (PMID, title, full abstract).
Grade each paper independently on this scale:

2 = DIRECT. The abstract itself contains the specific information that answers
    the question: the finding, number, recommendation or mechanism asked about.
    A reader could answer the question, at least in substantial part, from this
    abstract alone.
1 = RELATED. The paper is on the same topic (same drug class, trial or outcome)
    but its abstract does not actually answer the question. Typical cases: it
    mentions the trial or result only in passing while studying something else
    (cost-effectiveness, a subgroup, a mediation analysis); it is a broad review
    when the question asks about one specific trial or number; it covers a
    different population, drug or endpoint than the one asked about.
0 = IRRELEVANT.

Be strict about 2: topical overlap is not enough. If the question names a
specific trial, a paper earns 2 only if its abstract reports the result of that
trial that the question asks about.

Return ONLY a JSON object with this exact shape:
{"grades": [{"pmid": "PMID", "grade": 2, "reason": "short phrase"}]}

Include every paper exactly once. No prose outside the JSON."""


def grade_question(question: str, pmids: list[str], store: Store) -> dict[str, dict[str, object]]:
    """Returns pmid -> {"grade": int, "reason": str, "judge_mode": str}."""
    papers = store.get_papers(pmids)
    blocks = [
        f"PMID {p}: {papers[p].title.strip()}\n{(papers[p].abstract or '').strip()}" for p in pmids
    ]
    user = f"QUESTION:\n{question}\n\nPAPERS:\n" + "\n\n".join(blocks)
    text, mode = _call_with_backoff(GRADE_SYSTEM, user)
    raw = _extract_json(text, "{", "}")
    if raw is None:
        raise RuntimeError(f"unparseable judge output: {text[:120]!r}")
    out: dict[str, dict[str, object]] = {}
    for item in json.loads(raw).get("grades", []):
        pmid, grade = str(item.get("pmid", "")).replace("PMID", "").strip(), item.get("grade")
        if pmid in pmids and grade in (0, 1, 2):
            out[pmid] = {
                "grade": int(grade),
                "reason": str(item.get("reason", "")),
                "judge_mode": mode,
            }
    missing = [p for p in pmids if p not in out]
    if missing:
        raise RuntimeError(f"judge did not grade {missing}")
    return out


def load_human_grades() -> dict[str, dict[str, int]]:
    if not HUMAN_PATH.exists():
        return {}
    verdicts = json.loads(HUMAN_PATH.read_text())["verdicts"]
    return {
        qid: {pmid: _VERDICT_TO_GRADE[v["verdict"]] for pmid, v in papers.items()}
        for qid, papers in verdicts.items()
    }


def agreement(
    llm: dict[str, dict[str, int]], human: dict[str, dict[str, int]]
) -> dict[str, object]:
    """Judge-vs-human agreement on the labels both graded."""
    pairs = [
        (human[q][p], llm[q][p])
        for q in sorted(human)
        for p in sorted(human[q])
        if p in llm.get(q, {})
    ]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    observed = sum(h == m for h, m in pairs) / n
    kappa = cohen_kappa([h for h, _ in pairs], [m for _, m in pairs])
    confusion = {f"human={h},judge={m}": c for (h, m), c in sorted(Counter(pairs).items())}
    return {
        "n": n,
        "accuracy": round(observed, 3),
        "cohen_kappa": round(kappa, 3),
        "confusion": confusion,
    }


def _grades_literal(name: str, grades: dict[str, dict[str, int]]) -> list[str]:
    lines = [f"{name}: dict[str, dict[str, int]] = {{"]
    for qid in sorted(grades):
        inner = ", ".join(f"{_dq(p)}: {g}" for p, g in grades[qid].items())
        lines.append(f"    {_dq(qid)}: {{{inner}}},")
    lines.append("}")
    return lines


def write_module(llm: dict[str, dict[str, int]], human: dict[str, dict[str, int]]) -> None:
    lines = [
        '"""Graded relevance for the gold labels (2 direct, 1 related, 0 irrelevant).',
        "",
        "Generated by `scripts/grade_gold.py`; do not edit by hand.",
        "",
        "`LLM_GRADES` is the zero-shot judge reading each full abstract.",
        "`HUMAN_GRADES` is the manual spot-check (the judge never saw it).",
        "`GOLD_GRADES` is what the analysis uses: the human grade where one",
        "exists, the judge's grade otherwise.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        *_grades_literal("LLM_GRADES", llm),
        "",
        *_grades_literal("HUMAN_GRADES", human),
        "",
        "GOLD_GRADES: dict[str, dict[str, int]] = {",
        "    qid: {**grades, **HUMAN_GRADES.get(qid, {})} for qid, grades in LLM_GRADES.items()",
        "}",
        "",
    ]
    OUTPUT_PATH.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="Re-grade every question.")
    args = parser.parse_args()

    store = Store()
    audit: dict[str, dict[str, dict[str, object]]] = {}
    if not args.force and AUDIT_PATH.exists():
        audit = json.loads(AUDIT_PATH.read_text()).get("grades", {})
    human = load_human_grades()

    for q in QUESTIONS:
        if not q.gold_pmids:
            continue
        if set(audit.get(q.qid, {})) == set(q.gold_pmids):
            print(f"  {q.qid}: skip (already graded)")
            continue
        print(f"  {q.qid}: {q.question[:80]}", flush=True)
        try:
            audit[q.qid] = grade_question(q.question, q.gold_pmids, store)
        except Exception as e:  # leave ungraded; a later run retries it
            print(f"    ERROR: {e}")
            continue
        print(
            "    → " + ", ".join(f"{p}={g['grade']}" for p, g in audit[q.qid].items()), flush=True
        )
        AUDIT_PATH.write_text(json.dumps({"grades": audit}, indent=2))
        time.sleep(REQUEST_SLEEP_SEC)

    llm = {q: {p: int(g["grade"]) for p, g in papers.items()} for q, papers in audit.items()}  # type: ignore[call-overload]
    stats = agreement(llm, human)
    AUDIT_PATH.write_text(json.dumps({"agreement_with_human": stats, "grades": audit}, indent=2))
    write_module(llm, human)

    final = {q: {**g, **human.get(q, {})} for q, g in llm.items()}
    counts = Counter(g for papers in final.values() for g in papers.values())
    strict_questions = sum(1 for papers in final.values() if 2 in papers.values())
    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(
        f"  graded {sum(counts.values())} labels on {len(final)} questions: {dict(sorted(counts.items()))}"
    )
    print(f"  questions with at least one grade-2 paper: {strict_questions}/{len(final)}")
    print(f"  judge vs human: {stats}")


if __name__ == "__main__":
    main()
