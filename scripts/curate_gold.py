"""LLM-assisted gold-PMID curation for the eval question set.

For each EvalQuestion, pulls a BM25 candidate pool from the local corpus,
sends the candidate titles + abstracts to the LLM, and asks it to pick the
PMIDs that authoritatively answer the question. Writes the result to
`src/medrag/eval/gold_pmids.py`.

This is WEAK SUPERVISION: the LLM has only the corpus abstracts to work
with, and the corpus here is secondary GLP-1/CV literature (reviews,
meta-analyses, emulation studies), not the original trial papers. For each
question we therefore pick the best-available answer in the corpus. The
output should be spot-checked before publication — treat it as a seed, not
a ground truth.

Rationale for LLM curation over manual curation:
    - Scales to dozens of questions without a day of manual PubMed work.
    - Transparent and re-runnable when the corpus is re-ingested.
    - The prompt is auditable (lives in this file).

Rationale for BM25 candidate pool over hybrid:
    - Pure-keyword recall breaks a circularity concern: if we used dense
      retrieval to build the gold pool, the dense baseline would trivially
      win the eval. BM25 also isn't the method we're most trying to showcase.
    - 30 candidates gives plenty of headroom for the LLM to pick 2-5 gold.

Usage:
    python scripts/curate_gold.py
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time

from openai import APIStatusError

from medrag.agent import llm
from medrag.agent.graph import _extract_json
from medrag.config import PROJECT_ROOT, get_settings
from medrag.eval.questions import QUESTIONS
from medrag.retrieval.bm25 import BM25Retriever
from medrag.store import Store

# Pacing: OpenRouter paid models are permissive, so the inter-request gap
# is short; the 429 backoff stays as a safety net. Each request is ~2500 tokens (system + question + 15
# abstracts @ 500 chars) + ~300 output.
CANDIDATE_POOL_SIZE = 15
ABSTRACT_MAX_CHARS = 500
REQUEST_SLEEP_SEC = 2.0
RATE_LIMIT_BACKOFF_SEC = 60.0

OUTPUT_PATH = PROJECT_ROOT / "src" / "medrag" / "eval" / "gold_pmids.py"


CURATION_SYSTEM = """You are a medical literature curator building a retrieval
evaluation gold set for a question-answering system over GLP-1 cardiovascular
literature.

You will be given:
- a QUESTION,
- a CANDIDATE POOL of papers (PMID + title + abstract) from the system's corpus.

Your job: pick the PMIDs of papers in the candidate pool that authoritatively
answer the question. "Authoritatively" means the paper directly addresses the
question and a reader would cite it as evidence. The corpus is mostly reviews,
meta-analyses, and emulation studies — pick the best available answers in the
pool, not hypothetical landmark trial papers.

Rules:
- Return 1 to 5 PMIDs. Prefer 2-3 when there are multiple good answers.
- Only return PMIDs that appear in the candidate pool.
- If NO paper in the pool is a good answer, return an empty list.
- Be selective: a tangentially relevant review is NOT gold.

Return ONLY a JSON object with this exact shape:
{"gold_pmids": ["PMID1", "PMID2", ...], "reasoning": "one short sentence"}

No prose outside the JSON."""


def _render_candidates(retrieved: list) -> tuple[str, set[str]]:  # type: ignore[type-arg]
    """Render unique PMIDs + title + trimmed abstract for the LLM prompt."""
    seen: dict[str, str] = {}
    for rc in retrieved:
        pmid = rc.paper.pmid
        if pmid in seen or len(seen) >= CANDIDATE_POOL_SIZE:
            continue
        abstract = (rc.paper.abstract or "").strip().replace("\n", " ")
        if len(abstract) > ABSTRACT_MAX_CHARS:
            abstract = abstract[:ABSTRACT_MAX_CHARS] + "..."
        title = rc.paper.title.strip()
        seen[pmid] = (
            f"PMID {pmid}: {title}\n    {abstract}" if abstract else f"PMID {pmid}: {title}"
        )
    block = "\n\n".join(seen.values())
    return block, set(seen.keys())


def _call_with_backoff(system: str, user: str, max_retries: int = 5) -> str:
    """Retry on 429/413 rate-limit errors, honouring the retry-after header."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = llm.complete(system, user, max_tokens=384, temperature=0.0)
            return resp.text
        except APIStatusError as e:
            if attempt == max_retries or e.status_code not in (413, 429):
                raise
            # Use the server's retry-after header when available.
            retry_after = e.response.headers.get("retry-after")
            backoff = float(retry_after) if retry_after else RATE_LIMIT_BACKOFF_SEC
            print(
                f"    rate limited ({e.status_code}), attempt {attempt}/{max_retries}; sleeping {backoff}s"
            )
            time.sleep(backoff)
    return ""  # unreachable, but satisfies mypy


def curate_one(
    question: str,
    bm25: BM25Retriever,
) -> tuple[list[str], str]:
    retrieved = bm25.search(question, CANDIDATE_POOL_SIZE * 3)  # extra for dedup
    candidates_block, valid_pmids = _render_candidates(retrieved)
    if not valid_pmids:
        return [], "no candidates from BM25"

    user = f"QUESTION:\n{question}\n\nCANDIDATE POOL:\n{candidates_block}"
    text = _call_with_backoff(CURATION_SYSTEM, user)

    raw = _extract_json(text, "{", "}")
    if raw is None:
        return [], f"unparseable LLM output: {text[:120]}"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return [], f"JSON decode error: {e}"

    picks = obj.get("gold_pmids", [])
    reasoning = str(obj.get("reasoning", "")).strip()
    if not isinstance(picks, list):
        return [], "gold_pmids not a list"

    # Drop anything the LLM hallucinated outside the candidate pool.
    clean = [str(p) for p in picks if str(p) in valid_pmids]
    return clean, reasoning


def _dq(s: str) -> str:
    """Python literal using double quotes — matches ruff format's preference."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _dq_list(items: list[str]) -> str:
    return "[" + ", ".join(_dq(i) for i in items) + "]"


def write_gold_module(gold: dict[str, list[str]], reasoning: dict[str, str]) -> None:
    lines: list[str] = [
        '"""LLM-curated gold PMIDs for the eval question set.',
        "",
        "Generated by `scripts/curate_gold.py`. DO NOT edit by hand unless you're",
        "overriding a specific question — re-run the script instead. The `NOTES`",
        "dict records the LLM's one-line justification per question and is kept",
        "for audit / review.",
        "",
        "Curation method: BM25 top-15 candidates → LLM filter (``LLM_MODEL``). Weak",
        "supervision, not ground truth — spot-check before publication.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "GOLD_PMIDS: dict[str, list[str]] = {",
    ]
    for qid in sorted(gold.keys()):
        lines.append(f"    {_dq(qid)}: {_dq_list(gold[qid])},")
    lines.append("}")
    lines.append("")
    lines.append("NOTES: dict[str, str] = {")
    for qid in sorted(reasoning.keys()):
        note = textwrap.shorten(reasoning[qid], width=100, placeholder="...")
        lines.append(f"    {_dq(qid)}: {_dq(note)},")
    lines.append("}")
    lines.append("")
    OUTPUT_PATH.write_text("\n".join(lines))


def _load_existing_gold() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Resume: pull any gold already written to gold_pmids.py so a re-run
    only re-curates questions that failed last time."""
    try:
        from medrag.eval import gold_pmids as existing  # type: ignore[attr-defined]
    except ImportError:
        return {}, {}
    gold = dict(getattr(existing, "GOLD_PMIDS", {}))
    notes = dict(getattr(existing, "NOTES", {}))
    return gold, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate gold PMIDs for eval questions")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the OpenRouter model id (default: LLM_MODEL from .env)",
    )
    args = parser.parse_args()

    if args.model:
        get_settings().llm_model = args.model
        print(f"Using model override: {args.model}")

    store = Store()
    bm25 = BM25Retriever.from_store(store)
    print(f"BM25 corpus loaded: {len(bm25)} chunks")

    gold, reasoning = _load_existing_gold()
    if gold:
        print(f"resuming: {sum(1 for v in gold.values() if v)} questions already curated")

    for q in QUESTIONS:
        prior_note = reasoning.get(q.qid, "")
        already_curated = q.qid in gold and not prior_note.startswith("curation failed")
        if already_curated:
            print(f"  {q.qid}: skip ({len(gold[q.qid])} gold PMIDs, prior run)")
            continue
        print(f"  {q.qid}: {q.question[:80]}")
        try:
            picks, note = curate_one(q.question, bm25)
        except Exception as e:  # checkpoint and continue on any failure
            print(f"    ERROR: {e}")
            gold[q.qid] = []
            reasoning[q.qid] = f"curation failed: {e}"
        else:
            gold[q.qid] = picks
            reasoning[q.qid] = note
            print(f"    → {picks}  ({note[:80]})")

        # Checkpoint after every question so a mid-run crash doesn't lose work.
        write_gold_module(gold, reasoning)
        time.sleep(REQUEST_SLEEP_SEC)

    n_with_gold = sum(1 for v in gold.values() if v)
    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {n_with_gold}/{len(gold)} questions have at least one gold PMID")


if __name__ == "__main__":
    main()
