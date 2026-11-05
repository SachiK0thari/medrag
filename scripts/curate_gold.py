"""LLM-assisted gold-PMID curation for the eval question set.

For each EvalQuestion, builds a candidate pool from the local corpus, sends
the candidate titles + abstracts to the LLM, and asks it to pick the PMIDs
that authoritatively answer the question. Writes the result to
`src/medrag/eval/gold_pmids.py` and an audit trail (pool, per-retriever
provenance, picks) to `data/gold_curation_audit.json`.

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

Rationale for a pooled BM25 + dense candidate set (TREC-style pooling):
    - A paper can only become gold if it is in the candidate pool, so a pool
      drawn from ONE retriever biases the benchmark toward that retriever.
      The first version of this script pooled from BM25 only, and BM25-only
      then posted the best recall of any config — the fingerprint of that
      bias. Pooling the top-k of both first-stage retrievers removes it.
    - The reranker and the agent are deliberately NOT pool sources: they are
      the systems under test.
    - The pool is shuffled with a per-question seed before the LLM sees it,
      so list position carries no hint about which retriever ranked a paper
      highly (LLM judges are position-biased).

Usage:
    python scripts/curate_gold.py            # resume: only uncurated/failed questions
    python scripts/curate_gold.py --force    # re-curate every question from scratch
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
import time

from openai import APIStatusError

from medrag.agent import llm
from medrag.agent.graph import _extract_json
from medrag.config import PROJECT_ROOT, get_settings
from medrag.eval.questions import QUESTIONS
from medrag.models import Paper, RetrievedChunk
from medrag.retrieval.bm25 import BM25Retriever
from medrag.retrieval.dense import DenseRetriever
from medrag.store import Store

# Pacing: OpenRouter paid models are permissive, so the inter-request gap
# is short; the 429 backoff stays as a safety net. Each request is ~5000
# tokens (system + question + up to 30 abstracts @ 500 chars).
#
# The judge runs with reasoning ON: curation quality matters and latency does
# not. Reasoning tokens count against max_tokens, so the cap is generous; a
# 384-token cap once truncated every reply to an empty string, and 4000 was
# still not enough for ~1 question in 3 (the model deliberates over a 27-paper
# pool). If even this cap is exhausted, the question is retried once with
# reasoning OFF so it still gets labelled; the audit file records which mode
# produced each label.
JUDGE_MAX_TOKENS = 16000
JUDGE_FALLBACK_MAX_TOKENS = 1000
JUDGE_TIMEOUT_SEC = 300.0  # a reasoning reply can legitimately take minutes
PER_RETRIEVER_POOL_SIZE = 15  # unique papers taken from EACH of BM25 and dense
ABSTRACT_MAX_CHARS = 500
REQUEST_SLEEP_SEC = 2.0
RATE_LIMIT_BACKOFF_SEC = 60.0

OUTPUT_PATH = PROJECT_ROOT / "src" / "medrag" / "eval" / "gold_pmids.py"
AUDIT_PATH = get_settings().data_dir / "gold_curation_audit.json"


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


def _top_unique_papers(retrieved: list[RetrievedChunk], limit: int) -> list[Paper]:
    """First `limit` distinct papers in rank order (retrievers return chunks)."""
    seen: dict[str, Paper] = {}
    for rc in retrieved:
        if rc.paper.pmid not in seen:
            seen[rc.paper.pmid] = rc.paper
            if len(seen) >= limit:
                break
    return list(seen.values())


def build_pool(
    question: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    *,
    seed: str,
) -> tuple[list[Paper], dict[str, str]]:
    """Union of each retriever's top papers, shuffled.

    Returns the pooled papers and a provenance map pmid -> "bm25" | "dense" |
    "both", kept for the audit file only (never shown to the LLM).
    """
    fetch = PER_RETRIEVER_POOL_SIZE * 3  # over-fetch chunks; several map to one paper
    sparse = _top_unique_papers(bm25.search(question, fetch), PER_RETRIEVER_POOL_SIZE)
    dense_papers = _top_unique_papers(dense.search(question, fetch), PER_RETRIEVER_POOL_SIZE)

    provenance: dict[str, str] = {p.pmid: "bm25" for p in sparse}
    pool: dict[str, Paper] = {p.pmid: p for p in sparse}
    for p in dense_papers:
        provenance[p.pmid] = "both" if p.pmid in provenance else "dense"
        pool.setdefault(p.pmid, p)

    papers = list(pool.values())
    random.Random(seed).shuffle(papers)
    return papers, provenance


def _render_candidates(papers: list[Paper]) -> str:
    """PMID + title + trimmed abstract for the LLM prompt."""
    blocks: list[str] = []
    for paper in papers:
        abstract = (paper.abstract or "").strip().replace("\n", " ")
        if len(abstract) > ABSTRACT_MAX_CHARS:
            abstract = abstract[:ABSTRACT_MAX_CHARS] + "..."
        title = paper.title.strip()
        blocks.append(
            f"PMID {paper.pmid}: {title}\n    {abstract}"
            if abstract
            else f"PMID {paper.pmid}: {title}"
        )
    return "\n\n".join(blocks)


def _call_with_backoff(system: str, user: str, max_retries: int = 5) -> tuple[str, str]:
    """Ask the judge. Returns (reply text, judge mode: "reasoning" | "no_reasoning").

    Retries on 429/413 rate-limit errors, honouring the retry-after header.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = llm.complete(
                system,
                user,
                max_tokens=JUDGE_MAX_TOKENS,
                temperature=0.0,
                reasoning=True,
                timeout_s=JUDGE_TIMEOUT_SEC,
            )
            if not resp.truncated and resp.text.strip():
                return resp.text, "reasoning"
            print(
                f"    judge exhausted {JUDGE_MAX_TOKENS} tokens reasoning "
                f"(reasoning_tokens={resp.reasoning_tokens}); retrying without reasoning"
            )
            resp = llm.complete(
                system,
                user,
                max_tokens=JUDGE_FALLBACK_MAX_TOKENS,
                temperature=0.0,
                reasoning=False,
            )
            if resp.truncated or not resp.text.strip():
                raise RuntimeError(
                    f"empty/truncated judge reply in both modes (finish={resp.finish_reason})"
                )
            return resp.text, "no_reasoning"
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
    return "", "reasoning"  # unreachable, but satisfies mypy


def curate_one(
    qid: str,
    question: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
) -> tuple[list[str], str, dict[str, str], str]:
    """Returns (gold PMIDs, one-line reasoning, pool provenance, judge mode).

    Raises on an unusable judge reply so the caller records the question as
    failed (and a later resume retries it) rather than as "no gold exists".
    """
    papers, provenance = build_pool(question, bm25, dense, seed=qid)
    if not papers:
        return [], "no candidates from either retriever", provenance, "none"
    valid_pmids = set(provenance)

    user = f"QUESTION:\n{question}\n\nCANDIDATE POOL:\n{_render_candidates(papers)}"
    text, mode = _call_with_backoff(CURATION_SYSTEM, user)

    raw = _extract_json(text, "{", "}")
    if raw is None:
        raise RuntimeError(f"unparseable judge output: {text[:120]!r}")
    obj = json.loads(raw)

    picks = obj.get("gold_pmids", [])
    reasoning = str(obj.get("reasoning", "")).strip()
    if not isinstance(picks, list):
        raise RuntimeError("judge returned gold_pmids that is not a list")

    # Drop anything the LLM hallucinated outside the candidate pool.
    clean = [str(p) for p in picks if str(p) in valid_pmids]
    return clean, reasoning, provenance, mode


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
        "Curation method: pooled candidates (top-15 papers from BM25 plus top-15",
        "from dense retrieval, shuffled) → LLM filter (``LLM_MODEL``). Weak",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-curate every question, ignoring gold already in gold_pmids.py",
    )
    args = parser.parse_args()

    if args.model:
        get_settings().llm_model = args.model
        print(f"Using model override: {args.model}")

    store = Store()
    bm25 = BM25Retriever.from_store(store)
    dense = DenseRetriever(store=store)
    print(f"BM25 corpus loaded: {len(bm25)} chunks")

    if args.force:
        gold, reasoning = {}, {}
        print("--force: re-curating every question")
    else:
        gold, reasoning = _load_existing_gold()
        if gold:
            print(f"resuming: {sum(1 for v in gold.values() if v)} questions already curated")
    # Resume keeps the audit entries of questions that are not re-curated.
    audit: dict[str, dict[str, object]] = {}
    if not args.force and AUDIT_PATH.exists():
        audit = json.loads(AUDIT_PATH.read_text())

    for q in QUESTIONS:
        prior_note = reasoning.get(q.qid, "")
        already_curated = q.qid in gold and not prior_note.startswith("curation failed")
        if already_curated:
            print(f"  {q.qid}: skip ({len(gold[q.qid])} gold PMIDs, prior run)")
            continue
        print(f"  {q.qid}: {q.question[:80]}")
        try:
            picks, note, provenance, mode = curate_one(q.qid, q.question, bm25, dense)
        except Exception as e:  # checkpoint and continue on any failure
            print(f"    ERROR: {e}")
            gold[q.qid] = []
            reasoning[q.qid] = f"curation failed: {e}"
        else:
            gold[q.qid] = picks
            reasoning[q.qid] = note
            audit[q.qid] = {
                "question": q.question,
                "pool": provenance,
                "gold": {p: provenance[p] for p in picks},
                "reasoning": note,
                "judge_mode": mode,
            }
            AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            AUDIT_PATH.write_text(json.dumps(audit, indent=2))
            sources = [provenance[p] for p in picks]
            print(f"    → {picks}  from {sources}  [{mode}]  ({note[:70]})")

        # Checkpoint after every question so a mid-run crash doesn't lose work.
        write_gold_module(gold, reasoning)
        time.sleep(REQUEST_SLEEP_SEC)

    n_with_gold = sum(1 for v in gold.values() if v)
    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {n_with_gold}/{len(gold)} questions have at least one gold PMID")


if __name__ == "__main__":
    main()
