---
title: Medrag
emoji: 👀
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Agentic RAG over GLP-1 cardiovascular literature
---

# MedRAG — Agentic RAG over GLP-1 Cardiovascular Outcome Literature

> Medical affairs teams at GLP-1 manufacturers field hundreds of literature
> questions weekly from field medical liaisons. **MedRAG reduces a multi-hour
> manual literature review to a cited answer in under 20 seconds** over 500
> open-access papers on GLP-1 receptor agonists and cardiovascular outcomes.

[![ci](https://github.com/SachiK0thari/medrag/actions/workflows/ci.yml/badge.svg)](https://github.com/SachiK0thari/medrag/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org)
[![ruff](https://img.shields.io/badge/lint-ruff-orange)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/types-mypy%20strict-blue)](https://mypy-lang.org)

**Live demo:** [huggingface.co/spaces/SachiK/medrag](https://huggingface.co/spaces/SachiK/medrag)

---

## What it does

Given a clinical question (e.g. *"How did semaglutide affect MACE in SUSTAIN-6?"*),
MedRAG:

1. **Plans** the search with an LLM: atomic questions are searched as asked,
   and only genuinely multi-part questions are split into sub-queries.
2. **Retrieves** relevant chunks from a 500-paper PubMed/PMC corpus using a
   hybrid BM25 + dense pipeline fused via Reciprocal Rank Fusion.
3. **Reranks** the merged candidates against the original question with a
   cross-encoder (bge-reranker-v2-m3).
4. **Reflects** on whether retrieved evidence is sufficient; re-retrieves if not.
5. **Answers** with inline citations grounded in specific source chunks,
   returned as a strict Pydantic model.

Every step is type-checked, traced (latency + token spend), and evaluated.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full diagram.

```
question → decompose → [BM25 + dense → RRF → rerank] → reflect → answer
                            │                              │
                            └── PubMed/PMC corpus          └── cited Pydantic Answer
```

## Evaluation

33 held-out GLP-1 / cardiovascular questions, ten configurations, one run each
with `deepseek/deepseek-v4.1-flash`. The results and per-step traces are
committed, so every number below regenerates offline in a second:

```bash
python scripts/analyze_eval.py
```

| Config | Recall@10 | Recall@10 (strict gold) | MRR | RAGAS faithfulness | p50 latency | $/query |
|---|---|---|---|---|---|---|
| BM25 only | 0.490 | 0.620 | 0.408 | 0.868 | 1.2 s | $0.00034 |
| Dense only | 0.586 | 0.546 | 0.592 | 0.872 | 1.9 s | $0.00036 |
| Hybrid (RRF) | 0.601 | 0.722 | 0.521 | 0.933 | 1.5 s | $0.00035 |
| Hybrid + rerank | 0.652 | 0.750 | 0.658 | 0.906 | 6.6 s | $0.00026 |
| Agentic v1 (original loop) | 0.449 | 0.611 | 0.439 | 0.897 | 26.1 s | $0.00074 |
| **Agentic v2 (ships)** | **0.657** | **0.769** | **0.661** | 0.912 | 16.9 s | $0.00064 |

### Key finding: the first agent made retrieval worse, and the traces showed why

The original agentic loop scored 0.20 recall **below** the hybrid + rerank
pipeline it wraps (95% CI −0.33 to −0.08), at four times the latency. A
one-knob-at-a-time ablation over the saved traces located the cause:

| | Gold papers found by a sub-query | Gold papers in the final top 10 |
|---|---|---|
| Hybrid + rerank (no agent) | — | 57 / 90 |
| Agentic v1 | 69 / 90 | 40 / 90 |
| v1 + merged pool re-scored against the question | 75 / 90 | 59 / 90 |
| v1 + planner told not to pad the query list | 64 / 90 | 58 / 90 |

The agent's sub-queries *found* more gold papers than the baseline. It then
discarded them: each sub-query's results were reranked against that sub-query
and the lists were merged by score, but cross-encoder scores computed against
different queries are not comparable. Either fix alone restores baseline
recall. Reflection, the usual suspect, turned out not to matter.

The fixed agent **matches** the single-shot pipeline and does not beat it. The
planner judged 32 of the 33 questions atomic, so on this question set the loop
costs 2.6× the latency for the same papers. The full analysis, including the
controls and the per-question evidence, is in
[`docs/eval_findings.md`](docs/eval_findings.md).

### How far to trust these numbers

- **Gold labels are LLM-curated** from a pooled BM25 + dense candidate set
  (`scripts/curate_gold.py`). I hand-checked 28 of the 90 labels: none were
  irrelevant, 14 directly answer the question and 14 are only on topic.
- **Hence the strict column.** A second judge graded every label against the
  full abstract (`scripts/grade_gold.py`); it agreed with my 28 verdicts on 22
  (Cohen's κ = 0.57) and was never more lenient than me. Strict gold keeps the
  34 "directly answers" labels, which cover 18 questions.
- **Differences under about 0.05 are noise.** Recall deltas carry paired
  bootstrap confidence intervals, and two configs that are identical up to the
  first merge differed by 0.045 from planner sampling alone.
- **What is robust:** hybrid beats either retriever alone, the reranker's gain
  is in ordering (strict MRR 0.44 → 0.72), and agentic v1 is worse than the
  baseline. **What is not:** BM25 against dense, which flips between the two
  gold sets.

### What I'd do differently

1. **Route instead of always looping.** When the planner returns one query,
   run the single-shot path and skip reflection: baseline quality at baseline
   cost.
2. **Make the critic emit a search query.** Today the follow-up search uses
   the critic's complaint verbatim, and it changed recall on 2 of the 12
   questions where it fired.
3. **Add multi-hop questions.** Decomposition can only win on questions with
   independent parts, and this set has almost none.
4. **Repeat the agentic runs** to shrink the noise floor before claiming any
   difference smaller than 0.05.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Sources | PubMed E-utilities + PMC OAI-PMH | Free, structured, MeSH-tagged |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) | 384-d, CPU-friendly, top of MTEB at size |
| Vector DB | Pinecone (serverless) | Production-grade, free tier sufficient |
| Sparse | `rank_bm25` | Pure Python, zero deps |
| Rerank | `BAAI/bge-reranker-v2-m3` (local) | Cross-encoder precision boost |
| LLM | DeepSeek V4.1 Flash via OpenRouter | Cheap, fast, any OpenRouter model id via `LLM_MODEL` |
| Orchestration | LangGraph | Explicit state machine for agent loop |
| API | FastAPI + Pydantic | Type-safe contracts end to end |
| UI | Streamlit | Fast to ship, deploys to HF Spaces |
| Eval | Custom harness + RAGAS faithfulness | Recall@k, MRR, nDCG, graded gold, bootstrap CIs, per-step traces |
| Quality | ruff + mypy strict + pytest + GH Actions | Production-grade discipline |

The LLM client (`medrag.agent.llm`) talks to OpenRouter's OpenAI-compatible API, so
swapping to Claude, GPT, Llama, etc. is a one-line `LLM_MODEL` change.

## Quickstart

```bash
# 1. Install
git clone https://github.com/SachiK0thari/medrag.git
cd medrag
python -m venv .venv && source .venv/bin/activate
make dev   # installs with dev + eval extras, sets up pre-commit

# 2. Configure
cp .env.example .env
# Edit .env with your OpenRouter, Pinecone, and (optional) NCBI API keys

# 3. Ingest
make ingest        # ~10-15 min for 500 papers on a laptop

# 4. Evaluate
make eval                        # writes data/eval_results.csv + eval_traces.jsonl
python scripts/analyze_eval.py   # tables, confidence intervals, agent attribution

# 5. Run
make api           # FastAPI on http://localhost:8000
make ui            # Streamlit on http://localhost:8501
```

## Project layout

```
medrag/
├── src/medrag/
│   ├── models.py           ← Pydantic data contracts
│   ├── config.py           ← All env vars + tunables
│   ├── ingestion/          ← PubMed → PMC → chunk → embed
│   ├── retrieval/          ← BM25 + dense + RRF + rerank
│   ├── agent/              ← LangGraph loop, prompts, LLM wrapper
│   ├── eval/               ← Metrics, harness, held-out questions
│   ├── api/                ← FastAPI service
│   └── ui/                 ← Streamlit demo
├── tests/                  ← pytest, runs in CI
├── scripts/                ← ingest, run_eval, analyze_eval, curate_gold, grade_gold
├── data/eval_results.csv   ← Committed eval run (with eval_traces.jsonl)
├── docs/architecture.md    ← System diagram + design rationale
├── docs/eval_findings.md   ← Why the agent regressed and what fixed it
├── .github/workflows/ci.yml← Lint + types + tests on every PR
├── Dockerfile              ← HF Spaces deploy target
└── pyproject.toml          ← Pinned deps, ruff/mypy/pytest config
```

## Engineering practices

- **Strict typing** — `mypy --strict` over the whole `src/` tree, Pydantic for
  every data structure crossing a module boundary.
- **Lint + format** — ruff with bugbear, simplify, pyupgrade, pep8-naming, annotations.
- **CI** — every PR runs ruff, ruff-format check, mypy strict, and pytest.
- **Pre-commit hooks** — same checks locally before commit.
- **Reproducible ingestion** — raw PubMed metadata persisted to SQLite so
  re-chunking and re-embedding don't require re-fetching.
- **Trace-first observability** — every query produces a `QueryTrace` with
  per-step latency and token counts.
- **Versioned prompts** — `PROMPT_VERSION` in `agent.prompts` ensures eval results
  are tied to a specific prompt revision.

## Roadmap

- [ ] Cost vs quality embedding ablation (`bge-small` vs OpenAI `text-embedding-3-large`)
- [ ] Multi-topic ingestion (extend beyond GLP-1 / CV)

## License

MIT
