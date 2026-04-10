# MedRAG — Agentic RAG over GLP-1 Cardiovascular Outcome Literature

> Medical affairs teams at GLP-1 manufacturers field hundreds of literature
> questions weekly from field medical liaisons. **MedRAG reduces a multi-hour
> manual literature review to a 15-second cited answer** over 500+ open-access
> papers on GLP-1 receptor agonists and cardiovascular outcomes.

[![ci](https://github.com/yourusername/medrag/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/medrag/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org)
[![ruff](https://img.shields.io/badge/lint-ruff-orange)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/types-mypy%20strict-blue)](https://mypy-lang.org)

**Live demo:** _(Hugging Face Spaces link goes here after deployment)_

---

## What it does

Given a clinical question (e.g. *"How did semaglutide affect MACE in SUSTAIN-6?"*),
MedRAG:

1. **Decomposes** the question into focused sub-queries with an LLM planner.
2. **Retrieves** relevant chunks from a 500+ paper PubMed/PMC corpus using a
   hybrid BM25 + dense pipeline fused via Reciprocal Rank Fusion.
3. **Reranks** with a cross-encoder (bge-reranker-v2-m3) for precision.
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

Held-out question set of 30 expert-curated GLP-1 / cardiovascular questions,
each tagged with gold-standard PMIDs. Four configurations benchmarked:

| Config | Recall@10 | nDCG@10 | RAGAS faithfulness | p50 latency | $/query |
|---|---|---|---|---|---|
| Dense only (baseline) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| BM25 only | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Hybrid (RRF) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Hybrid + rerank | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **Full agentic loop** | **_TBD_** | **_TBD_** | **_TBD_** | _TBD_ | _TBD_ |

> Numbers populated by `make eval` after running the held-out set.
> See [`scripts/run_eval.py`](scripts/run_eval.py) and the harness in
> [`src/medrag/eval/`](src/medrag/eval/).

## Stack

| Layer | Choice | Why |
|---|---|---|
| Sources | PubMed E-utilities + PMC OAI-PMH | Free, structured, MeSH-tagged |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) | 384-d, CPU-friendly, top of MTEB at size |
| Vector DB | Pinecone (serverless) | Production-grade, free tier sufficient |
| Sparse | `rank_bm25` | Pure Python, zero deps |
| Rerank | `BAAI/bge-reranker-v2-m3` (local) | Cross-encoder precision boost |
| LLM | Llama 3.3 70B via Groq | Sub-second TTFT, free tier, swappable |
| Orchestration | LangGraph | Explicit state machine for agent loop |
| API | FastAPI + Pydantic | Type-safe contracts end to end |
| UI | Streamlit | Fast to ship, deploys to HF Spaces |
| Eval | Custom harness + RAGAS | Recall@k, nDCG, faithfulness, LLM-as-judge |
| Quality | ruff + mypy strict + pytest + GH Actions | Production-grade discipline |

The LLM client (`medrag.agent.llm`) is abstracted: swap Groq for Anthropic Claude
or OpenAI GPT-4o via a single environment variable.

## Quickstart

```bash
# 1. Install
git clone https://github.com/yourusername/medrag.git
cd medrag
python -m venv .venv && source .venv/bin/activate
make dev   # installs with dev + eval extras, sets up pre-commit

# 2. Configure
cp .env.example .env
# Edit .env with your Groq, Pinecone, and (optional) NCBI API keys

# 3. Ingest
make ingest        # ~10-15 min for 500 papers on a laptop

# 4. Evaluate
make eval          # writes data/eval_results.csv

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
├── scripts/                ← ingest.py, run_eval.py
├── docs/architecture.md    ← System diagram + design rationale
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
- [ ] Replace Streamlit with Next.js + shadcn frontend
- [ ] Langfuse self-hosted tracing
- [ ] Multi-topic ingestion (extend beyond GLP-1 / CV)
- [ ] Confidence calibration via LLM-as-judge ensemble

## License

MIT
