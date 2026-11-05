# Architecture

## System overview

```
                 ┌─────────────────────────────────────────────────────┐
                 │                    Ingestion (offline)              │
                 │                                                     │
   PubMed E-utils│  esearch ──▶ efetch (XML) ──▶ Paper records         │
        │        │                          │                          │
        │        │                          ▼                          │
   PMC OAI-PMH   │   Optional PMC full-text fetch                       │
                 │                          │                          │
                 │                          ▼                          │
                 │   Recursive section-aware chunker (≈384 tokens)      │
                 │                          │                          │
                 │                          ▼                          │
                 │   bge-small-en-v1.5 embeddings (384-d, normalized)   │
                 │                          │                          │
                 │           ┌──────────────┴──────────────┐            │
                 │           ▼                             ▼            │
                 │   SQLite (Paper, Chunk)            Pinecone index    │
                 └─────────────────────────────────────────────────────┘

                 ┌─────────────────────────────────────────────────────┐
                 │                  Query time (online)                 │
                 │                                                     │
   user question │     ┌──────────────┐                                 │
        ─────────┼────▶│  decompose   │  LLM planner → sub-queries      │
                 │     └──────┬───────┘                                 │
                 │            ▼                                         │
                 │     ┌──────────────┐  parallel per sub-query         │
                 │     │  retrieve    │  BM25 ┐                         │
                 │     │              │       ├─ RRF ─▶ rerank (bge)    │
                 │     │              │  dense┘                         │
                 │     └──────┬───────┘                                 │
                 │            ▼                                         │
                 │     ┌──────────────┐                                 │
                 │     │   reflect    │  sufficient? → loop or proceed  │
                 │     └──────┬───────┘                                 │
                 │            ▼                                         │
                 │     ┌──────────────┐                                 │
                 │     │   answer     │  cited Pydantic Answer          │
                 │     └──────┬───────┘                                 │
                 │            ▼                                         │
                 │       FastAPI + Streamlit                            │
                 └─────────────────────────────────────────────────────┘
```

## Why these choices

**PubMed/PMC over Google Search.** Structured metadata, free, rate-limited but
predictable, gives MeSH terms for free, and PMC OA gives clean full-text XML
without PDF parsing pain. Aligns with how a real medical-affairs tool would be built.

**bge-small-en-v1.5 embeddings.** 384-dim, runs on CPU, top of MTEB at its size
class, no API cost. Keeps the per-query cost floor at the LLM calls alone.

**Hybrid retrieval (BM25 + dense + RRF).** Dense alone misses exact-match queries
(drug names, trial acronyms like LEADER, SUSTAIN-6). Sparse alone misses paraphrase.
RRF is parameter-free and competitive with learned fusion.

**Cross-encoder reranker.** bge-reranker-v2-m3 boosts precision@k significantly
for the cost of one extra forward pass on the top-30 candidates. Cheap quality win.

**LangGraph for the agent.** Explicit state machine, easy to inspect, easy to add
nodes. Decompose → retrieve → reflect → answer captures the "innovative algorithm"
framing without being magic.

**OpenRouter + DeepSeek V4.1 Flash.** Cheap, fast, and the client speaks the
OpenAI-compatible API, so swapping to Claude or GPT is one `LLM_MODEL` change.

**Pinecone.** Recognizable on a resume, free tier handles this corpus, serverless
spin-up matches a real production deployment.

## Benchmarks

Per-config retrieval quality, faithfulness, cost and latency are in the
Evaluation section of the README, and the analysis of why the first agentic
loop regressed is in [`eval_findings.md`](eval_findings.md). `make eval`
writes per-question rows to `data/eval_results.csv` and per-step agent traces
to `data/eval_traces.jsonl`; `python scripts/analyze_eval.py` turns those two
files into every table quoted in the docs.

## Module map

| Module | Responsibility |
|---|---|
| `medrag.models` | Pydantic types (single source of truth) |
| `medrag.config` | All env vars and tunables |
| `medrag.ingestion.pubmed` | Async E-utilities client |
| `medrag.ingestion.pmc` | PMC OA full-text enrichment |
| `medrag.ingestion.chunker` | Section-aware recursive splitter |
| `medrag.ingestion.embedder` | bge embedding wrapper |
| `medrag.retrieval.bm25` | Sparse retrieval |
| `medrag.retrieval.dense` | Pinecone client |
| `medrag.retrieval.hybrid` | RRF fusion |
| `medrag.retrieval.reranker` | Cross-encoder rerank |
| `medrag.agent.llm` | OpenRouter wrapper, token + cost tracking |
| `medrag.agent.prompts` | Versioned prompt templates |
| `medrag.agent.graph` | LangGraph state machine |
| `medrag.eval.metrics` | Recall@k, MRR, nDCG |
| `medrag.eval.questions` | Held-out question set |
| `medrag.eval.harness` | Per-config evaluation runner |
| `medrag.api.main` | FastAPI service |
| `medrag.ui.app` | Streamlit demo |
