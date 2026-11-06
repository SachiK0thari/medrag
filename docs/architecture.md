# Architecture

## System overview

```
                 ┌─────────────────────────────────────────────────────┐
                 │                    Ingestion (offline)              │
                 │                                                     │
   PubMed E-utils│  esearch ──▶ efetch (XML) ──▶ Paper records         │
        │        │                          │                          │
        │        │                          ▼                          │
   PMC E-utils   │   PMC full text (JATS XML) where PMC has it         │
                 │                          │                          │
                 │                          ▼                          │
                 │   Section-aware chunker (≈384 tokens, max 512)      │
                 │                          │                          │
                 │                          ▼                          │
                 │   bge-small-en-v1.5 embeddings (384-d, normalized)  │
                 │                          │                          │
                 │           ┌──────────────┴──────────────┐           │
                 │           ▼                             ▼           │
                 │   SQLite (Paper, Chunk)            Pinecone index   │
                 └─────────────────────────────────────────────────────┘

                 ┌─────────────────────────────────────────────────────┐
                 │                  Query time (online)                │
                 │                                                     │
   user question │     ┌──────────────┐  LLM planner → the question    │
        ─────────┼────▶│  decompose   │  verbatim, plus sub-queries    │
                 │     └──────┬───────┘  only if it has several parts  │
                 │            ▼                                        │
                 │     ┌──────────────┐  for each query:               │
                 │     │  retrieve    │  BM25 ┐                        │
                 │     │              │       ├─ RRF ─▶ rerank (bge)   │
                 │     │              │  dense┘                        │
                 │     │              │  then merge the results and    │
                 │     │              │  rerank the pool against the   │
                 │     │              │  original question             │
                 │     └──────┬───────┘                                │
                 │            ▼                                        │
                 │     ┌──────────────┐                                │
                 │     │   reflect    │  sufficient? → answer; if not, │
                 │     │              │  a follow-up search (max 2)    │
                 │     └──────┬───────┘                                │
                 │            ▼                                        │
                 │     ┌──────────────┐                                │
                 │     │   answer     │  cited Pydantic Answer         │
                 │     └──────┬───────┘                                │
                 │            ▼                                        │
                 │       FastAPI + Streamlit                           │
                 └─────────────────────────────────────────────────────┘
```

## Why these choices

**PubMed/PMC over Google Search.** Structured metadata, free, rate-limited but
predictable, gives MeSH terms for free, and PMC OA gives clean full-text XML
without PDF parsing pain. Aligns with how a real medical-affairs tool would be built.
The first ingestion run did not get that far: a parser bug made it discard
every PMC body, so the first eval ran on 902 chunks of abstracts. With the
parser fixed the corpus was rebuilt in place (`make reingest`), and 278 of the
498 papers now carry full text, 8,647 chunks in all.

**Chunks sized for the embedder.** The chunker packs paragraphs up to about
384 tokens and splits a longer paragraph on sentence boundaries. Its budget is
in characters, and a results paragraph dense with numbers runs to far more
tokens per character than prose, so ingestion also counts each chunk with the
embedding model's own tokenizer and halves any that exceed its 512-token
limit. The embedder truncates silently, and the chunks it would have cut are
the ones holding the hazard ratios.

**bge-small-en-v1.5 embeddings.** 384-dim, runs on CPU, top of MTEB at its size
class, no API cost. Keeps the per-query cost floor at the LLM calls alone.

**Hybrid retrieval (BM25 + dense + RRF).** Dense alone misses exact-match queries
(drug names, trial acronyms like LEADER, SUSTAIN-6). Sparse alone misses paraphrase.
RRF is parameter-free and competitive with learned fusion. On the current
corpus fused retrieval scores recall@10 0.33, against 0.29 for BM25 and 0.32
for dense.

**Cross-encoder reranker.** bge-reranker-v2-m3 re-scores the fused candidates
against the query. It lifts MRR from 0.83 to 0.87 and recall@10 from 0.33 to
0.40 (95% CI on the recall gain 0.03 to 0.10). On the abstracts-only corpus its
value was ordering alone. It is also the slowest local step, about 4 seconds of
the 5.7-second single-shot latency.

**LangGraph for the agent.** An explicit state machine (decompose → retrieve →
reflect → answer) in which every node appends to a per-step trace. That trace is
why the agent is in the shape it is. On the abstracts-only corpus the first
version scored 0.20 recall@10 below the single-shot pipeline it wraps, and the
trace showed its sub-queries were finding more gold papers than the baseline and
then losing them at the merge.

**One rerank over the merged pool.** Each query's candidates are reranked
against that query, but the merged pool is then re-scored against the user's
original question. The first version sorted the merged pool by the per-query
scores, and cross-encoder scores computed against different queries are not
comparable, so a strong match for a tangential sub-query outranked a moderate
match for the actual question. That one change closed the recall gap on the
abstracts-only corpus, and on full text it recovers 0.07 recall and lifts MRR
from 0.72 to 0.82. See [`eval_findings.md`](eval_findings.md).

**A planner that is allowed to do nothing.** The planner is asked for the
smallest set of queries, keeps trial and drug names verbatim, and the user's
question is always searched as asked. Told to write 2 to 4 sub-queries, the first
planner padded every question with aspects nobody asked about. Given the choice,
it adds one keyword restatement of the question and nothing else. Over three
runs on the full-text corpus that scores the same as four sub-queries with the
fixed merge, at half the latency.

**OpenRouter + DeepSeek V4.1 Flash.** Cheap, fast, and the client speaks the
OpenAI-compatible API, so swapping to Claude or GPT is one `LLM_MODEL` change.

**Pinecone.** Serverless, so the demo has no vector store to host, and the free
tier covers this corpus. At 8,647 chunks an in-process index would be just as
fast; a managed store is the part that would not change at a larger scale.

## Benchmarks

Per-config retrieval quality, faithfulness, cost and latency are in the
"Key finding" section of the README, and the analysis of why the first agentic
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
