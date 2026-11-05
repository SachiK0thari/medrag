# Why the agentic loop regressed retrieval, and what fixed it

The first version of the LangGraph agent (decompose → retrieve → reflect →
answer) scored **worse** on retrieval than the plain hybrid + rerank pipeline
it was built on top of. This document records how the regression was
attributed and what each fix bought. All numbers below are recall@10 against
the 30 gold-labelled eval questions, one run per config, DeepSeek V4.1 Flash
via OpenRouter, temperature 0.1.

Reproduce with `python scripts/debug_agentic.py --variants agentic,agentic_v2`
(per-question JSONL with every sub-query and the PMIDs it surfaced), then
`python scripts/run_eval.py` for the full table.

## Headline

| Config | Recall@10 | Worse than baseline on | Mean sub-queries |
|---|---|---|---|
| hybrid_reranked (baseline, no LLM in retrieval) | 0.733 | — | 1 |
| agentic v1 (unmodified loop) | 0.467 | 17 / 30 | 3.7 |
| agentic_v2 (fixes below) | 0.733 | 1 / 30 | 1.4 |

v1 lost 0.27 recall. v2 recovers all of it, with fewer sub-queries and fewer
LLM calls than v1.

## Three hypotheses, one ablation each

Every knob below changes exactly one thing relative to v1. `MedRagAgent`
exposes each as a constructor argument; `medrag.eval.configs` registers one
eval config per knob.

| Variant | What changed | Recall@10 | Δ vs baseline | Verdict |
|---|---|---|---|---|
| agentic (v1) | nothing | 0.467 | −0.267 | — |
| agentic_no_reflect | reflect loop disabled | 0.433 | −0.300 | **H3 rejected**: reflection is not the cause |
| agentic_keep_orig | user's question always searched as sub-query 1 | 0.533 | −0.200 | partial |
| agentic_rerank_orig | merged pool re-scored against the original question | 0.678 | −0.056 | **H2 confirmed**: large effect |
| agentic_faithful | v2 planner prompt (no padding, keep named entities) | 0.717 | −0.017 | **H1 confirmed**: dominant cause |
| agentic_v2 | keep_orig + rerank_orig + faithful prompt | 0.733 | 0.000 | ships |

### H1 — Decomposition drift (dominant cause)

The v1 planner prompt demanded "2–4 focused sub-queries" each targeting "a
distinct aspect (mechanism, trial outcomes, safety, guidelines)". For a
question like *"How did semaglutide affect MACE in SUSTAIN-6?"* the model
dutifully invented a trial-design query, a statistics query and a safety
query the user never asked for. Mean sub-queries per question: 3.7. Only
3 of 30 questions were treated as atomic.

Evidence from the traces:

- 20 of the 64 gold PMIDs were never surfaced by *any* v1 sub-query. The
  paraphrases simply retrieved different papers than the question did.
- Gold hits by sub-query position were 30 / 23 / 18 / 22 for positions
  1–4. The first sub-query, the one closest to the user's wording, found the
  most gold; the padding queries found progressively less.

The fix is a prompt, not code: return the smallest set of queries (1–3),
keep trial acronyms, drug names and endpoints verbatim, and do not add
aspects the question did not ask about. That alone (agentic_faithful)
recovers 0.25 of the 0.27 lost. With the v2 prompt, 18 of 30 questions
get a single sub-query and the rest get two.

### H2 — Incomparable scores at merge (second cause)

v1 reranked each sub-query's candidates against *that sub-query*, then
concatenated the results and sorted by score. Cross-encoder scores are only
meaningful relative to the query they were computed for, so a strong match
for the tangential "guidelines" sub-query outranks a moderate match for the
actual question.

Evidence: in 14 of 30 questions, a sub-query **did** surface a gold PMID in
its own top-10, but that PMID was missing from the merged top-10. Sixteen
gold PMIDs were lost at the merge step alone. q02 is the clean example: all
four sub-queries mention "SUSTAIN-6 semaglutide", two of them found the gold
paper, and the merged list still dropped it.

The fix: after merging, re-score the whole deduplicated pool with the
cross-encoder against the original question. Even with the drifting v1
planner this brings recall to 0.678 (agentic_rerank_orig). It costs one
extra reranker pass over ≤40 chunks on CPU, no LLM tokens.

### H3 — Reflection loop (not a cause)

The reflect node asked for a follow-up query in only 3 of 30 v1 runs.
Disabling it (agentic_no_reflect) moved recall by −0.03, inside the run-to-run
noise of a stochastic planner. It is kept in v2 as a safety net for genuinely
under-retrieved questions; it is not where the recall went.

## What this says about agentic RAG here

On this benchmark the fixed agent **matches** the non-agentic retriever, it
does not beat it. That is the honest result, and the reason is in the eval
set: 28 of the 30 questions are single-hop. Decomposition can only add
recall when a question has independent parts, and the only two questions
where v2 split the query (q18, q20) were ones where the second query was a
near-paraphrase. There is nothing for the planner to win on.

What the agentic loop buys instead:

- A planner that correctly leaves atomic questions alone (18/30 single
  sub-query) while still being able to split comparative or multi-part ones.
- A reflection safety net that triggers rarely and cheaply.
- Per-step traces that made this analysis possible in one afternoon.

What it costs: roughly three LLM calls per question (decompose, reflect,
answer) against one for the baseline, so about 2× the tokens for equal
recall on this question set.

## Next steps the data points to

1. **Route, don't always decompose.** If the planner returns one sub-query,
   skip reflection and the merge rerank; the loop collapses to the baseline
   at baseline cost. Only pay for the agentic path on multi-part questions.
2. **Add multi-hop questions to the eval set.** Comparative questions
   ("semaglutide vs dulaglutide for MACE") are where decomposition should
   earn its keep. The current set cannot show that, so the claim stays
   untested until it does.
3. **Re-check v2 on the two divergent questions.** v2 lost one gold PMID on
   q20 and gained one on q25 relative to the baseline; both are single-run
   results at temperature 0.1 and should be re-run before being quoted.

## Caveats

- One run per config; the agentic variants are stochastic (temperature 0.1),
  the baseline is deterministic. Differences under ~0.03 are noise.
- Gold PMIDs were LLM-curated from a BM25 candidate pool and spot-checked,
  not clinician-labelled. See `scripts/curate_gold.py`.
- The README table reporting v1 at 0.417 was produced with an 8B model under
  rate limiting. With DeepSeek V4.1 Flash the same v1 loop scores 0.467, so
  the model choice accounts for a small part of the gap and the loop design
  for the rest.
