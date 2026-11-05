# Why the agentic loop regressed retrieval, and what fixed it

The first version of the LangGraph agent (decompose → retrieve → reflect →
answer) scored **worse** on retrieval than the plain hybrid + rerank pipeline
it wraps. This document records how the regression was attributed, what each
fix bought, and how far the benchmark itself can be trusted.

Setup: 33 questions × 10 configs, one run per config, DeepSeek V4.1 Flash via
OpenRouter with reasoning off, temperature 0.1. Every number below is
regenerated offline from two committed files (`data/eval_results.csv`,
`data/eval_traces.jsonl`):

```bash
python scripts/analyze_eval.py
```

## Headline

| Config | Recall@10 | Recall@10, strict gold | MRR | p50 latency | Cost / query |
|---|---|---|---|---|---|
| hybrid_reranked (baseline, no LLM in retrieval) | 0.652 | 0.750 | 0.658 | 6.6 s | $0.00026 |
| agentic v1 (original loop) | 0.449 | 0.611 | 0.439 | 26.1 s | $0.00074 |
| agentic_v2 (fixes below) | 0.657 | 0.769 | 0.661 | 16.9 s | $0.00064 |

1. **v1 was a regression.** Recall fell by 0.20 (95% CI −0.33 to −0.08). It
   was worse than the baseline on 16 of 33 questions and better on 5, at
   4× the latency and 2.8× the cost.
2. **v2 removes the regression.** Its recall difference from the baseline is
   +0.005 (95% CI −0.035 to +0.040).
3. **v2 does not beat the baseline.** It returns the same papers for 2.6× the
   latency and 2.5× the cost. On this question set the agentic loop is not
   worth what it costs; the last two sections say why and what would change
   that.

## Two gold sets

Gold papers were picked by an LLM judge from a pooled candidate set: the top
15 papers from BM25 plus the top 15 from dense retrieval, shuffled
(`scripts/curate_gold.py`). That gave 90 labels over 33 questions.

I then read 28 of those labels across 10 questions. None were irrelevant, but
only 14 directly answered the question. The other 14 were on topic without
containing the answer, for example a cost-effectiveness paper that mentions a
trial result in passing. Ten of those 14 weak labels came from the dense-only
part of the pool.

So the tables report two sets:

| Gold set | Definition | Size |
|---|---|---|
| Lenient | every curated label | 33 questions, 90 papers |
| Strict | graded "directly answers" | 18 questions, 34 papers |

The strict grades come from a second, zero-shot judge that reads the full
abstract and never sees my verdicts (`scripts/grade_gold.py`). On the 28
labels I reviewed it agreed with me on 22 (Cohen's κ = 0.57). All six
disagreements went one way: I said "direct" and the judge said "related". The
strict set is therefore conservative, and my grade overrides the judge
wherever both exist. Fifteen questions have no strict gold paper and drop out
of the strict column.

Recall differences are reported with a paired bootstrap 95% confidence
interval over questions. With 33 questions, most differences between
neighbouring configs are inside the noise, and the interval says which are
not.

## One ablation per hypothesis

Each variant changes exactly one thing relative to v1. `MedRagAgent` exposes
each knob as a constructor argument and `medrag.eval.configs` registers one
eval config per knob.

| Variant | What changed | Recall@10 | Δ vs baseline (95% CI) | Strict recall@10 | Gold surfaced by any sub-query | Gold in final top 10 |
|---|---|---|---|---|---|---|
| hybrid_reranked | baseline | 0.652 | — | 0.750 | — | 57 / 90 |
| agentic (v1) | nothing | 0.449 | −0.202 (−0.328 to −0.076) | 0.611 | 69 / 90 | 40 / 90 |
| agentic_keep_orig | question also searched verbatim | 0.424 | −0.227 (−0.369 to −0.091) | 0.546 | 75 / 90 | 37 / 90 |
| agentic_no_reflect | reflect loop disabled | 0.510 | −0.141 (−0.247 to −0.040) | 0.676 | 72 / 90 | 44 / 90 |
| agentic_rerank_orig | merged pool re-scored against the question | 0.672 | +0.020 (−0.056 to +0.101) | 0.787 | 75 / 90 | 59 / 90 |
| agentic_faithful | v2 planner prompt | 0.662 | +0.010 (−0.071 to +0.091) | 0.759 | 64 / 90 | 58 / 90 |
| agentic_v2 | all three fixes | 0.657 | +0.005 (−0.035 to +0.040) | 0.769 | 71 / 90 | 58 / 90 |

### The mechanism: gold is found, then discarded at the merge

The v1 planner is told to write "2–4 focused sub-queries", each on "a distinct
aspect (mechanism, trial outcomes, safety, guidelines)". It wrote four for
every one of the 33 questions. Between them those sub-queries surface 69 of
the 90 gold papers, more than the 57 the baseline has in its top 10. Coverage
is not the problem.

The problem is the merge. v1 reranks each sub-query's candidates against
*that sub-query*, concatenates the lists and sorts by score. A cross-encoder
score is only meaningful relative to the query it was computed for, so a
strong match for a tangential sub-query outranks a moderate match for the
actual question. Only 40 of the 69 surfaced gold papers survive into the final
top 10, and 21 of 33 questions lose a gold paper that one of their own
sub-queries had retrieved.

q02 is the clean example. *"How did semaglutide affect MACE in the SUSTAIN-6
trial?"* has two gold papers. The planner added a trial-design query, a
MACE-components query and a retinopathy-safety query. Two of the four
sub-queries retrieved both gold papers. The merged top 10 kept one.

Two independent fixes each remove the failure:

- **Re-score the merged pool against the original question**
  (`agentic_rerank_orig`). The four drifting sub-queries stay, and 59 gold
  papers reach the final top 10. This costs one extra reranker pass and no LLM
  tokens.
- **Stop the planner padding** (`agentic_faithful`). The v2 prompt asks for the
  smallest set of queries and forbids aspects the question did not ask about.
  It returned a single query for 32 of 33 questions, so there is nothing to
  merge. It surfaces fewer gold papers (64) and keeps nearly all of them (58).

`agentic_keep_orig` is the control that confirms the diagnosis. Searching the
user's question verbatim raises the surfaced count to 75, and the final count
*falls* to 37, because the extra results enter the same broken merge.

### Reflection is not the cause, and it does not help

- The reflect node asked for a follow-up search on 16 of 33 questions in v1
  and 12 of 33 in v2. Across those, the follow-up changed recall@10 on one
  question in v1 (worse) and two in v2 (one better, one worse).
- `agentic_no_reflect` scores 0.510 against v1's 0.449, which looks like a
  gain. It is not. The two configs are identical up to the first merge, and
  at that point they scored 0.510 and 0.465. That 0.045 gap is planner
  sampling noise at temperature 0.1, and it is a useful estimate of the noise
  floor for any single agentic run.
- The follow-up query is the critic's complaint used verbatim as a search
  string, for example *"No source reports the specific SUSTAIN-6 MACE hazard
  ratio/result; ..."*. That is a description of an absence, not a query.
- Reflection is the expensive step. The reflect call carries every retrieved
  source, so v1 uses 4,158 input tokens per question with reflection and 1,759
  without it. The baseline uses 1,641.

### What v2 actually does

v2 combines the three knobs. In practice the planner returns one keyword
restatement, which is searched alongside the verbatim question, and the merged
pool is reranked against the question. Recall matches `agentic_faithful`; MRR
is higher (0.661 against 0.635, and 0.717 against 0.602 on strict gold),
because ranking against the user's natural-language question works better than
ranking against the planner's keyword version of it.

## The non-agentic stages

| Config | Recall@10 | Strict recall@10 | MRR | Strict MRR | p50 latency |
|---|---|---|---|---|---|
| bm25_only | 0.490 | 0.620 | 0.408 | 0.407 | 1.2 s |
| dense_only | 0.586 | 0.546 | 0.592 | 0.432 | 1.9 s |
| hybrid_rrf | 0.601 | 0.722 | 0.521 | 0.444 | 1.5 s |
| hybrid_reranked | 0.652 | 0.750 | 0.658 | 0.723 | 6.6 s |

- **Hybrid beats either retriever alone under both gold sets.** That is the
  robust result.
- **BM25 against dense is not robust.** Dense wins on lenient gold and BM25
  wins on strict gold, which is what the spot-check predicts: the dense pool
  contributed most of the on-topic-but-not-answering labels. I make no claim
  either way.
- **The reranker's value is ordering, not coverage.** It adds 0.03 to 0.05
  recall, inside the noise, and lifts strict MRR from 0.444 to 0.723. It also
  accounts for about 5 seconds of latency on a laptop GPU.

## Faithfulness

RAGAS faithfulness was scored on every answer (330 of 330) and falls between
0.87 and 0.93 for all ten configs. I do not read anything into the differences
between configs: the answer prompt is shared, the judge is the same model that
wrote the answers, and the spread is small.

## What this says about agentic RAG here

The fixed agent matches the single-shot retriever and costs more. The reason
is in the question set. Given a prompt that allows it to leave a question
alone, the planner judged 32 of 33 questions atomic. Decomposition can only
add recall when a question has independent parts, and this benchmark has
almost none, so it cannot show the agent winning. It can show the agent
losing, and v1 did.

What the loop did buy is the per-step trace. Every table in this document
comes from it, and without it the regression would still read as "agentic RAG
is worse" instead of "the merge step compares scores that are not comparable".

## Next steps the data points to

1. **Route instead of always looping.** When the planner returns one query,
   run the baseline path on the user's question and skip reflection. That
   should give baseline quality at baseline cost plus one small planner call.
2. **Fix or remove the follow-up.** Have the critic emit a search query, not a
   complaint, or drop reflection until there are questions that need it.
3. **Add multi-hop questions.** Comparative questions such as "semaglutide
   against dulaglutide for MACE" are where decomposition should earn its
   cost. Until the set contains them, that claim is untested.
4. **Repeat the agentic runs.** One run per config leaves a 0.045 noise floor.
   Three to five runs would tighten every agentic interval above.

## Caveats

- One run per config. The baseline is deterministic and the agentic configs
  are not.
- Gold labels are LLM-curated, and the curating judge saw only the first 500
  characters of each abstract. The human check covers 28 of 90 labels.
- The strict set has 18 questions, so its intervals are wide. v1's strict
  recall deficit (−0.139, 95% CI −0.333 to +0.074) does not reach significance
  on its own; its MRR falls from 0.723 to 0.472.
- Latency includes network time and was measured on an Apple Silicon laptop
  with the reranker on the GPU.
- An earlier version of this analysis (see git history) used a 30-question
  gold set drawn from a BM25-only pool, and ran with a client bug that left
  the model's hidden reasoning on. There the planner prompt looked like the
  dominant fix (0.717) and the merge fix secondary (0.678). With pooled gold
  and reasoning off, each fix alone closes the gap. The v1 regression itself
  reproduced in both setups.
