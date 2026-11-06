# What the eval shows: a merge bug in the agent, and what full text changed

The first version of the LangGraph agent (decompose → retrieve → reflect →
answer) scored **worse** on retrieval than the plain hybrid + rerank pipeline
it wraps. This document records how that was attributed, what each fix bought,
what changed when the corpus went from abstracts to full text, and how far the
benchmark itself can be trusted.

Setup for both runs: 33 questions × 10 configs, DeepSeek V4.1 Flash via
OpenRouter with reasoning off, temperature 0.1.

| | Run 1 | Run 2 (current) |
|---|---|---|
| Corpus | 498 papers, abstracts only, 902 chunks | the same 498 papers, 278 with full text, 8,647 chunks |
| Answer prompt | each source cut to 600 characters | each source chunk in full |
| Runs | one per config | one per config, and three for `agentic`, `agentic_rerank_orig` and `agentic_v2` |
| Scored on | the first 10 distinct papers of the ranking | the papers behind the 10 chunks the answer is written from |
| Files | `data/eval_results_abstracts.csv`, `data/eval_traces_abstracts.jsonl` | `data/eval_results.csv`, `data/eval_traces.jsonl` and their `_rep2`, `_rep3` files |

Every number below regenerates offline from those committed files:

```bash
python scripts/analyze_eval.py                        # run 2
python scripts/analyze_eval.py --baseline agentic     # run 2, deltas against v1
python scripts/analyze_eval.py --baseline agentic_v2  # run 2, deltas against v2
python scripts/analyze_eval.py --baseline dense_only  # run 2, deltas against dense
python scripts/analyze_eval.py --results data/eval_results_abstracts.csv \
    --traces data/eval_traces_abstracts.jsonl --no-extension    # run 1
```

Run 1 is where the regression was found and diagnosed, and its analysis is
kept in full in the second half. Run 2 is the system as it is now.

## Summary

1. **v1 is a regression on both corpora.** It merges its sub-queries' results
   by cross-encoder scores that were computed against different queries. On
   abstracts that cost 0.20 recall. On full text it costs 0.057 (95% CI 0.015
   to 0.102, three runs) and ranking: MRR 0.72 against the baseline's 0.87.
2. **Either fix restores the baseline on both corpora.** Re-scoring the merged
   pool against the question recovers 0.074 recall over v1 (0.034 to 0.118).
   On full text this and the v1 deficit are measured on the extended gold
   set. The original 90 labels alone do not separate v1 from its fixes.
3. **v2, which ships, matches the baseline on both corpora.** On full text its
   recall differs by +0.001 (−0.019 to +0.018) over three runs, for 2.5× the
   latency. The agent does not beat single-shot retrieval on this question
   set.
4. **Full text fixed the answers.** v2 answers quoting a 95% confidence
   interval went from 5 of 33 to 17, and answers saying the sources lack the
   information went from 17 of 33 to 3.
5. **Retrieval scores from the two runs cannot be compared.** The gold labels
   were chosen on the abstracts corpus. On full text, 1.3 of the 8 papers in
   the baseline's context carry one of those labels, and the rest had to be
   graded afresh.

## Run 2: the full-text corpus

### What changed between the runs

- **The corpus.** A parser bug had made the first ingestion discard every PMC
  article body. With it fixed, `python scripts/ingest.py --in-place` rebuilt
  the same 498 papers: 278 now carry full text, and the index went from 902
  chunks to 8,647. The chunker was fixed on the way: it never split a
  paragraph longer than the target, and it dropped text before the first
  section header. No chunk now exceeds the embedder's 512-token limit.
- **The answer prompt.** Each source used to be cut to its first 600
  characters. An abstract states its result at the end: of the 113 abstracts
  whose first chunk carries an effect estimate, the cut left it visible in
  15. Sources now reach the model in full.
- **What is scored.** Run 1 scored the first 10 distinct papers of each
  config's ranking. With about two chunks per paper that was close to what
  the answer saw. With 28 chunks in a typical full-text paper it is not: the
  answer is written from 10 chunks, which are about 8 papers, and configs
  returned rankings of different lengths (10 chunks from one, 30 or 40 from
  another), so they were being compared on lists of 8 to 10 papers. Run 2
  scores every config on the same thing, the distinct papers behind the 10
  chunks its answer was written from. A gold paper ranked below those cannot
  be cited, whatever the config did with it.
- **Repeats.** The agentic configs are not deterministic. The three that the
  conclusions rest on were run three times, and a config's score on a
  question is its mean over its runs. The single-shot configs retrieve
  deterministically and were run once.

### The original gold labels are incomplete on this corpus

Gold papers were picked from a pool: the top 15 papers from BM25 and from
dense retrieval on the abstracts corpus, each shown to the curating judge as
the first 500 characters of its abstract. On full text that leaves most of
what a config retrieves without a usable label. Per question, in the main
run:

| Config | Papers in the answer's context | With a gold label | Judged, passed over | Never judged |
|---|---|---|---|---|
| bm25_only | 8.0 | 0.7 | 2.4 | 5.0 |
| dense_only | 7.6 | 1.2 | 3.6 | 2.8 |
| hybrid_reranked | 8.0 | 1.3 | 3.0 | 3.7 |
| agentic (v1) | 8.1 | 1.0 | 2.4 | 4.7 |
| agentic_v2 | 7.9 | 1.3 | 3.0 | 3.7 |

So I had them graded (`scripts/extend_gold.py`), on the scale the strict set
already uses. The candidates were the pool rebuilt on the full-text corpus by
the same rule, every paper in any config's top 10 in any run, and the gold
papers themselves. The judge read each abstract and the two body passages
that best match the question.

| Before the extension | Papers | Graded direct | Related | Irrelevant |
|---|---|---|---|---|
| Never judged | 977 | 224 | 595 | 158 |
| Judged, passed over | 415 | 99 | 263 | 53 |
| Gold, graded "related" on the abstract | 35 | 13 | 22 | 0 |

The counts are question–paper pairs: 1,427 of them over 315 distinct papers,
with 336 graded direct. Thirteen of the 336 were already gold, so the
extension adds 323 labels to the original 90.

A quarter of the papers the curation passed over on their abstract answer the
question once the body is read. The tables below therefore report two gold
sets:

| Gold set | Definition | Size |
|---|---|---|
| Original | the 90 curated labels, unchanged | 33 questions, 90 papers |
| Extended | original plus every paper graded direct since | 33 questions, 413 papers |

Recall against the original labels is a lower bound, because answering
papers count as misses. Recall against the extended set is not comparable
with run 1 either: there are 12.5 gold papers per question, not 2.7, so a
context of 8 papers cannot reach a recall of 1.0. For that reason the tables
from `analyze_eval.py` also give the plain count of gold papers in the
context. **Neither gold set says whether full text made paper-level retrieval
better or worse than run 1.** What they support is the comparison between
configs within run 2.

The extension labels come from one LLM judge. I hand-checked a random sample
of 30 of the 336 papers it graded direct (`HUMAN_CHECK` in
`src/medrag/eval/gold_extension.py`), reading the same abstract and passages
the judge read: 27 answer their question, 3 are only on topic and none is
irrelevant. That is 90% agreement, and with 30 papers the true rate could be
anywhere from 74% to 97% (Wilson 95% interval). The three stay in the
extended set: the sample measures the error, and removing only the errors it
happened to draw would not make the other labels better. The 880 papers the
judge graded related were not sampled. The original 90 labels and their
grades are untouched.

### Headline

| Config | Runs | Recall@10, extended gold | Δ vs baseline (95% CI) | Recall@10, original gold | MRR, extended | p50 latency | Cost / query |
|---|---|---|---|---|---|---|---|
| bm25_only | 1 | 0.288 | −0.107 (−0.141 to −0.075) | 0.293 | 0.605 | 1.6 s | $0.00066 |
| dense_only | 1 | 0.320 | −0.076 (−0.134 to −0.024) | 0.470 | 0.758 | 1.7 s | $0.00057 |
| hybrid_rrf | 1 | 0.332 | −0.064 (−0.101 to −0.030) | 0.424 | 0.826 | 1.8 s | $0.00065 |
| hybrid_reranked (baseline) | 1 | 0.396 | — | 0.510 | 0.869 | 5.7 s | $0.00073 |
| agentic (v1) | 3 | 0.338 | −0.057 (−0.102 to −0.015) | 0.446 | 0.716 | 20.8 s | $0.00136 |
| agentic_keep_orig | 1 | 0.366 | −0.030 (−0.080 to +0.015) | 0.470 | 0.718 | 20.6 s | $0.00152 |
| agentic_no_reflect | 1 | 0.373 | −0.023 (−0.073 to +0.028) | 0.419 | 0.672 | 19.9 s | $0.00081 |
| agentic_rerank_orig | 3 | 0.412 | +0.016 (−0.023 to +0.053) | 0.458 | 0.818 | 26.3 s | $0.00137 |
| agentic_faithful | 1 | 0.385 | −0.011 (−0.049 to +0.025) | 0.444 | 0.837 | 8.2 s | $0.00123 |
| agentic_v2 (ships) | 3 | 0.397 | +0.001 (−0.019 to +0.018) | 0.497 | 0.875 | 14.1 s | $0.00128 |

### The single-shot stages

- **The reranker now adds recall, not only ordering.** Each single-shot
  config is significantly below hybrid + rerank on the extended set. Against
  fused retrieval without reranking, the reranker gains 0.064 recall (95% CI
  0.030 to 0.101) and lifts MRR from 0.83 to 0.87. In run 1 its recall gain
  was inside the noise.
- **Fusion over dense alone is inside the noise.** Hybrid RRF is 0.012 above
  dense on the extended set (95% CI −0.043 to +0.075) and below it on the
  original labels (0.424 against 0.470). Run 1's "hybrid beats either
  retriever alone" does not carry over.
- **BM25 alone scores lowest under every gold set here.** A keyword now
  matches in many article bodies, and 5.0 of the 8 papers in its context are
  ones the curation never judged, the most of any config.
- **Its gap to dense is not settled.** On the extended set BM25 is 0.031
  below dense (95% CI −0.094 to +0.039) and on the strict version the two
  tie (0.286 against 0.290). On the original labels dense is well ahead
  (0.470 against 0.293), and those labels lean towards the dense pool. In
  run 1 the order flipped between gold sets, so I still make no claim about
  the pair.

### The agentic loop

- **v1 is a regression on full text too.** Over three runs its recall is
  0.057 below the baseline (95% CI 0.015 to 0.102) and its MRR is 0.716
  against 0.869, so the first answering paper sits lower in the context. In
  the main run its sub-queries surface 54 of the 90 originally labelled gold
  papers and 33 reach the answer's context; the baseline puts 43 there.
- **v1 is also the noisy config.** Its recall in the three runs was 0.293,
  0.354 and 0.369. The two configs that re-score the merged pool barely move
  (0.408 to 0.416 and 0.388 to 0.402). Merging by scores that are not
  comparable makes the outcome depend on which sub-queries the planner
  happens to write.
- **The merge fix alone restores the baseline.** `agentic_rerank_orig` keeps
  v1's four sub-queries and re-scores the merged pool against the question.
  Recall is 0.412: 0.074 above v1 (0.034 to 0.118) and level with the
  baseline (+0.016, −0.023 to +0.053). MRR recovers to 0.818.
- **So does the planner fix.** `agentic_faithful` returned a single query for
  all 33 questions, so there is nothing to merge, and scores 0.385 (−0.011,
  −0.049 to +0.025; one run).
- **All of that is on the extended gold set. The original 90 labels alone do
  not separate v1 from its fixes.** Against them v1 is 0.064 below the
  baseline (95% CI −0.160 to +0.029) and the re-scored merge is 0.012 above
  v1 (−0.052 to +0.077). Gold lost at the merge is 21 of 90 for v1 and 19
  with the re-score, where run 1 had 29 and 16. On the 18-question strict
  subset v1 is nominally above the baseline (0.670 against 0.611; +0.059,
  −0.108 to +0.216). Those labels cover 1.3 of the 8 papers in a context, so
  the full-text regression rests on the extension labels and on the sample
  of 30 that checks them.
- **v2 matches the baseline.** Recall differs by +0.001 (−0.019 to +0.018)
  and MRR is 0.875 against 0.869, for 2.5× the latency and 1.8× the cost. In
  practice v2 searches the question and one keyword restatement of it, which
  is close to searching once.
- **Four sub-queries do not beat one.** `agentic_rerank_orig` is 0.015 above
  v2 on the extended set (−0.023 to +0.051) and 0.031 above it on the strict
  version (−0.015 to +0.074). Neither interval excludes zero.
- **Reflection still shows no benefit.** The one run without it scored 0.373,
  above all three of v1's runs (0.293 to 0.369). That is no sign that the
  follow-up search helps, and one run cannot show that it hurts. The critic
  asked for 10 follow-up searches in v1's main run (32 in run 1) and 7 in
  v2's (24).

**Two artifacts worth recording.** A first pass over this run, made before
the labels were completed and with each config scored on the first 10
distinct papers of its ranking, showed v1 level with the baseline and the
re-scored v1 ahead of v2. Neither survived. v1 looked level because papers
the curation had passed over on their abstract were being counted as misses
for every config, and the baseline retrieves more of the ones that answer
the question. The re-scored v1 looked ahead of v2 because it returns 40
chunks and was scored on 10 papers, while v2 keeps fewer chunks and was
scored on 9. Completing the labels and scoring the answer's context removed
both.

### Answers

Counted over the answer text of each run's main pass (`95% CI` or
`confidence interval` for the first column; phrases such as "do not report"
for the second):

| Config | Quotes a 95% CI, run 1 → run 2 | Says the sources lack something, run 1 → run 2 |
|---|---|---|
| hybrid_reranked | 5 → 18 of 33 | 19 → 4 of 33 |
| agentic (v1) | 10 → 18 of 33 | 26 → 6 of 33 |
| agentic_v2 | 5 → 17 of 33 | 17 → 3 of 33 |

q02 is the clean example again. Asked how semaglutide affected MACE in
SUSTAIN-6, run 1's v2 answered that the sources did not contain the hazard
ratio. Run 2's answers "HR 0.74, 95% CI 0.58–0.95". These are counts of
strings, not checks of correctness; faithfulness is the check that the
quoted figures come from the sources.

Two changes produced this and the run does not separate them: bodies are
indexed, and the prompt no longer cuts each source at 600 characters. On
nine numeric questions tried by hand, each change helped on its own.

### Faithfulness

RAGAS faithfulness rose for every config, from 0.87–0.93 in run 1 to
0.96–0.98. One of the 528 answers could not be scored. I read the rise as an
effect of the prompt change, not of retrieval: the judge always saw each
source in full, and the answerer now sees the same text. An answer written
from a 600-character excerpt could say a figure was not reported when the
rest of the chunk reported it. As before, I read nothing into the
differences between configs.

### Cost and latency

- Answers now read about 3,000 input tokens where they read 1,600, so a
  single-shot query costs $0.0006 to $0.0007 where it cost $0.0003. v2 costs
  $0.0013 and takes 14.1 s at the median.
- The four single-shot configs make the same single call on near-identical
  token counts and were billed between $0.00019 and $0.00022 per thousand
  input tokens. Cost differences of that size are OpenRouter's billing, not
  the method.

## Run 1: the abstracts-only corpus

This is the analysis that found the regression, as written at the time. The
corpus held abstracts only, each source reached the model cut to 600
characters, one run was made per config, and retrieval was scored on the
first 10 distinct papers of each config's ranking. Its numbers regenerate from
the archived files with the third command at the top.

### Headline

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
   latency and 2.5× the cost. On this question set and corpus the agentic
   loop was not worth what it cost.

### Two gold sets

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

### One ablation per hypothesis

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

#### The mechanism: gold is found, then discarded at the merge

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

#### Reflection is not the cause, and it does not help

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

#### What v2 actually does

v2 combines the three knobs. In practice the planner returns one keyword
restatement, which is searched alongside the verbatim question, and the merged
pool is reranked against the question. Recall matches `agentic_faithful`; MRR
is higher (0.661 against 0.635, and 0.717 against 0.602 on strict gold),
because ranking against the user's natural-language question works better than
ranking against the planner's keyword version of it.

### The non-agentic stages

| Config | Recall@10 | Strict recall@10 | MRR | Strict MRR | p50 latency |
|---|---|---|---|---|---|
| bm25_only | 0.490 | 0.620 | 0.408 | 0.405 | 1.2 s |
| dense_only | 0.586 | 0.546 | 0.592 | 0.419 | 1.9 s |
| hybrid_rrf | 0.601 | 0.722 | 0.521 | 0.431 | 1.5 s |
| hybrid_reranked | 0.652 | 0.750 | 0.658 | 0.723 | 6.6 s |

- **Hybrid beats either retriever alone under both gold sets.** That is the
  robust result.
- **BM25 against dense is not robust.** Dense wins on lenient gold and BM25
  wins on strict gold, which is what the spot-check predicts: the dense pool
  contributed most of the on-topic-but-not-answering labels. I make no claim
  either way.
- **The reranker's value is ordering, not coverage.** It adds 0.03 to 0.05
  recall, inside the noise, and lifts strict MRR from 0.431 to 0.723. It also
  accounts for about 5 seconds of latency on a laptop GPU.

### Faithfulness

RAGAS faithfulness was scored on every answer (330 of 330) and falls between
0.87 and 0.93 for all ten configs. I do not read anything into the differences
between configs: the answer prompt is shared, the judge is the same model that
wrote the answers, and the spread is small.

### What this said about agentic RAG on that corpus

The fixed agent matched the single-shot retriever and cost more. The reason
is in the question set. Given a prompt that allows it to leave a question
alone, the planner judged 32 of 33 questions atomic. Decomposition can only
add recall when a question has independent parts, and this benchmark has
almost none, so it cannot show the agent winning. It can show the agent
losing, and v1 did.

What the loop did buy is the per-step trace. Every table in this document
comes from it, and without it the regression would still read as "agentic RAG
is worse" instead of "the merge step compares scores that are not comparable".

## Next steps the data points to

1. **Route instead of always looping.** v2 adds one restatement of the
   question and gets baseline retrieval for 2.5× the latency. When the planner
   has nothing to split, run the single-shot path and skip reflection.
2. **Hand-check the judge's "related" grades.** The sample of 30 covered the
   papers it graded direct, and 27 held. The 880 it graded related are
   unchecked, and each one that does answer its question is scored as a miss.
3. **Judge passages, not papers.** A config is credited for a paper whichever
   of its chunks it retrieved. The traces now record the chunk IDs of every
   answer's context, so the judge could grade those chunks directly.
4. **Fix or remove the follow-up.** The critic's complaint is still used
   verbatim as a search string, and reflection still shows no benefit.
5. **Add multi-hop questions.** Comparative questions such as "semaglutide
   against dulaglutide for MACE" are where decomposition should earn its
   cost, and the set has almost none. Until it does, this benchmark can show
   the agent losing and cannot show it winning.

## Caveats

- Three runs for three agentic configs, one for the other three. The
  single-shot configs retrieve deterministically.
- Gold labels are LLM-curated, and the curating judge saw only the first 500
  characters of each abstract. The human check covers 28 of the 90 original
  labels and a sample of 30 of the 336 the extension judge graded direct.
- The extension judge saw a paper's abstract and its two best-matching body
  passages, not the whole body, so "related" can mean the answer is in a
  passage it was not shown.
- Recall is scored on papers. A config is credited for a paper whichever of
  its chunks was retrieved, including one that does not hold the answer.
- The strict set of run 1 has 18 questions, so its intervals are wide. v1's
  strict recall deficit there (−0.139, 95% CI −0.333 to +0.074) does not
  reach significance on its own; its MRR falls from 0.723 to 0.460.
- Latency includes network time and was measured on an Apple Silicon laptop
  with the reranker on the GPU.
- An earlier version of the run 1 analysis (see git history) used a
  30-question gold set drawn from a BM25-only pool, and ran with a client bug
  that left the model's hidden reasoning on. There the planner prompt looked
  like the dominant fix (0.717) and the merge fix secondary (0.678). With
  pooled gold and reasoning off, each fix alone closed the gap. The v1
  regression itself reproduced in both setups.
