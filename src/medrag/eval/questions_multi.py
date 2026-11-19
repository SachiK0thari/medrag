"""Multi-part split: 10 questions with independent parts, for the agent loop.

The development and test questions are almost all atomic: with routing on,
97 of 99 development answers and 19 of 20 test answers skip the agent loop,
so those sets can show the loop costing time and cannot show it earning it.
Each question here asks about two (or three) things that a single search
for the whole question is unlikely to cover together: two trials, two
drugs, two outcomes, or two populations.

They were written after the routing decision, after checking only that the
corpus holds at least five papers matching each part's keywords (the
sparsest, STRIDE, has nine). No system was run on them before the question
text was fixed. They are labelled by the same three scripts as the other
splits, and nothing in the agent or retrievers may be changed because of
results on them.

Gold PMIDs are merged in from `medrag.eval.gold_pmids_multi`, written by
`scripts/curate_gold.py --split multi`.
"""

from __future__ import annotations

import importlib

from medrag.eval.questions import EvalQuestion

# Imported by name: the module only exists once the split has been curated.
try:
    GOLD_PMIDS: dict[str, list[str]] = importlib.import_module(
        "medrag.eval.gold_pmids_multi"
    ).GOLD_PMIDS
except ModuleNotFoundError:
    GOLD_PMIDS = {}


MULTI_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        qid="m01",
        question="Compare the MACE results of SUSTAIN-6 and LEADER, and explain how the two trial populations differed.",
        category="multi_trial",
    ),
    EvalQuestion(
        qid="m02",
        question="How did the cardiovascular results of EXSCEL with exenatide differ from those of Harmony Outcomes with albiglutide?",
        category="multi_trial",
    ),
    EvalQuestion(
        qid="m03",
        question="What did the FLOW trial show about kidney outcomes with semaglutide, and what renal findings did REWIND report for dulaglutide?",
        category="multi_trial",
    ),
    EvalQuestion(
        qid="m04",
        question="How do the cardiovascular results of PIONEER 6 and SOUL for oral semaglutide compare, and why did only one of them show superiority?",
        category="multi_trial",
    ),
    EvalQuestion(
        qid="m05",
        question="What did SELECT show about heart failure events in people without diabetes, and what did STEP-HFpEF show about heart failure symptoms?",
        category="multi_trial",
    ),
    EvalQuestion(
        qid="m06",
        question="Compare the cardiovascular outcome trial results of lixisenatide in ELIXA with those of efpeglenatide in AMPLITUDE-O.",
        category="multi_drug",
    ),
    EvalQuestion(
        qid="m07",
        question="What did SURPASS-CVOT show for tirzepatide against dulaglutide, and what did the STRIDE trial show for semaglutide in peripheral artery disease?",
        category="multi_trial",
    ),
    EvalQuestion(
        qid="m08",
        question="How does the risk of atrial fibrillation with GLP-1 receptor agonists compare with their effect on stroke?",
        category="multi_outcome",
    ),
    EvalQuestion(
        qid="m09",
        question="How do GLP-1 receptor agonists change LDL cholesterol and systolic blood pressure, and how much of their MACE reduction do those changes explain?",
        category="multi_outcome",
    ),
    EvalQuestion(
        qid="m10",
        question="How does the cardiovascular benefit of GLP-1 receptor agonists in patients with established atherosclerotic cardiovascular disease compare with the evidence in primary prevention?",
        category="multi_population",
    ),
]

# Merge curated gold PMIDs in, rebuilding the models as `questions.py` does.
MULTI_QUESTIONS = [
    q.model_copy(update={"gold_pmids": list(GOLD_PMIDS.get(q.qid, []))}) for q in MULTI_QUESTIONS
]
