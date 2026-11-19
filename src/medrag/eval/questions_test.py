"""Test split: 20 questions never used to tune or debug any configuration.

The 33 questions in `medrag.eval.questions` chose v2's defaults (every
ablation was scored on them), so they are a development set. These were
written afterwards, on topics none of the 33 cover, after checking only that
the corpus holds at least five papers matching each topic's keywords. No
system was run on them before the question text was fixed, and nothing in
the agent or retrievers may be changed because of results on them.

Gold PMIDs are merged in from `medrag.eval.gold_pmids_test`, written by
`scripts/curate_gold.py --split test` exactly as the development labels are.
"""

from __future__ import annotations

import importlib

from medrag.eval.questions import EvalQuestion

# Imported by name: the module only exists once the split has been curated.
try:
    GOLD_PMIDS: dict[str, list[str]] = importlib.import_module(
        "medrag.eval.gold_pmids_test"
    ).GOLD_PMIDS
except ModuleNotFoundError:
    GOLD_PMIDS = {}


TEST_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        qid="t01",
        question="What did the SOUL trial show about oral semaglutide and major adverse cardiovascular events?",
        category="trial_outcome",
    ),
    EvalQuestion(
        qid="t02",
        question="How did semaglutide affect symptoms and physical limitation in patients with obesity-related heart failure with preserved ejection fraction?",
        category="trial_outcome",
    ),
    EvalQuestion(
        qid="t03",
        question="What did the STRIDE trial find about semaglutide in patients with peripheral artery disease?",
        category="trial_outcome",
    ),
    EvalQuestion(
        qid="t04",
        question="Are GLP-1 receptor agonists associated with a higher or lower risk of atrial fibrillation?",
        category="outcome",
    ),
    EvalQuestion(
        qid="t05",
        question="Do GLP-1 receptor agonists increase the risk of diabetic retinopathy complications?",
        category="safety",
    ),
    EvalQuestion(
        qid="t06",
        question="What is the risk of pulmonary aspiration during anesthesia in patients taking GLP-1 receptor agonists, and how should the drugs be managed before surgery?",
        category="safety",
    ),
    EvalQuestion(
        qid="t07",
        question="What cardiovascular effects of GLP-1 receptor agonists have been reported in people with type 1 diabetes?",
        category="population",
    ),
    EvalQuestion(
        qid="t08",
        question="Does the cardiovascular benefit of GLP-1 receptor agonists differ between women and men?",
        category="subgroup",
    ),
    EvalQuestion(
        qid="t09",
        question="Is the cardiovascular benefit of GLP-1 receptor agonists maintained in older adults?",
        category="subgroup",
    ),
    EvalQuestion(
        qid="t10",
        question="How do cardiovascular outcomes after GLP-1 receptor agonist treatment compare with those after bariatric surgery in patients with obesity?",
        category="comparison",
    ),
    EvalQuestion(
        qid="t11",
        question="Does combining a GLP-1 receptor agonist with an SGLT2 inhibitor give more cardiovascular benefit than either drug alone?",
        category="comparison",
    ),
    EvalQuestion(
        qid="t12",
        question="How do GLP-1 receptor agonists affect LDL cholesterol and other blood lipids?",
        category="mechanism",
    ),
    EvalQuestion(
        qid="t13",
        question="What effect do GLP-1 receptor agonists have on epicardial adipose tissue?",
        category="mechanism",
    ),
    EvalQuestion(
        qid="t14",
        question="Do GLP-1 receptor agonists slow the progression of atherosclerotic plaque in imaging studies?",
        category="mechanism",
    ),
    EvalQuestion(
        qid="t15",
        question="Are GLP-1 receptor agonists associated with fewer cardiovascular events in patients with metabolic dysfunction-associated steatotic liver disease?",
        category="population",
    ),
    EvalQuestion(
        qid="t16",
        question="What do target trial emulation studies using real-world data show about GLP-1 receptor agonists and cardiovascular outcomes?",
        category="real_world",
    ),
    EvalQuestion(
        qid="t17",
        question="How often do patients discontinue GLP-1 receptor agonists, and what happens to cardiovascular risk factors after stopping?",
        category="real_world",
    ),
    EvalQuestion(
        qid="t18",
        question="What cardiovascular data exist for newer incretin-based agents such as orforglipron, retatrutide or cagrilintide-semaglutide?",
        category="efficacy",
    ),
    EvalQuestion(
        qid="t19",
        question="Do GLP-1 receptor agonists raise resting heart rate, and is the increase clinically important?",
        category="safety",
    ),
    EvalQuestion(
        qid="t20",
        question="Are GLP-1 receptor agonists cost-effective for reducing cardiovascular risk?",
        category="economics",
    ),
]

# Merge curated gold PMIDs in, rebuilding the models as `questions.py` does.
TEST_QUESTIONS = [
    q.model_copy(update={"gold_pmids": list(GOLD_PMIDS.get(q.qid, []))}) for q in TEST_QUESTIONS
]
