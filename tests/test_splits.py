"""The test split must stay separate from the development questions and files."""

from __future__ import annotations

from pathlib import Path

from medrag.eval.questions import QUESTIONS
from medrag.eval.questions_test import TEST_QUESTIONS
from medrag.eval.splits import load_generated, questions_for, split_files


def test_dev_split_keeps_the_original_file_names() -> None:
    files = split_files("dev", Path("/data"))
    assert files.results == Path("/data/eval_results.csv")
    assert files.curation_audit == Path("/data/gold_curation_audit.json")
    assert files.grades_human == Path("/data/gold_spotcheck_verdicts.json")
    assert files.gold_pmids_module.name == "gold_pmids.py"
    assert files.module_name("gold_grades") == "medrag.eval.gold_grades"


def test_test_split_never_shares_a_file_with_dev() -> None:
    dev, test = split_files("dev", Path("/data")), split_files("test", Path("/data"))
    dev_paths = {v for v in vars(dev).values() if isinstance(v, Path)}
    test_paths = {v for v in vars(test).values() if isinstance(v, Path)}
    assert not dev_paths & test_paths
    assert test.results == Path("/data/eval_results_test.csv")
    assert test.module_name("gold_grades") == "medrag.eval.gold_grades_test"


def test_questions_for_each_split() -> None:
    assert questions_for("dev") is QUESTIONS
    assert questions_for("test") is TEST_QUESTIONS


def test_test_questions_are_new() -> None:
    assert len(TEST_QUESTIONS) == 20
    assert len({q.qid for q in TEST_QUESTIONS}) == 20
    assert all(q.qid.startswith("t") for q in TEST_QUESTIONS)
    dev_text = {q.question.lower() for q in QUESTIONS}
    assert not {q.question.lower() for q in TEST_QUESTIONS} & dev_text


def test_load_generated_reads_dev_modules_and_tolerates_missing_ones() -> None:
    loaded = load_generated(split_files("dev", Path("/data")), "gold_grades", "GOLD_GRADES")
    assert loaded is not None and "q01" in loaded[0]
    assert load_generated(split_files("test", Path("/data")), "no_such_module", "X") is None
