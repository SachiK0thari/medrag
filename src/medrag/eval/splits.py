"""Question splits and the files each one reads and writes.

``dev`` is the original 33 questions: every ablation was scored on them and
v2's defaults were chosen from those scores. ``test`` is 20 questions written
afterwards and never used for tuning (`medrag.eval.questions_test`).

The two splits use the same scripts with separate files, so a test-split run
can never overwrite development labels or results. Dev files keep their
original names; test files add ``_test`` before the extension.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from medrag.eval.questions import QUESTIONS, EvalQuestion

Split = Literal["dev", "test"]
SPLITS: tuple[Split, ...] = ("dev", "test")

_EVAL_PKG = Path(__file__).resolve().parent


def _suffix(split: Split) -> str:
    return "" if split == "dev" else "_test"


def questions_for(split: Split) -> list[EvalQuestion]:
    if split == "dev":
        return QUESTIONS
    from medrag.eval.questions_test import TEST_QUESTIONS

    return TEST_QUESTIONS


@dataclass(frozen=True)
class SplitFiles:
    """Where one split's gold labels, audit trails and results live."""

    split: Split
    gold_pmids_module: Path  # scripts/curate_gold.py
    gold_grades_module: Path  # scripts/grade_gold.py
    gold_extension_module: Path  # scripts/extend_gold.py
    curation_audit: Path
    grades_audit: Path
    grades_human: Path  # hand-check of curated labels, if one was saved
    extension_audit: Path
    extension_human: Path  # hand-check of extension grades, if one was saved
    results: Path  # scripts/run_eval.py; traces sit beside it

    def module_name(self, stem: str) -> str:
        """Dotted name of this split's generated module, e.g. ``gold_grades``."""
        return f"medrag.eval.{stem}{_suffix(self.split)}"


def split_files(split: Split, data_dir: Path) -> SplitFiles:
    s = _suffix(split)
    return SplitFiles(
        split=split,
        gold_pmids_module=_EVAL_PKG / f"gold_pmids{s}.py",
        gold_grades_module=_EVAL_PKG / f"gold_grades{s}.py",
        gold_extension_module=_EVAL_PKG / f"gold_extension{s}.py",
        curation_audit=data_dir / f"gold_curation_audit{s}.json",
        grades_audit=data_dir / f"gold_grades_audit{s}.json",
        grades_human=data_dir / f"gold_spotcheck_verdicts{s}.json",
        extension_audit=data_dir / f"gold_extension_audit{s}.json",
        extension_human=data_dir / f"gold_extension_spotcheck_verdicts{s}.json",
        results=data_dir / f"eval_results{s}.csv",
    )


def load_generated(files: SplitFiles, stem: str, *names: str) -> tuple[Any, ...] | None:
    """Attributes `names` of this split's generated module `stem`; None if it does not exist yet."""
    try:
        module = importlib.import_module(files.module_name(stem))
    except ImportError:
        return None
    return tuple(getattr(module, n) for n in names)
