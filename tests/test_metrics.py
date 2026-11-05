"""Tests for retrieval metrics."""

from __future__ import annotations

import math

import pytest

from medrag.eval.metrics import (
    cohen_kappa,
    mean_reciprocal_rank,
    ndcg_at_k,
    paired_bootstrap_ci,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_perfect() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_at_k_partial() -> None:
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_empty_relevant() -> None:
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 2 / 3


def test_mrr_first_position() -> None:
    assert mean_reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_mrr_second_position() -> None:
    assert mean_reciprocal_rank(["x", "a"], {"a"}) == 0.5


def test_mrr_no_hit() -> None:
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect() -> None:
    # All relevant docs at top -> nDCG = 1
    assert math.isclose(ndcg_at_k(["a", "b"], {"a", "b"}, k=2), 1.0)


def test_ndcg_partial() -> None:
    val = ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3)
    assert 0.0 < val < 1.0


def test_paired_bootstrap_ci_brackets_a_clear_difference():
    lo, hi = paired_bootstrap_ci([0.2, 0.3, 0.25, 0.2, 0.35, 0.3, 0.25, 0.2], samples=2000)
    assert 0.2 <= lo < hi <= 0.35


def test_paired_bootstrap_ci_spans_zero_for_noise():
    lo, hi = paired_bootstrap_ci([0.5, -0.5, 0.25, -0.25, 0.0, 0.0], samples=2000)
    assert lo < 0.0 < hi


def test_paired_bootstrap_ci_is_deterministic_and_handles_empty():
    deltas = [0.1, -0.2, 0.3, 0.0]
    assert paired_bootstrap_ci(deltas, samples=500) == paired_bootstrap_ci(deltas, samples=500)
    assert paired_bootstrap_ci([]) == (0.0, 0.0)


def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 2, 1, 2], [1, 2, 1, 2]) == 1.0
    # 50% observed agreement with balanced labels is exactly chance.
    assert cohen_kappa([1, 1, 2, 2], [1, 2, 1, 2]) == 0.0
    # 2x2 table [[25, 10], [20, 45]]: observed 0.70; marginals 35/65 and 45/55
    # give expected 0.35*0.45 + 0.65*0.55 = 0.515, so kappa = 0.185 / 0.485.
    a = [1] * 25 + [1] * 10 + [0] * 20 + [0] * 45
    b = [1] * 25 + [0] * 10 + [1] * 20 + [0] * 45
    assert abs(cohen_kappa(a, b) - 0.185 / 0.485) < 1e-9


def test_cohen_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1])
