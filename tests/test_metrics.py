"""Tests for retrieval metrics."""

from __future__ import annotations

import math

from medrag.eval.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
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
