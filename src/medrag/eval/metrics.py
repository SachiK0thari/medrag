"""Retrieval evaluation metrics.

All metrics take a list of retrieved PMIDs (in rank order) and a set of
gold-standard relevant PMIDs, and return a scalar in [0, 1].
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Hashable, Sequence


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for pmid in top_k if pmid in relevant)
    return hits / k


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, pmid in enumerate(retrieved, start=1):
        if pmid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    dcg = 0.0
    for i, pmid in enumerate(retrieved[:k], start=1):
        if pmid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(metric_values: list[float]) -> float:
    return sum(metric_values) / len(metric_values) if metric_values else 0.0


def paired_bootstrap_ci(
    deltas: Sequence[float], *, samples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """95% confidence interval for the mean of per-question differences.

    `deltas` holds one value per question (metric under config A minus the
    same metric under config B). Questions are resampled with replacement, so
    the interval reflects how much the mean would move with a different draw
    of questions; an interval that spans 0 means the configs are not
    distinguishable on this question set.
    """
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Chance-corrected agreement between two raters over the same items."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    a_counts, b_counts = Counter(a), Counter(b)
    expected = sum(a_counts[label] * b_counts[label] for label in a_counts) / (n * n)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)
