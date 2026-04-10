"""Retrieval evaluation metrics.

All metrics take a list of retrieved PMIDs (in rank order) and a set of
gold-standard relevant PMIDs, and return a scalar in [0, 1].
"""

from __future__ import annotations

import math


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
