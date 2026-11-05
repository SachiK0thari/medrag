"""Pure-function tests for the eval harness (no LLM, no retrieval)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from medrag.eval import configs as cfg_mod
from medrag.eval.harness import (
    EvalRow,
    _percentile,
    _unique_preserving_order,
    estimate_cost_usd,
    summarize,
    write_results,
)


def _row(config: str, qid: str, **over: object) -> EvalRow:
    base = {
        "config": config,
        "qid": qid,
        "recall_at_10": 1.0,
        "mrr": 1.0,
        "ndcg_at_10": 1.0,
        "faithfulness": 0.8,
        "input_tokens": 100,
        "output_tokens": 10,
        "cost_usd": 0.001,
        "latency_ms": 1000.0,
    }
    base.update(over)
    return EvalRow(**base)  # type: ignore[arg-type]


def test_unique_preserving_order():
    assert _unique_preserving_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
    assert _unique_preserving_order([]) == []


def test_percentile_interpolates():
    vals = [10.0, 20.0, 30.0, 40.0]
    assert _percentile(vals, 0) == 10.0
    assert _percentile(vals, 100) == 40.0
    assert _percentile(vals, 50) == 25.0
    assert _percentile([], 50) == 0.0
    assert _percentile([7.0], 95) == 7.0


def test_estimate_cost_known_and_fallback_models():
    known = estimate_cost_usd(1_000_000, 1_000_000, "deepseek/deepseek-v4.1-flash")
    assert known == pytest.approx(0.15 + 0.60)
    unknown = estimate_cost_usd(1_000_000, 0, "some/unknown-model")
    assert unknown > 0.0


def test_summarize_groups_by_config_and_skips_missing_faithfulness():
    rows = [
        _row("a", "q1", recall_at_10=1.0, latency_ms=100.0, faithfulness=0.5),
        _row("a", "q2", recall_at_10=0.0, latency_ms=300.0, faithfulness=None),
        _row("b", "q1", recall_at_10=0.5, latency_ms=50.0, faithfulness=None),
    ]
    s = summarize(rows)
    assert set(s) == {"a", "b"}
    assert s["a"]["recall_at_10"] == 0.5
    assert s["a"]["faithfulness"] == 0.5  # only the scored row counts
    assert s["a"]["latency_ms_p50"] == 200.0
    assert s["a"]["n"] == 2.0
    assert s["b"]["faithfulness"] == 0.0  # no scored rows -> 0.0 sentinel


def test_write_results_blank_for_missing_faithfulness(tmp_path: Path):
    out = tmp_path / "nested" / "r.csv"
    write_results([_row("a", "q1", faithfulness=None), _row("a", "q2")], out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["faithfulness"] == ""
    assert rows[1]["faithfulness"] == "0.8"
    assert rows[0]["config"] == "a"


def test_write_results_includes_llm_call_diagnostics(tmp_path: Path):
    out = tmp_path / "r.csv"
    write_results([_row("a", "q1", llm_calls=3, truncated_calls=1)], out)
    with out.open() as f:
        row = next(csv.DictReader(f))
    assert row["llm_calls"] == "3"
    assert row["truncated_calls"] == "1"


def test_config_names_are_unique_and_include_ablations():
    names = cfg_mod.config_names()
    assert len(names) == len(set(names))
    for expected in (
        "dense_only",
        "bm25_only",
        "hybrid_rrf",
        "hybrid_reranked",
        "agentic",
        "agentic_keep_orig",
        "agentic_rerank_orig",
        "agentic_no_reflect",
        "agentic_faithful",
        "agentic_v2",
    ):
        assert expected in names


def test_build_configs_filters_and_rejects_unknown():
    picked = cfg_mod.build_configs(["agentic_v2", "bm25_only"])
    assert [c.name for c in picked] == ["agentic_v2", "bm25_only"]
    with pytest.raises(ValueError, match="unknown config"):
        cfg_mod.build_configs(["nope"])


def test_shared_components_cache_one_agent_per_knob_set(monkeypatch: pytest.MonkeyPatch):
    comps = cfg_mod._SharedComponents()
    # Avoid loading the real stack: swap in cheap sentinels.
    monkeypatch.setattr(comps, "hybrid", lambda: object())
    monkeypatch.setattr(comps, "reranker", lambda: object())
    monkeypatch.setattr(comps, "store", lambda: object())
    a1 = comps.agent(keep_original_query=True)
    a2 = comps.agent(keep_original_query=True)
    b = comps.agent(rerank_against_original=True)
    assert a1 is a2
    assert a1 is not b
    assert a1._keep_original_query is True
    assert b._rerank_against_original is True
    assert b._keep_original_query is False
