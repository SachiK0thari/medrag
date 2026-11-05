"""Tests for the SQLite store."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from medrag.models import Chunk, Paper
from medrag.store import Store


@pytest.fixture
def store(tmp_path: Path, sample_paper: Paper) -> Store:
    s = Store(tmp_path / "medrag.db")
    s.save_papers([sample_paper])
    s.save_chunks(
        [
            Chunk(
                chunk_id=f"c{i}",
                pmid=sample_paper.pmid,
                text=f"chunk {i}",
                section="abstract",
                chunk_index=i,
                n_tokens=2,
            )
            for i in range(3)
        ]
    )
    return s


def test_round_trip(store: Store, sample_paper: Paper) -> None:
    assert store.count_papers() == 1
    assert store.count_chunks() == 3
    assert store.load_all_papers() == [sample_paper]
    assert [c.chunk_id for c in store.load_all_chunks()] == ["c0", "c1", "c2"]


def test_lookups_return_only_what_exists(store: Store, sample_paper: Paper) -> None:
    assert store.get_papers([sample_paper.pmid, "missing"]) == {sample_paper.pmid: sample_paper}
    assert set(store.get_chunks(["c2", "missing"])) == {"c2"}
    assert store.get_papers([]) == {}
    assert store.get_chunks([]) == {}


def test_save_replaces_existing_rows(store: Store, sample_paper: Paper) -> None:
    store.save_papers([sample_paper.model_copy(update={"title": "Revised"})])
    assert store.count_papers() == 1
    assert store.get_papers([sample_paper.pmid])[sample_paper.pmid].title == "Revised"


def test_readable_from_other_threads(store: Store) -> None:
    """The API and the UI serve each request on a different thread.

    A default sqlite3 connection raises ProgrammingError there, which broke
    every query after the first one a server handled.
    """
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.get_chunks(["c0", "c1"]), range(16)))
    assert all(set(r) == {"c0", "c1"} for r in results)
