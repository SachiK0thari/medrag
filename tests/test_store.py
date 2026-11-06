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


def test_replace_chunks_removes_what_is_no_longer_produced(
    store: Store, sample_paper: Paper
) -> None:
    """Re-chunking a paper must not leave its old chunks behind."""
    other = sample_paper.model_copy(update={"pmid": "999"})
    store.save_papers([other])
    untouched = Chunk(
        chunk_id="other-0", pmid="999", text="t", section="abstract", chunk_index=0, n_tokens=1
    )
    store.save_chunks([untouched])

    new = [
        Chunk(
            chunk_id=cid,
            pmid=sample_paper.pmid,
            text=cid,
            section="results",
            chunk_index=i,
            n_tokens=1,
        )
        for i, cid in enumerate(["c0", "new-1"])
    ]
    stale = store.replace_chunks([sample_paper.pmid], new)

    assert stale == ["c1", "c2"]  # c0 is produced again, so it is not stale
    assert [c.chunk_id for c in store.load_all_chunks()] == ["c0", "new-1", "other-0"]
    assert store.get_chunks(["c0"])["c0"].section == "results"


def test_replace_chunks_can_leave_a_paper_with_none(store: Store, sample_paper: Paper) -> None:
    assert store.replace_chunks([sample_paper.pmid], []) == ["c0", "c1", "c2"]
    assert store.count_chunks() == 0
    assert store.count_papers() == 1
