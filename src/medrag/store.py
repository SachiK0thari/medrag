"""SQLite persistence for Paper and Chunk records.

Thin wrapper around stdlib sqlite3. Each row carries the full Pydantic JSON
dump in a `json` column so evolving Paper/Chunk is a one-sided change — only
columns we actually query on (pmid, chunk_id, section) are promoted.

This is the system of record for BM25 and the hydration source for dense
retrieval: Pinecone only stores vectors keyed by chunk_id, and we look up
the Chunk + Paper here after a query.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import TracebackType

from medrag.config import get_settings
from medrag.models import Chunk, Paper

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    pmid   TEXT PRIMARY KEY,
    pmcid  TEXT,
    title  TEXT NOT NULL,
    year   INTEGER,
    json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    pmid        TEXT NOT NULL,
    section     TEXT,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    json        TEXT NOT NULL,
    FOREIGN KEY (pmid) REFERENCES papers(pmid)
);

CREATE INDEX IF NOT EXISTS idx_chunks_pmid ON chunks(pmid);
"""


class Store:
    """SQLite-backed persistence for papers and chunks."""

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self._path = path or settings.sqlite_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---- writes ---------------------------------------------------------- #

    def save_papers(self, papers: Iterable[Paper]) -> int:
        rows = [(p.pmid, p.pmcid, p.title, p.year, p.model_dump_json()) for p in papers]
        self._conn.executemany(
            "INSERT OR REPLACE INTO papers (pmid, pmcid, title, year, json) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def save_chunks(self, chunks: Iterable[Chunk]) -> int:
        rows = [
            (
                c.chunk_id,
                c.pmid,
                c.section,
                c.chunk_index,
                c.text,
                c.model_dump_json(),
            )
            for c in chunks
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(chunk_id, pmid, section, chunk_index, text, json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    # ---- reads ----------------------------------------------------------- #

    def load_all_papers(self) -> list[Paper]:
        cur = self._conn.execute("SELECT json FROM papers")
        return [Paper.model_validate_json(row["json"]) for row in cur]

    def load_all_chunks(self) -> list[Chunk]:
        cur = self._conn.execute("SELECT json FROM chunks ORDER BY pmid, chunk_index")
        return [Chunk.model_validate_json(row["json"]) for row in cur]

    def get_papers(self, pmids: Sequence[str]) -> dict[str, Paper]:
        if not pmids:
            return {}
        placeholders = ",".join("?" * len(pmids))
        cur = self._conn.execute(
            f"SELECT pmid, json FROM papers WHERE pmid IN ({placeholders})",
            list(pmids),
        )
        return {row["pmid"]: Paper.model_validate_json(row["json"]) for row in cur}

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        cur = self._conn.execute(
            f"SELECT chunk_id, json FROM chunks WHERE chunk_id IN ({placeholders})",
            list(chunk_ids),
        )
        return {row["chunk_id"]: Chunk.model_validate_json(row["json"]) for row in cur}

    def count_papers(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])

    def count_chunks(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
