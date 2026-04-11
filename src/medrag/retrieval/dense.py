"""Dense retrieval against Pinecone.

Pinecone holds only vectors keyed by chunk_id; the Chunk + Paper payload
lives in SQLite. On search we hydrate matches from the Store so there is a
single source of truth for record content.
"""

from __future__ import annotations

import time
from typing import Any

from medrag.config import get_settings
from medrag.ingestion.embedder import embed_texts
from medrag.models import EmbeddedChunk, RetrievalMethod, RetrievedChunk
from medrag.store import Store

_UPSERT_BATCH = 100
_INDEX_READY_POLL_SEC = 1.0
_INDEX_READY_TIMEOUT_SEC = 120.0


def _index_ready(desc: Any) -> bool:  # noqa: ANN401 — pinecone SDK response shape varies
    """Pinecone describe_index response shape has drifted between SDK versions;
    support both dict-style and attribute-style access."""
    status = desc.get("status") if isinstance(desc, dict) else getattr(desc, "status", None)
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


class DenseRetriever:
    """Pinecone-backed dense retriever.

    Owns the index lifecycle (ensure_index, upsert) and the query path.
    `store` is only required for `search` — ingestion can construct this
    without a Store and call `ensure_index`/`upsert` alone.
    """

    def __init__(self, store: Store | None = None) -> None:
        self._settings = get_settings()
        self._store = store
        self._pc: Any = None
        self._index: Any = None

    # ---- lazy client ----------------------------------------------------- #

    def _client(self) -> Any:  # noqa: ANN401 — pinecone.Pinecone is untyped
        if self._pc is None:
            from pinecone import Pinecone

            self._pc = Pinecone(api_key=self._settings.pinecone_api_key.get_secret_value())
        return self._pc

    def _get_index(self) -> Any:  # noqa: ANN401 — pinecone Index handle is untyped
        if self._index is None:
            self._index = self._client().Index(self._settings.pinecone_index_name)
        return self._index

    # ---- index lifecycle ------------------------------------------------- #

    def ensure_index(self) -> None:
        """Create the serverless index if it doesn't exist and wait until ready."""
        from pinecone import ServerlessSpec

        name = self._settings.pinecone_index_name
        client = self._client()
        if name not in client.list_indexes().names():
            client.create_index(
                name=name,
                dimension=self._settings.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_region,
                ),
            )

        deadline = time.monotonic() + _INDEX_READY_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if _index_ready(client.describe_index(name)):
                return
            time.sleep(_INDEX_READY_POLL_SEC)
        raise TimeoutError(f"Pinecone index {name!r} did not become ready in time")

    def upsert(self, embedded: list[EmbeddedChunk]) -> int:
        """Upsert EmbeddedChunks into Pinecone in batches of 100."""
        if not embedded:
            return 0
        index = self._get_index()
        n = 0
        for i in range(0, len(embedded), _UPSERT_BATCH):
            batch = embedded[i : i + _UPSERT_BATCH]
            vectors = [
                {
                    "id": ec.chunk.chunk_id,
                    "values": ec.embedding,
                    "metadata": {
                        "pmid": ec.chunk.pmid,
                        "section": ec.chunk.section or "",
                        "chunk_index": ec.chunk.chunk_index,
                    },
                }
                for ec in batch
            ]
            index.upsert(vectors=vectors)
            n += len(batch)
        return n

    # ---- query path ------------------------------------------------------ #

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._store is None:
            raise RuntimeError(
                "DenseRetriever needs a Store to hydrate results on search; "
                "construct with DenseRetriever(store=...)"
            )
        vectors = embed_texts([query])
        if not vectors:
            return []

        response = self._get_index().query(
            vector=vectors[0],
            top_k=top_k,
            include_values=False,
            include_metadata=False,
        )
        matches: list[Any] = list(
            response.get("matches", []) if isinstance(response, dict) else response.matches
        )
        if not matches:
            return []

        chunk_ids = [str(m["id"]) for m in matches]
        chunks = self._store.get_chunks(chunk_ids)
        pmids = {c.pmid for c in chunks.values()}
        papers = self._store.get_papers(list(pmids))

        results: list[RetrievedChunk] = []
        for m in matches:
            cid = str(m["id"])
            chunk = chunks.get(cid)
            if chunk is None:
                continue
            paper = papers.get(chunk.pmid)
            if paper is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    paper=paper,
                    score=float(m.get("score", 0.0)),
                    rank=len(results) + 1,
                    method=RetrievalMethod.DENSE,
                )
            )
        return results
