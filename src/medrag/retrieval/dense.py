"""Dense retrieval against Pinecone."""

from __future__ import annotations

from medrag.models import RetrievedChunk

# TODO(saturday): wrap pinecone.Pinecone client.
# - ensure_index(): create serverless index if missing (dim from settings)
# - upsert(embedded_chunks): batch into groups of 100
# - search(query_vector, top_k): query() then hydrate Chunk + Paper from SQLite


class DenseRetriever:
    def __init__(self) -> None:
        pass

    def ensure_index(self) -> None:
        raise NotImplementedError("implement Saturday afternoon")

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError("implement Saturday evening")
