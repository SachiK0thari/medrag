"""Cross-encoder reranking with bge-reranker-v2-m3.

Takes the top candidates from hybrid retrieval and re-scores each (query, chunk)
pair with a cross-encoder. Substantially slower than bi-encoder retrieval, which
is why it runs only on a small top-N from the fusion stage.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from medrag.config import get_settings
from medrag.models import RetrievalMethod, RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _load_model() -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    return CrossEncoder(settings.reranker_model)


class Reranker:
    """Cross-encoder reranker (BAAI/bge-reranker-v2-m3 by default)."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = _load_model()
        pairs = [(query, rc.chunk.text) for rc in chunks]
        scores = model.predict(pairs, show_progress_bar=False)
        scored = sorted(
            zip(chunks, scores, strict=True),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )
        return [
            RetrievedChunk(
                chunk=rc.chunk,
                paper=rc.paper,
                score=float(score),
                rank=i + 1,
                method=RetrievalMethod.HYBRID_RERANKED,
            )
            for i, (rc, score) in enumerate(scored[:top_k])
        ]
