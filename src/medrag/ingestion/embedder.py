"""Dense embeddings via sentence-transformers.

BAAI/bge-small-en-v1.5 by default — 384-dim, CPU-friendly, strong on MTEB.
Lazy-loads the model on first use so imports stay cheap.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from medrag.config import get_settings
from medrag.models import Chunk, EmbeddedChunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns unit-normalized float vectors."""
    if not texts:
        return []
    model = _load_model()
    settings = get_settings()
    vectors = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Attach embeddings to a batch of chunks."""
    vectors = embed_texts([c.text for c in chunks])
    return [EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(chunks, vectors, strict=True)]
