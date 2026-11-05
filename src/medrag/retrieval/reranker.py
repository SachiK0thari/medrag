"""Cross-encoder reranking with bge-reranker-v2-m3.

Takes the top candidates from hybrid retrieval and re-scores each (query, chunk)
pair with a cross-encoder. Substantially slower than bi-encoder retrieval, which
is why it runs only on a small top-N from the fusion stage.

Apple Silicon note: on the MPS backend PyTorch keeps driver memory for every
distinct input shape it has seen, and that memory is never returned (not even
by ``torch.mps.empty_cache()``). The stock ``CrossEncoder.predict`` pads each
batch to its longest pair, so nearly every call is a new shape; over a long
eval run the process grew past 17 GB and stalled the machine in swap. On MPS
we therefore score in fixed-shape batches (`_plan_batches`): pairs are padded
to a small ladder of lengths and every batch is filled to a constant size, so
the set of shapes — and the memory — stays bounded. Scores are unchanged up to
float noise because padding is masked out.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from medrag.config import get_settings
from medrag.models import RetrievalMethod, RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

# Fixed-shape scoring (MPS only). Pair lengths are rounded up to
# _BASE_LENGTH * 2**n, and the batch size shrinks as the length grows so each
# batch holds the same number of tokens.
_BASE_LENGTH = 512
_BASE_BATCH = 16


def _bucket_length(n_tokens: int, max_length: int) -> int:
    """Padded length for a pair of `n_tokens`: 512, 1024, 2048, ... capped at `max_length`."""
    size = _BASE_LENGTH
    while size < n_tokens and size < max_length:
        size *= 2
    return min(size, max_length)


def _plan_batches(lengths: list[int], max_length: int) -> list[tuple[int, int, list[int]]]:
    """Group pair indices into fixed-shape batches.

    Returns ``(padded_length, batch_size, indices)`` tuples. Every batch for a
    given padded length has the same `batch_size`; a batch with fewer indices
    than that is meant to be filled with dummy rows by the caller. Longest
    buckets come last so the common case runs first.
    """
    by_bucket: dict[int, list[int]] = {}
    for i, n in enumerate(lengths):
        by_bucket.setdefault(_bucket_length(n, max_length), []).append(i)

    plan: list[tuple[int, int, list[int]]] = []
    for padded in sorted(by_bucket):
        batch_size = max(1, _BASE_BATCH * _BASE_LENGTH // padded)
        indices = by_bucket[padded]
        for start in range(0, len(indices), batch_size):
            plan.append((padded, batch_size, indices[start : start + batch_size]))
    return plan


@lru_cache(maxsize=1)
def _load_model() -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    return CrossEncoder(settings.reranker_model)


def _score_fixed_shape(model: CrossEncoder, query: str, texts: list[str]) -> list[float]:
    """Score (query, text) pairs using only a bounded set of tensor shapes."""
    import torch

    tokenizer, network = model.tokenizer, model.model
    max_length = int(min(tokenizer.model_max_length, network.config.max_position_embeddings - 2))
    lengths = [
        len(ids)
        for ids in tokenizer(
            [query] * len(texts),
            texts,
            truncation="longest_first",
            max_length=max_length,
        )["input_ids"]
    ]

    scores = [0.0] * len(texts)
    for padded, batch_size, indices in _plan_batches(lengths, max_length):
        # Fill the batch with empty documents so its shape never varies.
        batch_texts = [texts[i] for i in indices] + [""] * (batch_size - len(indices))
        encoded = tokenizer(
            [query] * batch_size,
            batch_texts,
            padding="max_length",
            truncation="longest_first",
            max_length=padded,
            return_tensors="pt",
        ).to(network.device)
        with torch.inference_mode():
            logits = network(**encoded).logits.view(-1)
        # Same activation CrossEncoder.predict applies to a single-logit model.
        probs = torch.sigmoid(logits.float()).cpu().tolist()
        for i, p in zip(indices, probs, strict=False):
            scores[i] = float(p)
    return scores


class Reranker:
    """Cross-encoder reranker (BAAI/bge-reranker-v2-m3 by default)."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = _load_model()
        texts = [rc.chunk.text for rc in chunks]
        if model.model.device.type == "mps":
            scores = _score_fixed_shape(model, query, texts)
        else:
            scores = [
                float(s)
                for s in model.predict([(query, t) for t in texts], show_progress_bar=False)
            ]
        scored = sorted(
            zip(chunks, scores, strict=True),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return [
            RetrievedChunk(
                chunk=rc.chunk,
                paper=rc.paper,
                score=score,
                rank=i + 1,
                method=RetrievalMethod.HYBRID_RERANKED,
            )
            for i, (rc, score) in enumerate(scored[:top_k])
        ]
