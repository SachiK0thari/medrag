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

Hosted backend: on two shared CPUs (the free Hugging Face Space) the model
needs about 2.2 s per pair, and an agent answer scores about 72 pairs, so
reranking was 97% of a 167-second answer. With ``RERANKER_BACKEND=pinecone``
the pairs go to Pinecone's hosted copy of the same model instead (`_score_hosted`).
On two test questions its scores were within 0.004 of the local model's and
the top 10 came out in the same order. If the hosted call fails (the plan's
monthly quota, an outage) the pairs are scored locally, slowly, instead.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

from medrag.config import get_settings
from medrag.models import RetrievalMethod, RetrievedChunk

if TYPE_CHECKING:
    from pinecone import Pinecone
    from sentence_transformers import CrossEncoder

_log = logging.getLogger(__name__)

# Pinecone's hosted reranker takes at most this many documents per request.
_HOSTED_MAX_DOCS = 100

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
    import torch
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    model = CrossEncoder(settings.reranker_model)
    # On CPU the thread count against the cores actually granted decides the
    # speed: a container can report its host's cores while being limited to two.
    _log.info(
        "reranker %s on %s: %d torch threads, %s CPUs visible (%s usable)",
        settings.reranker_model,
        model.model.device.type,
        torch.get_num_threads(),
        os.cpu_count(),
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "unknown",
    )
    return model


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


def _score_local(query: str, texts: list[str]) -> list[float]:
    """Score (query, text) pairs with the cross-encoder loaded in this process."""
    model = _load_model()
    if model.model.device.type == "mps":
        return _score_fixed_shape(model, query, texts)
    return [float(s) for s in model.predict([(query, t) for t in texts], show_progress_bar=False)]


@lru_cache(maxsize=1)
def _pinecone_client() -> Pinecone:
    from pinecone import Pinecone

    return Pinecone(api_key=get_settings().pinecone_api_key.get_secret_value())


def _score_hosted(query: str, texts: list[str]) -> list[float]:
    """Score (query, text) pairs with Pinecone's hosted copy of the reranker.

    Pinecone names its hosted models without the Hugging Face organisation
    ("bge-reranker-v2-m3"). It accepts 1,024 tokens per pair; the chunker caps
    a chunk at 512 embedder tokens (the longest is 549 reranker tokens), so
    "truncate" is a guard and does not cut anything in this corpus.
    """
    model = get_settings().reranker_model.rsplit("/", 1)[-1]
    scores = [0.0] * len(texts)
    for start in range(0, len(texts), _HOSTED_MAX_DOCS):
        batch = texts[start : start + _HOSTED_MAX_DOCS]
        result = _pinecone_client().inference.rerank(
            model=model,
            query=query,
            documents=batch,
            top_n=len(batch),
            return_documents=False,
            parameters={"truncate": "END"},
        )
        for row in result.data:
            scores[start + row.index] = float(row.score)
    return scores


class Reranker:
    """Cross-encoder reranker (BAAI/bge-reranker-v2-m3 by default)."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def _score(self, query: str, texts: list[str]) -> list[float]:
        if self._settings.reranker_backend == "pinecone":
            try:
                return _score_hosted(query, texts)
            except Exception as exc:
                # Quota exhausted (429), outage, bad key: a slow answer beats none.
                _log.warning(
                    "hosted reranker failed (%s: %s); scoring %d pairs locally",
                    type(exc).__name__,
                    str(exc)[:200],
                    len(texts),
                )
        return _score_local(query, texts)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._score(query, [rc.chunk.text for rc in chunks])
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
