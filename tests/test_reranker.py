"""Fixed-shape batch planning for the reranker (pure functions, no model load)."""

from __future__ import annotations

from medrag.retrieval.reranker import _BASE_BATCH, _BASE_LENGTH, _bucket_length, _plan_batches

_MAX = 8192


def test_bucket_length_rounds_up_to_a_doubling_ladder() -> None:
    assert _bucket_length(1, _MAX) == _BASE_LENGTH
    assert _bucket_length(_BASE_LENGTH, _MAX) == _BASE_LENGTH
    assert _bucket_length(_BASE_LENGTH + 1, _MAX) == 2 * _BASE_LENGTH
    assert _bucket_length(1070, _MAX) == 4 * _BASE_LENGTH


def test_bucket_length_is_capped_at_the_model_maximum() -> None:
    assert _bucket_length(50_000, _MAX) == _MAX
    # A cap that is not on the ladder still wins.
    assert _bucket_length(700, 600) == 600


def test_plan_covers_every_index_exactly_once() -> None:
    lengths = [300, 520, 90, 1070, 511, 512, 513, 2000]
    plan = _plan_batches(lengths, _MAX)
    seen = [i for _, _, indices in plan for i in indices]
    assert sorted(seen) == list(range(len(lengths)))


def test_plan_uses_a_bounded_set_of_shapes() -> None:
    # 500 pairs with arbitrary lengths must still map to a handful of shapes:
    # that bound is what keeps MPS driver memory from growing.
    lengths = [17 + (i * 37) % 1500 for i in range(500)]
    shapes = {(padded, batch_size) for padded, batch_size, _ in _plan_batches(lengths, _MAX)}
    assert shapes <= {
        (_BASE_LENGTH, _BASE_BATCH),
        (2 * _BASE_LENGTH, _BASE_BATCH // 2),
        (4 * _BASE_LENGTH, _BASE_BATCH // 4),
    }


def test_plan_batches_never_exceed_their_batch_size() -> None:
    plan = _plan_batches([100] * (2 * _BASE_BATCH + 3), _MAX)
    assert [len(indices) for _, _, indices in plan] == [_BASE_BATCH, _BASE_BATCH, 3]
    assert all(batch_size == _BASE_BATCH for _, batch_size, _ in plan)


def test_plan_shrinks_batch_size_as_length_grows() -> None:
    plan = _plan_batches([100, 1000, 8000], _MAX)
    by_length = {padded: batch_size for padded, batch_size, _ in plan}
    assert by_length[_BASE_LENGTH] == _BASE_BATCH
    assert by_length[2 * _BASE_LENGTH] == _BASE_BATCH // 2
    assert by_length[_MAX] == 1


def test_plan_of_nothing_is_empty() -> None:
    assert _plan_batches([], _MAX) == []
