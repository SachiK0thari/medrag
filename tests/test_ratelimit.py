"""Sliding-window rate limiter, driven by a fake clock."""

from __future__ import annotations

import pytest

from medrag import ratelimit
from medrag.ratelimit import DAY_S, Decision, RateLimiter, client_from_forwarded


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make(clock: FakeClock, *, per_client: int = 3, window_s: float = 60, global_per_day: int = 0):
    return RateLimiter(
        per_client=per_client, window_s=window_s, global_per_day=global_per_day, clock=clock
    )


def test_allows_up_to_the_client_limit_then_refuses(clock: FakeClock):
    limiter = make(clock)
    assert all(limiter.acquire("a").allowed for _ in range(3))
    refused = limiter.acquire("a")
    assert not refused.allowed
    assert refused.scope == "client"
    assert refused.retry_after_s == pytest.approx(60)


def test_clients_are_counted_separately(clock: FakeClock):
    limiter = make(clock, per_client=1)
    assert limiter.acquire("a").allowed
    assert not limiter.acquire("a").allowed
    assert limiter.acquire("b").allowed


def test_window_slides(clock: FakeClock):
    limiter = make(clock, per_client=2)
    assert limiter.acquire("a").allowed  # t=0
    clock.advance(40)
    assert limiter.acquire("a").allowed  # t=40
    clock.advance(10)
    refused = limiter.acquire("a")  # t=50: the first hit frees up at t=60
    assert not refused.allowed
    assert refused.retry_after_s == pytest.approx(10)
    clock.advance(10)
    assert limiter.acquire("a").allowed  # t=60
    assert not limiter.acquire("a").allowed  # hits at t=40 and t=60 remain


def test_refused_requests_are_not_counted(clock: FakeClock):
    limiter = make(clock, per_client=1)
    assert limiter.acquire("a").allowed
    for _ in range(10):
        clock.advance(5)
        assert not limiter.acquire("a").allowed
    clock.advance(10)  # t=60: only the one allowed hit ever counted
    assert limiter.acquire("a").allowed


def test_global_cap_spans_clients_and_lasts_a_day(clock: FakeClock):
    limiter = make(clock, per_client=0, global_per_day=2)
    assert limiter.acquire("a").allowed
    clock.advance(3600)
    assert limiter.acquire("b").allowed
    refused = limiter.acquire("c")
    assert not refused.allowed
    assert refused.scope == "global"
    assert refused.retry_after_s == pytest.approx(DAY_S - 3600)
    clock.advance(DAY_S - 3600)
    assert limiter.acquire("c").allowed


def test_a_client_refusal_does_not_spend_global_budget(clock: FakeClock):
    limiter = make(clock, per_client=1, global_per_day=2)
    assert limiter.acquire("a").allowed
    assert not limiter.acquire("a").allowed
    assert limiter.acquire("b").allowed  # second and last global slot
    assert limiter.acquire("c").scope == "global"


def test_zero_turns_a_window_off(clock: FakeClock):
    limiter = make(clock, per_client=0, global_per_day=0)
    assert all(limiter.acquire("a").allowed for _ in range(1000))


def test_idle_clients_are_swept(clock: FakeClock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ratelimit, "_SWEEP_THRESHOLD", 5)
    limiter = make(clock)
    for i in range(5):
        limiter.acquire(f"old-{i}")
    clock.advance(61)
    limiter.acquire("new")
    assert set(limiter._clients) == {"new"}


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (Decision(True), ""),
        (Decision(False, "client", 42.2), "Try again in 43 seconds."),
        (Decision(False, "client", 1500), "Try again in 25 minutes."),
        (Decision(False, "global", 7 * 3600), "Try again in 7 hours."),
    ],
)
def test_message(decision: Decision, expected: str):
    assert decision.message.endswith(expected)


def test_global_message_names_the_daily_budget():
    assert "daily" in Decision(False, "global", 60).message


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("203.0.113.7", "203.0.113.7"),
        ("203.0.113.7, 10.0.0.1, 10.0.0.2", "203.0.113.7"),
        ("  203.0.113.7 ,10.0.0.1", "203.0.113.7"),
        ("", "local"),
        (None, "local"),
    ],
)
def test_client_from_forwarded(header: str | None, expected: str):
    assert client_from_forwarded(header) == expected


def test_get_limiter_reads_settings_and_is_shared(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RATE_LIMIT_PER_CLIENT", "7")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "120")
    monkeypatch.setenv("RATE_LIMIT_GLOBAL_PER_DAY", "50")
    ratelimit.get_settings.cache_clear()
    ratelimit.get_limiter.cache_clear()
    try:
        limiter = ratelimit.get_limiter()
        assert (limiter.per_client, limiter.window_s, limiter.global_per_day) == (7, 120, 50)
        assert ratelimit.get_limiter() is limiter
    finally:
        ratelimit.get_settings.cache_clear()
        ratelimit.get_limiter.cache_clear()
