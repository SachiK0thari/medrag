"""In-process rate limiter for the public entry points (Streamlit UI, FastAPI).

Two sliding windows, both counted in questions:

* per client (default 10 an hour), so one visitor cannot use up the demo;
* global over 24 hours (default 200), which is what bounds LLM spend. The
  client key comes from a proxy header a caller can forge, so the per-client
  window is a fairness limit and the global one is the budget.

State lives in memory: it is shared by every session of one process and is
lost on restart. That fits a single-container demo; several workers would
need a shared store. The eval harness calls the agent directly and is never
limited.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from medrag.config import get_settings

DAY_S = 24 * 60 * 60

# Sweep idle clients once the table grows past this many keys.
_SWEEP_THRESHOLD = 1024


@dataclass(frozen=True)
class Decision:
    allowed: bool
    # Which window refused the request; None when allowed.
    scope: Literal["client", "global"] | None = None
    # Seconds until the refusing window has room again; 0 when allowed.
    retry_after_s: float = 0.0

    @property
    def message(self) -> str:
        """User-facing explanation of a refusal (empty when allowed)."""
        if self.allowed:
            return ""
        wait = _humanize(self.retry_after_s)
        if self.scope == "global":
            return f"The demo has reached its daily question budget. Try again in {wait}."
        return f"You have reached the question limit for this demo. Try again in {wait}."


def _humanize(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, math.ceil(seconds))} seconds"
    minutes = math.ceil(seconds / 60)
    if minutes < 90:
        return f"{minutes} minutes"
    return f"{math.ceil(minutes / 60)} hours"


class RateLimiter:
    """Sliding-window limiter: `per_client` per `window_s`, `global_per_day` per 24 h.

    A limit of 0 turns that window off. A refused request is not counted, so
    retrying while limited does not push the retry time back.
    """

    def __init__(
        self,
        *,
        per_client: int,
        window_s: float,
        global_per_day: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.per_client = per_client
        self.window_s = window_s
        self.global_per_day = global_per_day
        self._clock = clock
        self._lock = threading.Lock()
        self._clients: dict[str, deque[float]] = {}
        self._global: deque[float] = deque()

    def acquire(self, client: str) -> Decision:
        """Count one request for `client` if both windows have room."""
        with self._lock:
            now = self._clock()
            _expire(self._global, now - DAY_S)
            hits = self._clients.get(client)
            if hits is not None:
                _expire(hits, now - self.window_s)

            if self.global_per_day > 0 and len(self._global) >= self.global_per_day:
                return Decision(False, "global", self._global[0] + DAY_S - now)
            if self.per_client > 0 and hits is not None and len(hits) >= self.per_client:
                return Decision(False, "client", hits[0] + self.window_s - now)

            if self.global_per_day > 0:
                self._global.append(now)
            if self.per_client > 0:
                if hits is None:
                    if len(self._clients) >= _SWEEP_THRESHOLD:
                        self._sweep(now)
                    hits = self._clients.setdefault(client, deque())
                hits.append(now)
            return Decision(True)

    def _sweep(self, now: float) -> None:
        for key in list(self._clients):
            hits = self._clients[key]
            _expire(hits, now - self.window_s)
            if not hits:
                del self._clients[key]


def _expire(hits: deque[float], cutoff: float) -> None:
    while hits and hits[0] <= cutoff:
        hits.popleft()


def client_from_forwarded(forwarded_for: str | None, fallback: str = "local") -> str:
    """Client key from an ``X-Forwarded-For`` value: its first (original-client) entry."""
    first = (forwarded_for or "").split(",")[0].strip()
    return first or fallback


@lru_cache
def get_limiter() -> RateLimiter:
    """Process-wide limiter built from settings."""
    settings = get_settings()
    return RateLimiter(
        per_client=settings.rate_limit_per_client,
        window_s=settings.rate_limit_window_s,
        global_per_day=settings.rate_limit_global_per_day,
    )
