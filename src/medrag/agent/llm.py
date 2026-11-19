"""Thin wrapper around OpenRouter (OpenAI-compatible API) with token + cost tracking.

Model-agnostic by design: swap LLM_MODEL to any OpenRouter model id, or
replace this file with an Anthropic / OpenAI client and the rest of the
codebase doesn't change.

Two pieces of module state are process-global and unlocked: `_usage_stack`
(the `track_usage` buckets) and `_last_call_time` (the optional throttle).
Both assume one caller at a time, which holds for the eval and curation
scripts that use them. Concurrent callers (Streamlit sessions, API requests)
can make the throttle under-wait and would see each other's calls in an open
`track_usage` bucket; neither is used on those paths. The per-call limits from
`online_limits` live in a `ContextVar`, so they are safe across threads.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from medrag.config import get_settings

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion


_log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    # Actual USD cost as reported by OpenRouter (``usage.cost``); None when the
    # provider did not return it, in which case the harness falls back to its
    # static pricing table.
    cost_usd: float | None = None
    # "stop" on a normal completion; "length" when the output hit max_tokens.
    finish_reason: str | None = None
    # Hidden reasoning tokens billed as output. They count against max_tokens,
    # so a reasoning model with a small cap can return empty visible text.
    reasoning_tokens: int = 0

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


_client: OpenAI | None = None

# Backoff between retries; a module attribute so tests can make it instant.
_retry_wait: wait_base = wait_exponential(multiplier=2, min=2, max=60)


@dataclass(frozen=True)
class CallLimits:
    timeout_s: float
    max_attempts: int


_call_limits: ContextVar[CallLimits | None] = ContextVar("medrag_llm_call_limits", default=None)


@contextmanager
def online_limits() -> Iterator[None]:
    """Use the short timeout and few retries meant for a person waiting on an answer.

    The defaults (``LLM_TIMEOUT_S``, ``LLM_MAX_ATTEMPTS``) are patient because
    the eval and curation scripts would rather wait than lose a row. The
    public entry points wrap each answer in this block, so a stalled provider
    fails in under a minute per call instead of retrying for many minutes.
    """
    settings = get_settings()
    token = _call_limits.set(
        CallLimits(
            timeout_s=settings.llm_online_timeout_s,
            max_attempts=settings.llm_online_max_attempts,
        )
    )
    try:
        yield
    finally:
        _call_limits.reset(token)


# Active token-usage buckets. Each entry is a list appended to by every
# `complete()` call while the bucket is in scope. The harness uses this to
# attribute tokens to a single (config, question) pair without threading
# accumulators through every call site.
_usage_stack: list[list[LLMResponse]] = []


@contextmanager
def track_usage() -> Iterator[list[LLMResponse]]:
    """Collect every `LLMResponse` produced by `complete()` within this block."""
    bucket: list[LLMResponse] = []
    _usage_stack.append(bucket)
    try:
        yield bucket
    finally:
        _usage_stack.pop()


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        from openai import OpenAI

        settings = get_settings()
        _client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key.get_secret_value(),
            # Optional attribution headers; OpenRouter uses these for its
            # per-app leaderboard and nothing else.
            default_headers={"X-Title": "medrag"},
        )
    return _client


_last_call_time: float = 0.0


def complete(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: bool | None = None,
    timeout_s: float | None = None,
) -> LLMResponse:
    """Single chat completion with optional throttle + retry on rate limits.

    `reasoning` overrides ``LLM_REASONING_ENABLED`` for this call (the gold
    curation judge turns it on; the latency-sensitive agent leaves it off).
    `timeout_s` overrides ``LLM_TIMEOUT_S`` (and the `online_limits` timeout);
    a timed-out request is retried. Without an explicit cap a stalled
    connection waits out the SDK's 10-minute default, which once froze a batch
    job on a single request. Attempts come from ``LLM_MAX_ATTEMPTS``, or from
    `online_limits` inside that block.
    """
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    settings = get_settings()
    client = _get_client()
    limits = _call_limits.get()
    max_attempts = limits.max_attempts if limits else settings.llm_max_attempts
    if timeout_s is None:
        timeout_s = limits.timeout_s if limits else settings.llm_timeout_s

    # OpenRouter-specific knobs go in extra_body (the OpenAI SDK passes them
    # through verbatim). ``usage.include`` makes the response carry the real
    # USD cost; ``reasoning`` toggles the model's thinking mode.
    #
    # The reasoning flag is ALWAYS sent. Omitting it does not mean "off": it
    # leaves the provider default, and DeepSeek V4.1 Flash reasons by default.
    # Hidden reasoning tokens count against max_tokens, so with the flag
    # omitted a tightly capped call could spend its whole budget thinking and
    # return an empty string (this silently broke gold curation once).
    use_reasoning = settings.llm_reasoning_enabled if reasoning is None else reasoning
    effective_max_tokens = max_tokens or settings.llm_max_tokens
    extra_body: dict[str, Any] = {
        "usage": {"include": True},
        "reasoning": {"enabled": use_reasoning},
    }

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        stop=stop_after_attempt(max_attempts),
        wait=_retry_wait,
        before_sleep=lambda rs: _log.warning(
            "LLM call failed with %s, retrying in %.0fs (attempt %d of %d)",
            type(rs.outcome.exception()).__name__ if rs.outcome else "error",
            rs.next_action.sleep if rs.next_action else 0,
            rs.attempt_number,
            max_attempts,
        ),
        reraise=True,
    )
    def _call() -> ChatCompletion:
        global _last_call_time
        # Optional proactive throttle (LLM_THROTTLE_S). Off by default; useful
        # when running against a provider with a tight requests-per-minute cap.
        if settings.llm_throttle_s > 0:
            elapsed = time.time() - _last_call_time
            if elapsed < settings.llm_throttle_s:
                time.sleep(settings.llm_throttle_s - elapsed)
        _last_call_time = time.time()
        return client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=effective_max_tokens,
            extra_body=extra_body,
            timeout=timeout_s,
        )

    resp = _call()
    usage = resp.usage
    choice = resp.choices[0]
    # OpenRouter adds a non-standard ``cost`` field when usage accounting is on.
    raw_cost = getattr(usage, "cost", None) if usage else None
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    raw_reasoning = getattr(details, "reasoning_tokens", None) if details else None
    response = LLMResponse(
        text=choice.message.content or "",
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        model=resp.model or settings.llm_model,
        cost_usd=float(raw_cost) if isinstance(raw_cost, int | float) else None,
        finish_reason=choice.finish_reason,
        reasoning_tokens=raw_reasoning if isinstance(raw_reasoning, int) else 0,
    )
    if response.truncated:
        _log.warning(
            "LLM output truncated at max_tokens=%d (reasoning_tokens=%d, visible_chars=%d)",
            effective_max_tokens,
            response.reasoning_tokens,
            len(response.text),
        )
    for bucket in _usage_stack:
        bucket.append(response)
    return response
