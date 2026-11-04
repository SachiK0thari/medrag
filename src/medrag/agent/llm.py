"""Thin wrapper around OpenRouter (OpenAI-compatible API) with token + cost tracking.

Model-agnostic by design: swap LLM_MODEL to any OpenRouter model id, or
replace this file with an Anthropic / OpenAI client and the rest of the
codebase doesn't change.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from medrag.config import get_settings

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion


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


_client: OpenAI | None = None

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
) -> LLMResponse:
    """Single chat completion with optional throttle + retry on rate limits."""
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    settings = get_settings()
    client = _get_client()

    # OpenRouter-specific knobs go in extra_body (the OpenAI SDK passes them
    # through verbatim). ``usage.include`` makes the response carry the real
    # USD cost; ``reasoning`` toggles the model's thinking mode.
    extra_body: dict[str, Any] = {"usage": {"include": True}}
    if settings.llm_reasoning_enabled:
        extra_body["reasoning"] = {"enabled": True}

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        before_sleep=lambda rs: print(
            f"  [llm-retry] {type(rs.outcome.exception()).__name__ if rs.outcome else 'error'}"
            f" — waiting {rs.next_action.sleep if rs.next_action else 0:.0f}s"
            f" (attempt {rs.attempt_number})..."
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
            max_tokens=max_tokens or settings.llm_max_tokens,
            extra_body=extra_body,
        )

    resp = _call()
    usage = resp.usage
    # OpenRouter adds a non-standard ``cost`` field when usage accounting is on.
    raw_cost = getattr(usage, "cost", None) if usage else None
    response = LLMResponse(
        text=resp.choices[0].message.content or "",
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        model=resp.model or settings.llm_model,
        cost_usd=float(raw_cost) if isinstance(raw_cost, int | float) else None,
    )
    for bucket in _usage_stack:
        bucket.append(response)
    return response
