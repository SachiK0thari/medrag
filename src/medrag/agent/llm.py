"""Thin wrapper around the Groq SDK with token + cost tracking.

Model-agnostic by design: swap GROQ_MODEL or replace this file with an
Anthropic / OpenAI client and the rest of the codebase doesn't change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from medrag.config import get_settings

if TYPE_CHECKING:
    from groq import Groq


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=get_settings().groq_api_key.get_secret_value())
    return _client


def complete(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Single chat completion. Synchronous on purpose — Groq is fast enough."""
    settings = get_settings()
    client = _get_client()
    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
    )
    usage = resp.usage
    return LLMResponse(
        text=resp.choices[0].message.content or "",
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        model=settings.groq_model,
    )
