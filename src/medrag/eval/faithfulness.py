"""RAGAS faithfulness scorer for generated answers.

Uses the RAGAS library's ``Faithfulness`` metric — "what fraction of atomic
claims in the answer are supported by the retrieved contexts?" — instead of
a hand-rolled prompt.  The LLM calls go through OpenRouter's OpenAI-compatible
endpoint via ``langchain-openai``, which RAGAS already pulls in as a
transitive dependency.
"""

from __future__ import annotations

import asyncio
import logging
import math

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.exceptions import LLMDidNotFinishException
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

from medrag.config import get_settings
from medrag.models import RetrievedChunk

_log = logging.getLogger(__name__)

_llm: LangchainLLMWrapper | None = None


def _get_llm() -> LangchainLLMWrapper:
    """Lazily build a RAGAS-compatible LLM wrapper over OpenRouter."""
    global _llm
    if _llm is None:
        settings = get_settings()
        chat = ChatOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=0.0,
            # RAGAS asks for a JSON list of every claim in the answer; leave
            # room for it, and keep hidden reasoning from eating the budget
            # (see the note in medrag.agent.llm.complete).
            max_tokens=4096,
            extra_body={
                "reasoning": {"enabled": False},
                **({"provider": routing} if (routing := settings.llm_provider_routing) else {}),
            },
        )
        _llm = LangchainLLMWrapper(chat)
    return _llm


def compute_faithfulness(
    question: str,
    answer_text: str,
    contexts: list[RetrievedChunk],
) -> float | None:
    """Return RAGAS faithfulness score in [0, 1], or None on failure.

    Returns None (not 0.0) when contexts/answer are empty or scoring fails,
    so the harness can distinguish "0% supported" from "scorer failed" in the CSV.
    Failures are logged at WARNING with the exception type so a blank column
    in the CSV is never a mystery.
    """
    if not contexts or not answer_text.strip():
        _log.warning("faithfulness skipped: empty contexts or answer")
        return None

    sample = SingleTurnSample(
        user_input=question,
        response=answer_text,
        retrieved_contexts=[rc.chunk.text.strip() for rc in contexts],
    )

    scorer = Faithfulness()
    scorer.llm = _get_llm()

    # RAGAS goes through its own client, so the empty-reply retry in
    # medrag.agent.llm.complete does not cover it. The failure is the same:
    # DeepSeek V4.1 Flash now and then reasons despite the flag, spends the
    # token cap and RAGAS raises LLMDidNotFinishException. It is intermittent,
    # so score once more before giving up; any other error is not retried.
    score: object = None
    for attempt in (1, 2):
        try:
            score = asyncio.run(scorer.single_turn_ascore(sample))
            break
        except LLMDidNotFinishException as e:
            if attempt == 1:
                _log.warning("faithfulness scorer hit the token cap; retrying once")
                continue
            _log.warning("faithfulness scorer failed: %s: %s", type(e).__name__, e)
            return None
        except Exception as e:
            _log.warning("faithfulness scorer failed: %s: %s", type(e).__name__, e)
            return None

    if score is None or not isinstance(score, int | float) or math.isnan(score):
        _log.warning("faithfulness scorer returned non-numeric result: %r", score)
        return None

    return float(score)
