"""RAGAS faithfulness scorer for generated answers.

Uses the RAGAS library's ``Faithfulness`` metric — "what fraction of atomic
claims in the answer are supported by the retrieved contexts?" — instead of
a hand-rolled prompt.  The LLM calls go through OpenRouter's OpenAI-compatible
endpoint via ``langchain-openai``, which RAGAS already pulls in as a
transitive dependency.
"""

from __future__ import annotations

import asyncio
import math

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

from medrag.config import get_settings
from medrag.models import RetrievedChunk

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
            max_tokens=1024,
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
    """
    if not contexts or not answer_text.strip():
        return None

    sample = SingleTurnSample(
        user_input=question,
        response=answer_text,
        retrieved_contexts=[rc.chunk.text.strip() for rc in contexts],
    )

    scorer = Faithfulness()
    scorer.llm = _get_llm()

    try:
        score = asyncio.run(scorer.single_turn_ascore(sample))
    except Exception:
        return None

    if score is None or not isinstance(score, int | float) or math.isnan(score):
        return None

    return float(score)
