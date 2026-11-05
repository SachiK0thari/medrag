"""Tests for the OpenRouter client wrapper, with the SDK client faked out."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from medrag.agent import llm


class FakeCompletions:
    def __init__(self, *, content: str, finish: str, reasoning_tokens: int | None, cost: object):
        self.kwargs: dict[str, object] = {}
        self._content = content
        self._finish = finish
        self._reasoning = reasoning_tokens
        self._cost = cost

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        details = (
            SimpleNamespace(reasoning_tokens=self._reasoning)
            if self._reasoning is not None
            else None
        )
        usage = SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            cost=self._cost,
            completion_tokens_details=details,
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content=self._content), finish_reason=self._finish
        )
        return SimpleNamespace(choices=[choice], usage=usage, model="fake/model")


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch):
    def _install(
        content: str = "hello",
        finish: str = "stop",
        reasoning_tokens: int | None = 0,
        cost: object = 0.001,
    ) -> FakeCompletions:
        completions = FakeCompletions(
            content=content, finish=finish, reasoning_tokens=reasoning_tokens, cost=cost
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(llm, "_get_client", lambda: client)
        return completions

    return _install


def test_complete_maps_response_fields(fake_client):
    fake_client(content="hi", reasoning_tokens=3, cost=0.5)
    r = llm.complete("sys", "user")
    assert r.text == "hi"
    assert (r.input_tokens, r.output_tokens) == (11, 7)
    assert r.model == "fake/model"
    assert r.cost_usd == 0.5
    assert r.finish_reason == "stop"
    assert r.reasoning_tokens == 3
    assert r.truncated is False


def test_reasoning_flag_is_always_sent_and_defaults_off(fake_client):
    """Omitting the flag leaves the provider default (ON for some models)."""
    c = fake_client()
    llm.complete("sys", "user")
    assert c.kwargs["extra_body"]["reasoning"] == {"enabled": False}  # type: ignore[index]
    assert c.kwargs["extra_body"]["usage"] == {"include": True}  # type: ignore[index]


def test_reasoning_can_be_overridden_per_call(fake_client):
    c = fake_client()
    llm.complete("sys", "user", reasoning=True)
    assert c.kwargs["extra_body"]["reasoning"] == {"enabled": True}  # type: ignore[index]


def test_max_tokens_and_temperature_overrides(fake_client):
    c = fake_client()
    llm.complete("sys", "user", max_tokens=256, temperature=0.0)
    assert c.kwargs["max_tokens"] == 256
    assert c.kwargs["temperature"] == 0.0
    assert c.kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


def test_truncation_is_flagged_and_logged(fake_client, caplog: pytest.LogCaptureFixture):
    fake_client(content="", finish="length", reasoning_tokens=384)
    with caplog.at_level(logging.WARNING, logger="medrag.agent.llm"):
        r = llm.complete("sys", "user", max_tokens=384)
    assert r.truncated is True
    assert r.text == ""
    assert r.reasoning_tokens == 384
    assert "truncated at max_tokens=384" in caplog.text


def test_missing_cost_and_details_are_tolerated(fake_client):
    fake_client(cost=None, reasoning_tokens=None)
    r = llm.complete("sys", "user")
    assert r.cost_usd is None
    assert r.reasoning_tokens == 0


def test_none_content_becomes_empty_string(fake_client):
    fake_client(content=None)  # type: ignore[arg-type]
    assert llm.complete("sys", "user").text == ""


def test_track_usage_collects_nested_calls(fake_client):
    fake_client()
    with llm.track_usage() as outer:
        llm.complete("s", "u")
        with llm.track_usage() as inner:
            llm.complete("s", "u")
        llm.complete("s", "u")
    assert len(inner) == 1
    assert len(outer) == 3
    # Buckets are popped on exit: a later call is not recorded anywhere.
    llm.complete("s", "u")
    assert len(outer) == 3


def test_request_timeout_defaults_to_setting_and_can_be_overridden(fake_client):
    c = fake_client()
    llm.complete("sys", "user")
    assert c.kwargs["timeout"] == llm.get_settings().llm_timeout_s
    llm.complete("sys", "user", timeout_s=300.0)
    assert c.kwargs["timeout"] == 300.0
