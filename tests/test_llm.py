"""Tests for the OpenRouter client wrapper, with the SDK client faked out."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
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
        return SimpleNamespace(
            choices=[choice], usage=usage, model="fake/model", provider="FakeProvider"
        )


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


def test_ignored_providers_are_sent_to_openrouter(fake_client):
    c = fake_client()
    llm.complete("sys", "user")
    assert c.kwargs["extra_body"]["provider"] == {"ignore": ["Novita"]}  # type: ignore[index]


def test_no_provider_field_when_nothing_is_ignored(fake_client, monkeypatch):
    monkeypatch.setattr(llm.get_settings(), "llm_ignore_providers", " , ")
    c = fake_client()
    llm.complete("sys", "user")
    assert "provider" not in c.kwargs["extra_body"]  # type: ignore[operator]


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


class SequenceCompletions(FakeCompletions):
    """Returns the given (content, finish, reasoning_tokens) replies in order."""

    def __init__(self, replies: list[tuple[str, str, int]]) -> None:
        super().__init__(content="", finish="stop", reasoning_tokens=0, cost=0.0)
        self.replies = replies
        self.calls = 0

    def create(self, **kwargs: object) -> SimpleNamespace:
        self._content, self._finish, self._reasoning = self.replies[self.calls]
        self.calls += 1
        return super().create(**kwargs)


@pytest.fixture
def sequence(monkeypatch: pytest.MonkeyPatch):
    def _install(*replies: tuple[str, str, int]) -> SequenceCompletions:
        completions = SequenceCompletions(list(replies))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(llm, "_get_client", lambda: client)
        return completions

    return _install


def test_provider_is_recorded_and_logged_on_truncation(fake_client, caplog):
    fake_client(content="partial", finish="length", reasoning_tokens=0)
    with caplog.at_level(logging.WARNING, logger="medrag.agent.llm"):
        r = llm.complete("sys", "user")
    assert r.provider == "FakeProvider"
    assert "provider=FakeProvider" in caplog.text


def test_empty_truncated_reply_with_reasoning_off_is_retried_once(sequence, caplog):
    c = sequence(("", "length", 1024), ("answer", "stop", 0))
    with llm.track_usage() as usage, caplog.at_level(logging.WARNING, logger="medrag.agent.llm"):
        r = llm.complete("sys", "user")
    assert r.text == "answer"
    assert c.calls == 2
    # The failed attempt was billed, so it stays in the usage bucket.
    assert [u.truncated for u in usage] == [True, False]
    assert "retrying once" in caplog.text


def test_empty_reply_is_retried_only_once(sequence):
    c = sequence(("", "length", 1024), ("", "length", 1024), ("never", "stop", 0))
    r = llm.complete("sys", "user")
    assert r.text == ""
    assert r.truncated is True
    assert c.calls == 2


@pytest.mark.parametrize(
    ("first", "kwargs"),
    [
        (("partial text", "length", 0), {}),  # visible text: a real cap hit, keep it
        (("", "stop", 0), {}),  # not truncated
        (("", "length", 1024), {"reasoning": True}),  # reasoning on: cap too small
    ],
)
def test_no_retry_outside_the_empty_reasoning_off_case(sequence, first, kwargs):
    c = sequence(first, ("second", "stop", 0))
    llm.complete("sys", "user", **kwargs)
    assert c.calls == 1


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


class FlakyCompletions(FakeCompletions):
    """Times out `failures` times, then answers."""

    def __init__(self, failures: int) -> None:
        super().__init__(content="ok", finish="stop", reasoning_tokens=0, cost=0.0)
        self.failures = failures
        self.timeouts: list[object] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        from openai import APITimeoutError

        self.timeouts.append(kwargs["timeout"])
        if len(self.timeouts) <= self.failures:
            raise APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
        return super().create(**kwargs)


@pytest.fixture
def flaky(monkeypatch: pytest.MonkeyPatch):
    from tenacity import wait_none

    monkeypatch.setattr(llm, "_retry_wait", wait_none())

    def _install(failures: int) -> FlakyCompletions:
        completions = FlakyCompletions(failures)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(llm, "_get_client", lambda: client)
        return completions

    return _install


def test_default_limits_are_patient(flaky):
    c = flaky(failures=5)
    assert llm.complete("sys", "user").text == "ok"
    settings = llm.get_settings()
    assert len(c.timeouts) == settings.llm_max_attempts == 6
    assert set(c.timeouts) == {settings.llm_timeout_s}


def test_online_limits_cut_timeout_and_attempts(flaky):
    from openai import APITimeoutError

    c = flaky(failures=5)
    settings = llm.get_settings()
    with llm.online_limits(), pytest.raises(APITimeoutError):
        llm.complete("sys", "user")
    assert len(c.timeouts) == settings.llm_online_max_attempts == 2
    assert set(c.timeouts) == {settings.llm_online_timeout_s}


def test_online_limits_end_with_the_block(flaky):
    c = flaky(failures=0)
    with llm.online_limits():
        llm.complete("sys", "user")
    llm.complete("sys", "user")
    assert c.timeouts == [llm.get_settings().llm_online_timeout_s, llm.get_settings().llm_timeout_s]


def test_explicit_timeout_wins_over_online_limits(flaky):
    c = flaky(failures=0)
    with llm.online_limits():
        llm.complete("sys", "user", timeout_s=300.0)
    assert c.timeouts == [300.0]


def test_retries_are_logged_not_printed(flaky, caplog, capsys):
    flaky(failures=1)
    with caplog.at_level(logging.WARNING, logger="medrag.agent.llm"):
        llm.complete("sys", "user")
    assert "APITimeoutError, retrying" in caplog.text
    assert "attempt 1 of 6" in caplog.text
    assert capsys.readouterr().out == ""
