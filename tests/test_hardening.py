"""Tests for hardening: LLM retry/backoff and per-session token budget."""

from __future__ import annotations

import pytest
from google.genai import errors

from app.services import agent_loop as agent_loop_mod
from app.services.agent_loop import LoopResult, remaining_token_budget
from app.services.llm_client import LlmClient


class _FakeUsage:
    prompt_token_count = 3
    candidates_token_count = 2


class _FakeResponse:
    text = "ok"
    function_calls: list[object] = []
    candidates: list[object] = []
    parsed = None
    usage_metadata = _FakeUsage()


@pytest.mark.asyncio
async def test_generate_content_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LlmClient(api_key="x", model="fake")
    calls = {"n": 0}

    async def flaky(**kwargs: object) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] < 3:
            raise errors.ServerError(503, {"error": {"message": "unavailable"}})
        return _FakeResponse()

    # Patch the underlying SDK call and remove backoff waiting.
    monkeypatch.setattr(client._client.aio.models, "generate_content", flaky)
    monkeypatch.setattr(
        "app.services.llm_client.wait_exponential", lambda **_: (lambda *a, **k: 0)
    )

    result = await client.generate([], "sys")
    assert result.text == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on the third


@pytest.mark.asyncio
async def test_generate_content_gives_up_on_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LlmClient(api_key="x", model="fake")

    async def bad(**kwargs: object) -> _FakeResponse:
        raise errors.ClientError(400, {"error": {"message": "bad request"}})

    monkeypatch.setattr(client._client.aio.models, "generate_content", bad)
    with pytest.raises(errors.ClientError):
        await client.generate([], "sys")


def test_format_history_uses_configured_per_turn_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from app.services import agent_loop

    monkeypatch.setattr(agent_loop.settings, "history_turn_chars", 3000)
    monkeypatch.setattr(agent_loop.settings, "history_max_turns", 12)
    long_answer = "x" * 2500  # would have been truncated at the old 600 cap
    messages = [
        SimpleNamespace(role="user", content="q"),
        SimpleNamespace(role="assistant", content=long_answer),
    ]
    digest = agent_loop.format_history(messages)
    assert long_answer in digest  # full 2500 chars retained, not clipped to 600


def test_remaining_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import agent_loop

    monkeypatch.setattr(agent_loop.settings, "session_token_budget", 0)
    assert remaining_token_budget(1000) is None  # 0 = unlimited

    monkeypatch.setattr(agent_loop.settings, "session_token_budget", 500)
    assert remaining_token_budget(200) == 300
    assert remaining_token_budget(600) == 0  # clamped, never negative


@pytest.mark.asyncio
async def test_agent_loop_stops_when_budget_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no remaining budget the loop returns immediately, no LLM call."""

    called = {"n": 0}

    class _Llm:
        model = "fake"

        async def generate(self, *a: object, **k: object) -> LoopResult:
            called["n"] += 1
            raise AssertionError("should not call the model")

    result = await agent_loop_mod.run_agent_loop(
        _Llm(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        [],
        "sys",
        token_budget=0,
    )
    assert result.hit_budget is True
    assert called["n"] == 0
