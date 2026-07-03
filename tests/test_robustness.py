"""Tests for tool caching, circuit breaking, timeouts, and verification."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from app.db.database import async_session_factory
from app.repositories.agent_session_repository import AgentSessionRepository
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.ledger_repository import LedgerRepository
from app.repositories.message_repository import MessageRepository
from app.services.orchestrator import _looks_like_failure
from app.tools import tool_cache, tool_guard
from app.tools.base import Tool, ToolContext


# ---- circuit breaker -------------------------------------------------------
def test_circuit_breaker_trips_after_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.tools import tool_guard as tg

    monkeypatch.setattr(tg.settings, "circuit_trip_threshold", 2)
    sid = uuid4()
    assert not tg.is_tripped(sid, "browser_use")
    tg.record_failure(sid, "browser_use")
    assert not tg.is_tripped(sid, "browser_use")
    tg.record_failure(sid, "browser_use")
    assert tg.is_tripped(sid, "browser_use")
    tg.record_success(sid, "browser_use")  # a success resets it
    assert not tg.is_tripped(sid, "browser_use")
    tg.reset(sid)


# ---- cache -----------------------------------------------------------------
def test_tool_cache_roundtrip() -> None:
    sid = uuid4()
    assert tool_cache.get(sid, "web_search", {"q": "x"}) is None
    tool_cache.put(sid, "web_search", {"q": "x"}, {"results": [1]})
    assert tool_cache.get(sid, "web_search", {"q": "x"}) == {"results": [1]}
    # Different args miss.
    assert tool_cache.get(sid, "web_search", {"q": "y"}) is None
    tool_cache.reset(sid)
    assert tool_cache.get(sid, "web_search", {"q": "x"}) is None


# ---- refusal detector ------------------------------------------------------
def test_looks_like_failure() -> None:
    assert _looks_like_failure("") is True
    assert _looks_like_failure("n/a") is True
    assert _looks_like_failure("I was unable to access the site.") is True
    assert _looks_like_failure("The capital is Tokyo, population 14 million.") is False


# ---- executor integration: timeout + cache via run_agent_loop --------------
@pytest_asyncio.fixture
async def loop_ctx(
    client: object, tmp_path: Path
) -> AsyncIterator[tuple[ToolContext, MessageRepository, LedgerRepository]]:
    async with async_session_factory() as db:
        session = await AgentSessionRepository(db).create("robust")
        ctx = ToolContext(
            session_id=session.id,
            artifacts=ArtifactRepository(db),
            data_dir=tmp_path,
        )
        yield ctx, MessageRepository(db), LedgerRepository(db)
        tool_cache.reset(session.id)
        tool_guard.reset(session.id)


@pytest.mark.asyncio
async def test_execute_tool_times_out(
    loop_ctx: tuple[ToolContext, MessageRepository, LedgerRepository],
) -> None:
    ctx, messages, _ = loop_ctx

    async def slow(_ctx: ToolContext, _args: dict) -> dict:
        await asyncio.sleep(5)
        return {"ok": True}

    tool = Tool("slow", "d", {"type": "object", "properties": {}}, slow, timeout=0.2)
    from google.genai import types

    monkey = {"slow": tool}
    from app.services import agent_loop as al

    al.TOOLS_BY_NAME.update(monkey)
    try:
        parts: list[types.Part] = []
        trace = await al._execute_tool(
            ctx, messages, types.FunctionCall(name="slow", args={}), parts
        )
    finally:
        al.TOOLS_BY_NAME.pop("slow", None)
    assert "timed out" in trace.result_summary


@pytest.mark.asyncio
async def test_execute_tool_uses_cache(
    loop_ctx: tuple[ToolContext, MessageRepository, LedgerRepository],
) -> None:
    ctx, messages, _ = loop_ctx
    calls = {"n": 0}

    async def counter(_ctx: ToolContext, _args: dict) -> dict:
        calls["n"] += 1
        return {"value": 42}

    from google.genai import types

    from app.services import agent_loop as al

    tool = Tool(
        "web_search", "d", {"type": "object", "properties": {}}, counter
    )
    original = al.TOOLS_BY_NAME.get("web_search")
    al.TOOLS_BY_NAME["web_search"] = tool
    try:
        for _ in range(3):
            await al._execute_tool(
                ctx,
                messages,
                types.FunctionCall(name="web_search", args={"q": "same"}),
                [],
            )
    finally:
        if original is not None:
            al.TOOLS_BY_NAME["web_search"] = original
    assert calls["n"] == 1  # ran once, served from cache twice
