"""Direct handler tests for the LLM-backed agent tools (gemini_search).

Uses a fake LLM client so no network/browser is exercised. The browser_use
tool is not unit-tested here (it drives a real browser); it is covered by a
manual live smoke test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from app.db.database import async_session_factory
from app.repositories.agent_session_repository import AgentSessionRepository
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.ledger_repository import LedgerRepository
from app.tools.base import ToolContext
from app.tools.gemini_search import gemini_search_tool


class FakeGroundedLlm:
    model = "fake"

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def grounded_search(
        self, prompt: str
    ) -> tuple[str, list[str], int, int]:
        self.queries.append(prompt)
        return ("Canberra is the capital.", ["https://example.com/au"], 9, 5)


@pytest_asyncio.fixture
async def grounded_ctx(
    client: object, tmp_path: Path
) -> AsyncIterator[ToolContext]:
    async with async_session_factory() as db:
        session = await AgentSessionRepository(db).create("grounded")
        yield ToolContext(
            session_id=session.id,
            artifacts=ArtifactRepository(db),
            data_dir=tmp_path,
            llm=FakeGroundedLlm(),  # type: ignore[arg-type]
            ledger=LedgerRepository(db),
        )


@pytest.mark.asyncio
async def test_gemini_search_returns_answer_and_sources(
    grounded_ctx: ToolContext,
) -> None:
    result = await gemini_search_tool.handler(
        grounded_ctx, {"query": "capital of Australia"}
    )
    assert result["answer"] == "Canberra is the capital."
    assert result["sources"] == ["https://example.com/au"]


@pytest.mark.asyncio
async def test_gemini_search_without_llm_errors(
    client: object, tmp_path: Path
) -> None:
    async with async_session_factory() as db:
        session = await AgentSessionRepository(db).create("no-llm")
        ctx = ToolContext(
            session_id=session.id,
            artifacts=ArtifactRepository(db),
            data_dir=tmp_path,
        )
    result = await gemini_search_tool.handler(ctx, {"query": "x"})
    assert "error" in result
