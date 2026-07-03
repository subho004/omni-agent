"""Tests for PageIndex: section-tree builder + doc_navigate tool."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from app.db.database import async_session_factory
from app.repositories.agent_session_repository import AgentSessionRepository
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.ledger_repository import LedgerRepository
from app.retrieval.page_index import build_sections, render_outline
from app.schemas.retrieval import SectionSelection
from app.tools.base import ToolContext
from app.tools.doc_navigate import doc_navigate_tool

_DOC = """# Guide

Intro text.

## Installation

Run pip install foo to set it up.

## Usage

Call foo.run() to start.

### Advanced

Pass foo.run(fast=True) for speed.
"""


def test_build_sections_tracks_breadcrumbs() -> None:
    sections = build_sections(_DOC)
    titles = [s.title for s in sections]
    assert "Installation" in titles
    assert "Usage" in titles
    advanced = next(s for s in sections if s.title == "Advanced")
    assert advanced.breadcrumb == "Guide > Usage > Advanced"
    assert advanced.level == 3


def test_build_sections_no_headings_single_section() -> None:
    sections = build_sections("just a paragraph, no headings")
    assert len(sections) == 1
    assert "paragraph" in sections[0].text


def test_render_outline_has_ids() -> None:
    outline = render_outline(build_sections(_DOC))
    assert outline.startswith("[0]")
    assert "Installation" in outline


class FakeNavLlm:
    model = "fake-nav"

    def __init__(self, pick: list[int]) -> None:
        self.pick = pick
        self.outline_seen = ""

    async def generate_structured(
        self, prompt: str, system_instruction: str, response_schema: type
    ) -> tuple[object, int, int]:
        self.outline_seen = prompt
        return (
            SectionSelection(section_ids=self.pick, reason="matches"),
            10,
            3,
        )


@pytest_asyncio.fixture
async def nav_ctx(
    client: object, tmp_path: Path
) -> AsyncIterator[tuple[ToolContext, FakeNavLlm]]:
    fake = FakeNavLlm(pick=[])
    async with async_session_factory() as db:
        session = await AgentSessionRepository(db).create("nav")
        ctx = ToolContext(
            session_id=session.id,
            artifacts=ArtifactRepository(db),
            data_dir=tmp_path,
            llm=fake,  # type: ignore[arg-type]
            ledger=LedgerRepository(db),
        )
        yield ctx, fake


@pytest.mark.asyncio
async def test_doc_navigate_returns_selected_section(
    nav_ctx: tuple[ToolContext, FakeNavLlm],
) -> None:
    ctx, fake = nav_ctx
    sections = build_sections(_DOC)
    usage_id = next(s.index for s in sections if s.title == "Usage")
    fake.pick = [usage_id]

    md_path = ctx.data_dir / "guide.md"
    md_path.write_text(_DOC, encoding="utf-8")
    artifact = await ctx.artifacts.create(
        session_id=ctx.session_id,
        kind="parsed",
        name="guide.md",
        uri=str(md_path),
    )

    result = await doc_navigate_tool.handler(
        ctx, {"artifact_id": str(artifact.id), "query": "How do I start?"}
    )
    assert len(result["sections"]) == 1
    assert result["sections"][0]["breadcrumb"] == "Guide > Usage"
    assert "foo.run()" in result["sections"][0]["text"]


@pytest.mark.asyncio
async def test_doc_navigate_rejects_non_parsed(
    nav_ctx: tuple[ToolContext, FakeNavLlm],
) -> None:
    ctx, _ = nav_ctx
    artifact = await ctx.artifacts.create(
        session_id=ctx.session_id,
        kind="upload",
        name="raw.bin",
        uri="/tmp/raw.bin",
    )
    result = await doc_navigate_tool.handler(
        ctx, {"artifact_id": str(artifact.id), "query": "x"}
    )
    assert "error" in result
