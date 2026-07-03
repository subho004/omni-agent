"""Tests for the vision path: auto-describe on upload + analyze_image tool.

A fake LLM with a `describe_image` method stands in for Gemini vision, so
no network is used. A tiny valid PNG is uploaded through the service.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from app.db.database import async_session_factory
from app.repositories.agent_session_repository import AgentSessionRepository
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.ledger_repository import LedgerRepository
from app.repositories.message_repository import MessageRepository
from app.services.session_service import SessionService
from app.tools.analyze_image import analyze_image_tool
from app.tools.base import ToolContext

# Smallest valid 1x1 PNG.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class FakeVisionLlm:
    model = "fake-vision"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def describe_image(
        self, prompt: str, image_bytes: bytes, mime_type: str
    ) -> tuple[str, int, int]:
        self.calls.append(prompt)
        assert mime_type == "image/png"
        assert image_bytes == _PNG_BYTES
        return ("A single black pixel.", 11, 4)


@pytest_asyncio.fixture
async def vision_setup(
    client: object, tmp_path: Path
) -> AsyncIterator[tuple[FakeVisionLlm, "ToolContext", object]]:
    fake = FakeVisionLlm()
    async with async_session_factory() as db:
        session = await AgentSessionRepository(db).create("vision")
        service = SessionService(
            sessions=AgentSessionRepository(db),
            messages=MessageRepository(db),
            artifacts=ArtifactRepository(db),
            llm=fake,  # type: ignore[arg-type]
            ledger=LedgerRepository(db),
        )
        ctx = ToolContext(
            session_id=session.id,
            artifacts=ArtifactRepository(db),
            data_dir=tmp_path,
            llm=fake,  # type: ignore[arg-type]
            ledger=LedgerRepository(db),
        )
        yield fake, ctx, service


@pytest.mark.asyncio
async def test_image_upload_is_auto_described(
    vision_setup: tuple[FakeVisionLlm, ToolContext, SessionService],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake, ctx, service = vision_setup
    # Route uploads into the test's temp data dir.
    from app.services import session_service as ss

    monkeypatch.setattr(ss.settings, "data_dir", str(tmp_path))

    artifact = await service.save_upload(ctx.session_id, "pixel.png", _PNG_BYTES)
    assert artifact.summary == "A single black pixel."
    assert fake.calls  # vision was invoked


@pytest.mark.asyncio
async def test_analyze_image_tool(
    vision_setup: tuple[FakeVisionLlm, ToolContext, SessionService],
) -> None:
    fake, ctx, _ = vision_setup
    img_path = ctx.data_dir / "img.png"
    img_path.write_bytes(_PNG_BYTES)
    artifact = await ctx.artifacts.create(
        session_id=ctx.session_id,
        kind="upload",
        name="img.png",
        uri=str(img_path),
    )

    result = await analyze_image_tool.handler(
        ctx, {"artifact_id": str(artifact.id), "question": "What is shown?"}
    )
    assert result["answer"] == "A single black pixel."
    assert result["image"] == "img.png"


@pytest.mark.asyncio
async def test_analyze_image_rejects_non_image(
    vision_setup: tuple[FakeVisionLlm, ToolContext, SessionService],
) -> None:
    fake, ctx, _ = vision_setup
    artifact = await ctx.artifacts.create(
        session_id=ctx.session_id,
        kind="upload",
        name="notes.txt",
        uri="/tmp/notes.txt",
    )
    result = await analyze_image_tool.handler(
        ctx, {"artifact_id": str(artifact.id), "question": "?"}
    )
    assert "error" in result
