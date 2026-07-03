"""Direct handler tests for offline tools (python_exec, bm25_search).

These exercise the tool handlers without the LLM or network by building a
ToolContext against the test database session.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from app.db.database import async_session_factory
from app.repositories.agent_session_repository import AgentSessionRepository
from app.repositories.artifact_repository import ArtifactRepository
from app.tools.base import ToolContext
from app.tools.bash_exec import bash_exec_tool
from app.tools.bm25_search import bm25_search_tool
from app.tools.python_exec import python_exec_tool


@pytest_asyncio.fixture
async def tool_ctx(client: object, tmp_path: Path) -> AsyncIterator[ToolContext]:
    """A ToolContext backed by a real (test) session and a temp data dir.

    Depends on the `client` fixture so the schema is created/dropped.
    """

    async with async_session_factory() as db:
        session = await AgentSessionRepository(db).create("tool test")
        yield ToolContext(
            session_id=session.id,
            artifacts=ArtifactRepository(db),
            data_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_python_exec_runs_and_captures_stdout(tool_ctx: ToolContext) -> None:
    result = await python_exec_tool.handler(
        tool_ctx, {"code": "print(6 * 7)"}
    )
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "42"
    assert result["files_created"] == []


@pytest.mark.asyncio
async def test_python_exec_reports_created_files(tool_ctx: ToolContext) -> None:
    result = await python_exec_tool.handler(
        tool_ctx,
        {"code": "open('out.txt', 'w').write('hi')"},
    )
    assert result["exit_code"] == 0
    assert "out.txt" in result["files_created"]


@pytest.mark.asyncio
async def test_python_exec_timeout(tool_ctx: ToolContext) -> None:
    result = await python_exec_tool.handler(
        tool_ctx, {"code": "import time; time.sleep(5)", "timeout": 1}
    )
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_bash_exec_runs_and_captures_stdout(tool_ctx: ToolContext) -> None:
    result = await bash_exec_tool.handler(
        tool_ctx, {"command": "echo $((6 * 7))"}
    )
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "42"


@pytest.mark.asyncio
async def test_bash_exec_reports_created_files(tool_ctx: ToolContext) -> None:
    result = await bash_exec_tool.handler(
        tool_ctx, {"command": "printf hi > out.txt"}
    )
    assert result["exit_code"] == 0
    assert "out.txt" in result["files_created"]


@pytest.mark.asyncio
async def test_bash_exec_nonzero_exit_and_stderr(tool_ctx: ToolContext) -> None:
    result = await bash_exec_tool.handler(
        tool_ctx, {"command": "echo oops >&2; exit 3"}
    )
    assert result["exit_code"] == 3
    assert "oops" in result["stderr"]


@pytest.mark.asyncio
async def test_bash_exec_timeout(tool_ctx: ToolContext) -> None:
    result = await bash_exec_tool.handler(
        tool_ctx, {"command": "sleep 5", "timeout": 1}
    )
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_bm25_search_finds_relevant_passage(tool_ctx: ToolContext) -> None:
    # Seed a parsed markdown artifact on disk + in the repo.
    doc = (
        "The mitochondria is the powerhouse of the cell.\n\n"
        + ("Filler sentence about unrelated topics. " * 40)
        + "\n\nQuantum entanglement links two particles across distance."
    )
    md_path = tool_ctx.data_dir / "doc.md"
    md_path.write_text(doc, encoding="utf-8")
    artifact = await tool_ctx.artifacts.create(
        session_id=tool_ctx.session_id,
        kind="parsed",
        name="doc.md",
        uri=str(md_path),
    )

    result = await bm25_search_tool.handler(
        tool_ctx, {"artifact_id": str(artifact.id), "query": "quantum entanglement"}
    )
    assert result["count"] >= 1
    assert "entanglement" in result["matches"][0]["passage"].lower()


@pytest.mark.asyncio
async def test_bm25_search_rejects_non_parsed_artifact(
    tool_ctx: ToolContext,
) -> None:
    artifact = await tool_ctx.artifacts.create(
        session_id=tool_ctx.session_id,
        kind="upload",
        name="raw.bin",
        uri="/tmp/raw.bin",
    )
    result = await bm25_search_tool.handler(
        tool_ctx, {"artifact_id": str(artifact.id), "query": "anything"}
    )
    assert "error" in result
