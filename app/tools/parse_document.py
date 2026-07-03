"""Document parsing tool backed by MarkItDown.

Converts a stored artifact (pdf/docx/xlsx/pptx/html/…) to markdown, saves
the markdown as a new artifact, and returns a bounded excerpt — the full
text never enters the model context directly.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from markitdown import MarkItDown

from app.core.config import settings
from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

_markitdown = MarkItDown()


def _convert_sync(path: str) -> str:
    return _markitdown.convert(path).text_content


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    artifact_id = UUID(str(args["artifact_id"]))
    artifact = await ctx.artifacts.find_by_id(artifact_id)
    if artifact is None or artifact.session_id != ctx.session_id:
        return {"error": f"Artifact '{artifact_id}' not found in this session"}

    logger.info("parse_document: %s (%s)", artifact.name, artifact_id)
    try:
        markdown = await asyncio.to_thread(_convert_sync, artifact.uri)
    except Exception as exc:
        logger.exception("MarkItDown failed for %s", artifact.uri)
        return {"error": f"Failed to parse document: {exc}"}

    parsed_dir = ctx.data_dir / "parsed" / str(ctx.session_id)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    md_path = parsed_dir / f"{artifact_id}.md"
    md_path.write_text(markdown, encoding="utf-8")

    parsed = await ctx.artifacts.create(
        session_id=ctx.session_id,
        kind="parsed",
        name=f"{artifact.name}.md",
        uri=str(md_path),
        summary=markdown[:500],
    )
    return {
        "parsed_artifact_id": str(parsed.id),
        "total_chars": len(markdown),
        "excerpt": markdown[:settings.tool_excerpt_chars],
        "truncated": len(markdown) > settings.tool_excerpt_chars,
    }


async def _handle_read(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    artifact_id = UUID(str(args["artifact_id"]))
    offset = max(int(args.get("offset", 0)), 0)
    artifact = await ctx.artifacts.find_by_id(artifact_id)
    if artifact is None or artifact.session_id != ctx.session_id:
        return {"error": f"Artifact '{artifact_id}' not found in this session"}
    if artifact.kind != "parsed":
        return {"error": "read_artifact only works on parsed (markdown) artifacts"}

    text = await asyncio.to_thread(
        lambda: open(artifact.uri, encoding="utf-8").read()
    )
    chunk = text[offset : offset + settings.tool_excerpt_chars]
    return {
        "content": chunk,
        "offset": offset,
        "total_chars": len(text),
        "has_more": offset + settings.tool_excerpt_chars < len(text),
    }


parse_document_tool = Tool(
    name="parse_document",
    description=(
        "Convert a stored artifact (PDF, DOCX, XLSX, PPTX, HTML, …) to "
        "markdown. Takes the artifact_id returned by download_file or a file "
        "upload. Returns a parsed_artifact_id plus the first part of the "
        "text; use read_artifact with an offset to read more."
    ),
    parameters={
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "Artifact id of the file to parse.",
            },
        },
        "required": ["artifact_id"],
    },
    handler=_handle,
)

read_artifact_tool = Tool(
    name="read_artifact",
    description=(
        "Read a chunk of an already-parsed markdown artifact starting at a "
        "character offset. Use after parse_document when the document is "
        "longer than the returned excerpt."
    ),
    parameters={
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "The parsed_artifact_id to read.",
            },
            "offset": {
                "type": "integer",
                "description": "Character offset to start reading from (default 0).",
            },
        },
        "required": ["artifact_id"],
    },
    handler=_handle_read,
)
