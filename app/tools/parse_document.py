"""Document parsing tool backed by MarkItDown.

Converts a stored artifact (pdf/docx/xlsx/pptx/html/…) to markdown, saves
the markdown as a new artifact, and returns a bounded excerpt — the full
text never enters the model context directly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

from markitdown import MarkItDown

from app.core.config import settings
from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

_markitdown = MarkItDown()

# Non-parsed artifacts read_artifact may return verbatim (text formats only —
# never binary like PDF/DOCX/images, which must go through parse_document). Lets
# an agent inspect raw crawl HTML (hrefs/JSON blobs/form targets the markdown
# dropped) or a downloaded JSON/CSV endpoint without markitdown re-mangling it.
_READABLE_TEXT_EXTENSIONS = {
    ".html", ".htm", ".txt", ".md", ".json", ".csv", ".tsv", ".xml",
    ".yaml", ".yml", ".log",
}


def _convert_sync(path: str) -> str:
    return _markitdown.convert(path).text_content


def convert_to_markdown(path: str) -> str:
    """Convert a document at ``path`` to markdown (blocking). Shared by
    parse_document and corpus_search's auto-parse path."""

    return _convert_sync(path)


async def ensure_parsed_markdown(
    ctx: ToolContext, artifact: Any
) -> tuple[Any, str]:
    """Parse a source artifact to markdown and persist it as a parsed artifact.

    Converts the document, writes the markdown next to the session's parsed
    files, and records a ``kind="parsed"`` artifact referencing it. Returns the
    parsed artifact and its markdown text. Used by both parse_document and
    corpus_search (which auto-parses any not-yet-parsed upload before indexing).
    """

    markdown = await asyncio.to_thread(convert_to_markdown, artifact.uri)
    parsed_dir = ctx.data_dir / "parsed" / str(ctx.session_id)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    md_path = parsed_dir / f"{artifact.id}.md"
    md_path.write_text(markdown, encoding="utf-8")

    parsed = await ctx.artifacts.create(
        session_id=ctx.session_id,
        kind="parsed",
        name=f"{artifact.name}.md",
        uri=str(md_path),
        summary=markdown[:500],
    )
    return parsed, markdown


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    artifact_id = UUID(str(args["artifact_id"]))
    artifact = await ctx.artifacts.find_by_id(artifact_id)
    if artifact is None or artifact.session_id != ctx.session_id:
        return {"error": f"Artifact '{artifact_id}' not found in this session"}

    logger.info("parse_document: %s (%s)", artifact.name, artifact_id)
    try:
        parsed, markdown = await ensure_parsed_markdown(ctx, artifact)
    except Exception as exc:
        logger.exception("MarkItDown failed for %s", artifact.uri)
        return {"error": f"Failed to parse document: {exc}"}

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
    if (
        artifact.kind != "parsed"
        and Path(artifact.uri).suffix.lower() not in _READABLE_TEXT_EXTENSIONS
    ):
        return {
            "error": (
                "read_artifact reads parsed markdown or text artifacts "
                "(html/json/csv/xml/txt/…). For binary files (PDF, DOCX) run "
                "parse_document first."
            )
        }

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
        "Read a chunk of a text artifact starting at a character offset. Works "
        "on parsed markdown (after parse_document, when the doc is longer than "
        "the excerpt) AND on raw text artifacts like the html_artifact_id from "
        "crawl_url or a downloaded JSON/CSV/XML file — use it to inspect the raw "
        "DOM or embedded data (hrefs, JSON blobs, form targets) the markdown "
        "dropped. Not for binary files (PDF, DOCX): parse_document those first."
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
