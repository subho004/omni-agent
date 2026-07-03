"""Download tool: fetch a URL into the session's artifact store."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
DOWNLOAD_TIMEOUT_S = 60.0


def _filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    return name or "download.bin"


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    url = str(args["url"])
    logger.info("download_file: %s", url)

    async with httpx.AsyncClient(
        follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_S
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        if len(response.content) > MAX_DOWNLOAD_BYTES:
            return {"error": f"File exceeds {MAX_DOWNLOAD_BYTES} byte limit"}

    downloads_dir = ctx.data_dir / "downloads" / str(ctx.session_id)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    file_path = downloads_dir / _filename_from_url(url)
    file_path.write_bytes(response.content)

    artifact = await ctx.artifacts.create(
        session_id=ctx.session_id,
        kind="download",
        name=file_path.name,
        uri=str(file_path),
        summary=f"Downloaded from {url}",
    )
    return {
        "artifact_id": str(artifact.id),
        "name": file_path.name,
        "size_bytes": len(response.content),
        "content_type": response.headers.get("content-type", "unknown"),
    }


download_file_tool = Tool(
    name="download_file",
    description=(
        "Download a file or web page from a URL and store it as a session "
        "artifact. Returns an artifact_id. Follow up with parse_document on "
        "the artifact_id to read its content as markdown."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to download."},
        },
        "required": ["url"],
    },
    handler=_handle,
)
