"""Vision tool: answer a question about an image artifact (Phase 10).

Uses the multimodal Gemini model to inspect an uploaded/downloaded image
and answer a specific question. Token usage is recorded in the session
ledger when one is available on the context.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext, image_mime_for

logger = get_logger(__name__)


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.llm is None:
        return {"error": "Vision is unavailable (no LLM client on context)"}

    artifact_id = UUID(str(args["artifact_id"]))
    question = str(args["question"])
    artifact = await ctx.artifacts.find_by_id(artifact_id)
    if artifact is None or artifact.session_id != ctx.session_id:
        return {"error": f"Artifact '{artifact_id}' not found in this session"}

    mime = image_mime_for(artifact.name)
    if mime is None:
        return {"error": f"Artifact '{artifact.name}' is not a supported image"}

    logger.info("analyze_image: %s (%s)", artifact.name, artifact_id)
    image_bytes = await asyncio.to_thread(
        lambda: open(artifact.uri, "rb").read()
    )
    answer, in_tok, out_tok = await ctx.llm.describe_image(
        question, image_bytes, mime
    )
    if ctx.ledger is not None:
        await ctx.ledger.create(ctx.session_id, ctx.llm.model, in_tok, out_tok)

    return {"answer": answer, "image": artifact.name}


analyze_image_tool = Tool(
    name="analyze_image",
    description=(
        "Ask a question about an uploaded or downloaded image (PNG, JPEG, "
        "GIF, WEBP, BMP) using vision. Pass the image's artifact_id and a "
        "specific question; returns the model's answer about the image."
    ),
    parameters={
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "Artifact id of the image to inspect.",
            },
            "question": {
                "type": "string",
                "description": "What to ask about the image.",
            },
        },
        "required": ["artifact_id", "question"],
    },
    handler=_handle,
)
