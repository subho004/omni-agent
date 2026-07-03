"""Reasoning-based document navigation tool (PageIndex, Phase 4 primary).

Builds a section outline of a parsed document and lets the model reason over
its structure to select the relevant sections, then returns their full text.
Unlike bm25_search (lexical similarity), this retrieves what is *logically*
relevant by navigating the document's structure.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.page_index import build_sections, render_outline
from app.schemas.retrieval import SectionSelection
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

MAX_SECTIONS = 5

_NAV_SYSTEM = (
    "You navigate a document by its structure. Given a question and an outline "
    "of numbered sections (with heading paths and previews), return the ids of "
    "the sections most likely to contain the answer. Prefer a few precise "
    "sections over many."
)


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.llm is None:
        return {"error": "doc_navigate unavailable (no LLM client on context)"}

    artifact_id = UUID(str(args["artifact_id"]))
    query = str(args["query"])
    artifact = await ctx.artifacts.find_by_id(artifact_id)
    if artifact is None or artifact.session_id != ctx.session_id:
        return {"error": f"Artifact '{artifact_id}' not found in this session"}
    if artifact.kind != "parsed":
        return {"error": "doc_navigate only works on parsed (markdown) artifacts"}

    logger.info("doc_navigate: %r in %s", query, artifact_id)
    text = await asyncio.to_thread(
        lambda: open(artifact.uri, encoding="utf-8").read()
    )
    sections = build_sections(text)
    outline = render_outline(sections)

    selection, in_tok, out_tok = await ctx.llm.generate_structured(
        prompt=f"Question:\n{query}\n\nDocument outline:\n{outline}",
        system_instruction=_NAV_SYSTEM,
        response_schema=SectionSelection,
    )
    if ctx.ledger is not None:
        await ctx.ledger.create(ctx.session_id, ctx.llm.model, in_tok, out_tok)

    chosen_ids = (
        selection.section_ids
        if isinstance(selection, SectionSelection)
        else []
    )
    valid = [i for i in chosen_ids if 0 <= i < len(sections)][:MAX_SECTIONS]
    reason = selection.reason if isinstance(selection, SectionSelection) else ""

    results = [
        {
            "section_id": i,
            "breadcrumb": sections[i].breadcrumb,
            "text": sections[i].text[: settings.doc_section_chars],
        }
        for i in valid
    ]
    return {
        "sections": results,
        "reason": reason,
        "total_sections": len(sections),
    }


doc_navigate_tool = Tool(
    name="doc_navigate",
    description=(
        "Find the answer to a question inside a long parsed document by "
        "reasoning over its section structure (headings). Returns the full "
        "text of the most relevant sections. Prefer this over bm25_search "
        "when the document is well-structured and the question needs the right "
        "section rather than keyword matches."
    ),
    parameters={
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "The parsed_artifact_id to navigate.",
            },
            "query": {
                "type": "string",
                "description": "The question to locate in the document.",
            },
        },
        "required": ["artifact_id", "query"],
    },
    handler=_handle,
)
