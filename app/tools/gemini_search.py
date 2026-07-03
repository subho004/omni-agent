"""Gemini grounded-search agent tool (docs/implementation-plan.md Phase 8).

Fast factual discovery via Gemini's built-in Google Search grounding.
Returns a synthesised answer plus the source URLs it was grounded on.
Prefer this over web_search when you want a direct answer rather than a
list of links to open.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.llm is None:
        return {"error": "Grounded search unavailable (no LLM client on context)"}

    query = str(args["query"])
    logger.info("gemini_search: %s", query)
    answer, sources, in_tok, out_tok = await ctx.llm.grounded_search(query)
    if ctx.ledger is not None:
        await ctx.ledger.create(ctx.session_id, ctx.llm.model, in_tok, out_tok)

    return {"answer": answer, "sources": sources}


gemini_search_tool = Tool(
    name="gemini_search",
    description=(
        "Ask a question and get a direct, up-to-date answer grounded in "
        "Google Search, with source URLs. Use for fast factual lookups when "
        "you want a synthesised answer rather than a list of links to open "
        "and read yourself (that's web_search)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question to answer with grounded search.",
            },
        },
        "required": ["query"],
    },
    handler=_handle,
)
