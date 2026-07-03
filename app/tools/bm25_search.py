"""BM25 lexical retrieval over a parsed markdown artifact.

Phase 4 lexical fallback (docs/implementation-plan.md): splits a parsed
document into overlapping passages, ranks them against a query with
Okapi BM25, and returns the top passages. Lets the model pull the
relevant part of a long document into context without loading all of it.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from uuid import UUID

from rank_bm25 import BM25Plus

from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

PASSAGE_CHARS = 1_000
PASSAGE_OVERLAP = 200
MAX_TOP_K = 20
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _split_passages(text: str) -> list[str]:
    step = PASSAGE_CHARS - PASSAGE_OVERLAP
    return [
        text[start : start + PASSAGE_CHARS]
        for start in range(0, max(len(text), 1), step)
    ]


def _rank(text: str, query: str, top_k: int) -> list[dict[str, Any]]:
    passages = _split_passages(text)
    tokenized = [_tokenize(p) for p in passages]
    # BM25Plus (idf = log((N+1)/n)) keeps scores positive, avoiding the
    # negative-IDF pathology BM25Okapi hits when a query term appears in the
    # majority of passages of a small single-document corpus.
    bm25 = BM25Plus(tokenized)
    query_tokens = set(_tokenize(query))
    scores = bm25.get_scores(list(query_tokens))
    ranked = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
    # Only return passages that actually share a term with the query — every
    # passage gets a positive BM25Plus baseline, so score alone can't gate.
    return [
        {"passage": passages[i], "score": round(float(scores[i]), 3), "index": i}
        for i in ranked[:top_k]
        if query_tokens & set(tokenized[i])
    ]


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    artifact_id = UUID(str(args["artifact_id"]))
    query = str(args["query"])
    top_k = min(int(args.get("top_k", 5)), MAX_TOP_K)

    artifact = await ctx.artifacts.find_by_id(artifact_id)
    if artifact is None or artifact.session_id != ctx.session_id:
        return {"error": f"Artifact '{artifact_id}' not found in this session"}
    if artifact.kind != "parsed":
        return {"error": "bm25_search only works on parsed (markdown) artifacts"}

    logger.info("bm25_search: %r in %s", query, artifact_id)
    text = await asyncio.to_thread(
        lambda: open(artifact.uri, encoding="utf-8").read()
    )
    matches = await asyncio.to_thread(_rank, text, query, top_k)
    return {"matches": matches, "count": len(matches)}


bm25_search_tool = Tool(
    name="bm25_search",
    description=(
        "Keyword-search inside a parsed (markdown) document artifact and "
        "return the most relevant passages, ranked by BM25. Use this to find "
        "the specific part of a long document that answers a question instead "
        "of reading the whole thing."
    ),
    parameters={
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "The parsed_artifact_id to search within.",
            },
            "query": {"type": "string", "description": "Keywords to search for."},
            "top_k": {
                "type": "integer",
                "description": "Number of passages to return (default 5, max 20).",
            },
        },
        "required": ["artifact_id", "query"],
    },
    handler=_handle,
)
