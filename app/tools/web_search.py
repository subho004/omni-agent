"""Web search tool backed by ddgs (DuckDuckGo metasearch)."""

from __future__ import annotations

import asyncio
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

# Upper bound on results per call; the model may request fewer.
MAX_RESULTS_CAP = 25
DEFAULT_RESULTS = 10
# Retry transient backend failures (rate limits, timeouts) before giving up.
MAX_ATTEMPTS = 3


def _is_no_results(exc: BaseException) -> bool:
    """ddgs raises a generic DDGSException for a genuinely empty result set."""

    return isinstance(exc, DDGSException) and "no results" in str(exc).lower()


def _is_transient(exc: BaseException) -> bool:
    """Backend errors worth retrying (rate limits, timeouts) — but not 'no results'."""

    return isinstance(exc, DDGSException) and not _is_no_results(exc)


def _search_sync(query: str, max_results: int, region: str) -> list[dict[str, str]]:
    with DDGS() as ddgs:
        rows = ddgs.text(query, max_results=max_results, region=region)
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in rows
    ]


def _search_with_retry(
    query: str, max_results: int, region: str
) -> list[dict[str, str]]:
    """Search with exponential backoff on transient backend failures.

    A genuinely empty result set (or exhausted retries) returns ``[]`` rather
    than raising, so the agent can reformulate the query or try another tool
    instead of registering a hard tool failure.
    """

    try:
        for attempt in Retrying(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=1, max=8),
            reraise=True,
        ):
            with attempt:
                return _search_sync(query, max_results, region)
    except DDGSException as exc:
        logger.info("web_search: no usable results for %s (%s)", query, exc)
    return []


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args["query"])
    max_results = min(int(args.get("max_results", DEFAULT_RESULTS)), MAX_RESULTS_CAP)
    region = str(args.get("region") or settings.search_region)
    logger.info("web_search: %s [region=%s]", query, region)
    results = await asyncio.to_thread(_search_with_retry, query, max_results, region)
    return {"results": results, "count": len(results)}


web_search_tool = Tool(
    name="web_search",
    description=(
        "Search the web and return result titles, URLs and snippets. Use this "
        "to DISCOVER pages or documents. To read a page's full content, pass "
        "its URL to download_file and then parse_document."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (default 10, max 25).",
            },
            "region": {
                "type": "string",
                "description": (
                    "Optional ddgs region code to localize results, e.g. "
                    "'wt-wt' (worldwide, default), 'in-en' (India), 'uk-en', "
                    "'us-en'. Set this to match the query's country/locale when "
                    "the topic is region-specific (e.g. an Indian company or "
                    "regulator); otherwise omit for unbiased worldwide results."
                ),
            },
        },
        "required": ["query"],
    },
    handler=_handle,
)
