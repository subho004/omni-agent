"""Web-content tool backed by crawl4ai (docs/implementation-plan.md Phase 5).

Renders a URL in a headless browser and returns clean markdown — no LLM
extraction layer. Best for JavaScript-heavy or article-style pages where a
raw download would miss content. The full markdown is stored as an
artifact; only a bounded excerpt is returned to the model.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

CRAWL_TIMEOUT_MS = 135_000


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    url = str(args["url"])
    logger.info("crawl_url: %s", url)

    # Imported lazily: crawl4ai pulls in a heavy dependency tree and a
    # headless browser, which we don't want to load unless the tool is used.
    from crawl4ai import AsyncWebCrawler

    from app.tools.browser_session import new_crawl4ai_configs

    # Fresh, randomised, stealth-hardened browser session for every crawl.
    browser_config, run_config = new_crawl4ai_configs(
        headless=True, page_timeout_ms=CRAWL_TIMEOUT_MS
    )
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
    except Exception as exc:
        logger.exception("crawl_url failed for %s", url)
        return {"error": f"Crawl failed: {type(exc).__name__}: {exc}"}

    if not result.success:
        return {"error": f"Crawl failed: {result.error_message or 'unknown error'}"}

    markdown = str(result.markdown or "")
    crawled_dir = ctx.data_dir / "crawled" / str(ctx.session_id)
    crawled_dir.mkdir(parents=True, exist_ok=True)
    md_path = crawled_dir / f"{abs(hash(url))}.md"
    md_path.write_text(markdown, encoding="utf-8")

    artifact = await ctx.artifacts.create(
        session_id=ctx.session_id,
        kind="parsed",
        name=f"crawl-{md_path.stem}.md",
        uri=str(md_path),
        summary=f"Crawled {url}",
    )
    return {
        "parsed_artifact_id": str(artifact.id),
        "url": url,
        "total_chars": len(markdown),
        "excerpt": markdown[:settings.tool_excerpt_chars],
        "truncated": len(markdown) > settings.tool_excerpt_chars,
    }


crawl_url_tool = Tool(
    name="crawl_url",
    description=(
        "Fetch a web page and return its main content as clean markdown, "
        "rendering JavaScript with a headless browser. Use for articles and "
        "dynamic pages. The result is a parsed markdown artifact you can then "
        "search with bm25_search or read with read_artifact. For binary files "
        "(PDF, DOCX) use download_file + parse_document instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to crawl."},
        },
        "required": ["url"],
    },
    handler=_handle,
    timeout=360.0,
    breakable=True,
)
