"""Web-content tool backed by crawl4ai (docs/implementation-plan.md Phase 5).

Renders a URL in a headless browser and returns clean markdown — no LLM
extraction layer. Best for JavaScript-heavy or article-style pages where a
raw download would miss content. The full markdown is stored as an
artifact; only a bounded excerpt is returned to the model.

Also surfaces a structured link map (absolute href + anchor text, split
internal/external) and stores the raw rendered HTML as an artifact, so an
agent that lands on a base page can see the exact targets to follow next
(source-chasing) or drive deeper DOM automation on the HTML.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

from app.core.config import settings
from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

CRAWL_TIMEOUT_MS = 135_000

# Hrefs that aren't navigable targets worth following.
_SKIP_PREFIXES = ("#", "javascript:", "mailto:", "tel:")


def _normalize_links(
    links: Any, base_url: str, cap: int
) -> tuple[list[dict[str, str]], bool]:
    """Flatten crawl4ai's link map into a bounded, deduped follow-list.

    Accepts both the pydantic form (``links.internal`` / ``links.external``,
    each a list of ``Link`` objects) and a plain-dict fallback, since the shape
    varies by crawl4ai version. crawl4ai already resolves relative hrefs against
    the page URL; we ``urljoin`` again defensively so a relative href from the
    dict path (or a future version) still becomes absolute instead of dropped.
    Keeps ``{text, url, scope}`` for navigable http hrefs only, skips the page's
    own self-links (e.g. bare ``#frag`` anchors), dedupes by url, and caps the
    total. Returns ``(links, truncated)``.

    Note: only ``<a href>`` anchors from the *rendered* (post-JS) DOM appear
    here — JS/jQuery navigation with no href (onclick buttons, SPA click
    handlers, form submits) won't, by design. For those the agent reads the raw
    HTML (html_artifact_id) or drives browser_use.
    """

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for scope in ("internal", "external"):
        group = getattr(links, scope, None)
        if group is None and isinstance(links, dict):
            group = links.get(scope)
        for item in group or []:
            raw = str(getattr(item, "href", None) or (
                item.get("href") if isinstance(item, dict) else "") or "").strip()
            if not raw or raw.lower().startswith(_SKIP_PREFIXES):
                continue
            href = urljoin(base_url, raw)
            if not href.lower().startswith(("http://", "https://")):
                continue
            if href == base_url or href in seen:  # skip self-links + dupes
                continue
            text = str(getattr(item, "text", None) or (
                item.get("text") if isinstance(item, dict) else "") or "").strip()
            seen.add(href)
            out.append({"text": text[:200], "url": href, "scope": scope})
    return out[:cap], len(out) > cap


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
    stem = str(abs(hash(url)))
    md_path = crawled_dir / f"{stem}.md"
    md_path.write_text(markdown, encoding="utf-8")

    artifact = await ctx.artifacts.create(
        session_id=ctx.session_id,
        kind="parsed",
        name=f"crawl-{stem}.md",
        uri=str(md_path),
        summary=f"Crawled {url}",
    )

    # Structured link map for source-chasing: the exact hrefs the agent can
    # follow next (crawl_url again, or download_file for a linked file).
    links, links_truncated = _normalize_links(
        getattr(result, "links", None), url, settings.crawl_max_links
    )

    # Raw rendered HTML as a download artifact, readable back via read_artifact,
    # so an agent can inspect the DOM (hrefs/embedded JSON/form targets the
    # markdown dropped) when the flat link map isn't enough.
    html_artifact_id: str | None = None
    html = str(getattr(result, "html", "") or "")
    if html:
        html_path = crawled_dir / f"{stem}.html"
        html_path.write_text(html, encoding="utf-8")
        html_artifact = await ctx.artifacts.create(
            session_id=ctx.session_id,
            kind="download",
            name=f"crawl-{stem}.html",
            uri=str(html_path),
            summary=f"Raw HTML of {url}",
        )
        html_artifact_id = str(html_artifact.id)

    return {
        "parsed_artifact_id": str(artifact.id),
        "url": url,
        "total_chars": len(markdown),
        "excerpt": markdown[:settings.tool_excerpt_chars],
        "truncated": len(markdown) > settings.tool_excerpt_chars,
        "links": links,
        "link_count": len(links),
        "links_truncated": links_truncated,
        "html_artifact_id": html_artifact_id,
    }


crawl_url_tool = Tool(
    name="crawl_url",
    description=(
        "Fetch a web page and return its main content as clean markdown, "
        "rendering JavaScript with a headless browser. Use for articles and "
        "dynamic pages. The result is a parsed markdown artifact you can then "
        "search with bm25_search or read with read_artifact. For binary files "
        "(PDF, DOCX) use download_file + parse_document instead.\n"
        "Also returns 'links' — a structured list of the page's outbound links "
        "({text, url, scope}) — use these as the exact targets to FOLLOW when a "
        "page references another document (crawl_url them, or download_file a "
        "linked file). 'html_artifact_id' holds the raw rendered HTML: read it "
        "with read_artifact to inspect hrefs, embedded JSON, or form targets the "
        "markdown dropped."
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
