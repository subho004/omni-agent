"""Sitemap discovery tool: enumerate a site's page URLs, fast.

When a source site holds a LOT of data spread across many pages, crawling
blindly link-by-link is slow and misses pages that aren't linked from where
the agent landed. This tool asks the site itself for its full URL inventory:
it reads ``robots.txt`` and the well-known sitemap locations, then walks the
sitemaps recursively and returns the same-host page URLs it finds. The agent
then feeds the relevant ones into ``crawl_url`` / ``download_file``.

httpx + parse only — no browser — so it is cheap and broad. It handles every
format the sitemaps.org protocol / Google support (XML sitemap index, XML
urlset, RSS/Atom feeds, plain-text and markdown sitemaps, and ``.gz``
compression on any of them). Bounded by ``settings.sitemap_doc_budget``
(documents fetched) and ``settings.sitemap_max_urls`` (URLs returned).

GUARDRAIL: a large returned count is an *inventory*, not a to-do list — the
agent must still pick which pages actually matter and not render/probe them all.
"""

from __future__ import annotations

import gzip
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.core.config import settings
from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
# Non-HTML sitemap entries worth special handling.
_TEXT_EXTENSIONS = (".md", ".markdown", ".txt")
_XML_EXTENSIONS = (".xml", ".xml.gz")
# Well-known sitemap locations tried when robots.txt names none. Covers standard
# XML sitemaps, plain-text/markdown indexes (llms.txt), and agentic-discovery /
# Shopify UCP files that some merchants publish.
_WELL_KNOWN = (
    "/sitemap.xml", "/sitemap_index.xml", "/sitemap.xml.gz",
    "/sitemap.txt", "/llms.txt", "/llms-full.txt",
    "/agents.md", "/sitemap_agentic_discovery.xml",
    "/.well-known/ucp",
)


def _decode_sitemap_body(resp: httpx.Response) -> str:
    """Return a sitemap document's text, transparently gunzipping when needed.

    Sitemaps may be served gzip-compressed at the *file* level (``.gz`` files
    with Content-Type ``application/gzip``/``octet-stream``) rather than via
    HTTP Content-Encoding — so httpx won't auto-decompress them. Detect the
    gzip magic bytes (``1f 8b``) and inflate manually; otherwise fall back to
    httpx's already-decoded text.
    """

    try:
        content = resp.content or b""
    except Exception:
        return resp.text or ""
    if content[:2] == b"\x1f\x8b":  # gzip magic number
        try:
            return gzip.decompress(content).decode("utf-8", errors="replace")
        except Exception:
            return resp.text or ""
    return resp.text or ""


def _extract_links_from_text(body: str, source_url: str) -> list[str]:
    """Extract URLs from markdown/text/JSON content (absolute + relative).

    Handles markdown links ``[label](url)``, bare absolute URLs, relative paths
    (resolved against ``source_url``), and URLs embedded in JSON string values.
    """

    urls: list[str] = []
    for match in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", body):
        urls.append(match[1].strip())
    for raw in re.findall(r"https?://[^\s<>\"'\)\]},]+", body):
        urls.append(raw.rstrip(".,;:!?\"')]"))

    resolved: list[str] = []
    seen_local: set[str] = set()
    for candidate in urls:
        candidate = candidate.strip()
        if not candidate or candidate.startswith("#"):
            continue
        if not candidate.startswith(("http://", "https://")):
            candidate = urljoin(source_url, candidate)
        if candidate not in seen_local:
            seen_local.add(candidate)
            resolved.append(candidate)
    return resolved


def _attr_str(value: Any) -> str:
    """Normalize a bs4 attribute to a string (multi-valued attrs come as lists)."""

    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value) if value is not None else ""


def _path_ends_with(url: str, extensions: tuple[str, ...]) -> bool:
    """True if the URL's path ends with any of the given extensions."""

    try:
        path = urlparse(url).path.lower().rstrip("/")
    except Exception:
        return False
    return any(path.endswith(ext) for ext in extensions)


async def _discover_sitemap_urls(
    base_url: str,
    *,
    max_urls: int,
    timeout: float,
    sitemap_doc_budget: int,
) -> list[str]:
    """Same-host page URLs mined from robots.txt + sitemap (recursive).

    Content-agnostic: returns ALL same-host page URLs, deciding index-vs-pages by
    document STRUCTURE (parent tag), not a URL-string heuristic, so a sitemap
    literally full of real pages is handled correctly regardless of its filename.
    """

    seeds: list[str] = []
    seen: set[str] = set()
    netloc = urlparse(base_url).netloc.lower()

    def _add(url: str) -> None:
        if not url or url in seen or len(seeds) >= max_urls:
            return
        try:
            if urlparse(url).netloc.lower() != netloc:
                return
        except Exception:
            return
        seen.add(url)
        seeds.append(url)

    headers = {"User-Agent": _USER_AGENT}
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=timeout, headers=headers
        ) as client:
            sitemap_urls: list[str] = []
            try:
                robots = await client.get(urljoin(base_url, "/robots.txt"))
                if robots.status_code == 200:
                    for line in robots.text.splitlines():
                        if line.lower().startswith("sitemap:"):
                            sitemap_urls.append(line.split(":", 1)[1].strip())
            except Exception:
                logger.debug("discover_sitemap: robots.txt fetch failed for %s", base_url)

            sitemap_urls.extend(urljoin(base_url, wk) for wk in _WELL_KNOWN)

            sm_queue = list(dict.fromkeys(sitemap_urls))
            visited_sm: set[str] = set()
            budget = sitemap_doc_budget

            def _queue_sitemap(url: str) -> None:
                if url and url not in visited_sm and url not in sm_queue:
                    sm_queue.append(url)

            while sm_queue and budget > 0 and len(seeds) < max_urls:
                sm = sm_queue.pop(0)
                if not sm or sm in visited_sm:
                    continue
                visited_sm.add(sm)
                budget -= 1
                try:
                    resp = await client.get(sm)
                    if resp.status_code != 200:
                        continue
                    body = _decode_sitemap_body(resp)
                except Exception:
                    continue

                try:
                    xsoup: BeautifulSoup | None = BeautifulSoup(body, "xml")
                except Exception:
                    xsoup = None

                found_structured = False
                if xsoup is not None:
                    # Sitemap INDEX entries (<sitemap><loc>) → nested sitemaps.
                    for sm_tag in xsoup.find_all("sitemap"):
                        loc = sm_tag.find("loc")
                        nested = (loc.get_text() if loc else "").strip()
                        if nested:
                            found_structured = True
                            _queue_sitemap(nested)

                    # URLSET entries (<url><loc>) → actual page URLs. Non-HTML loc
                    # entries: .md/.txt are mined for links; .xml are queued as
                    # nested sitemaps (unusual but observed in the wild).
                    text_file_urls: list[str] = []
                    for url_tag in xsoup.find_all("url"):
                        loc = url_tag.find("loc")
                        page = (loc.get_text() if loc else "").strip()
                        if not page:
                            continue
                        found_structured = True
                        if _path_ends_with(page, _TEXT_EXTENSIONS):
                            text_file_urls.append(page)
                        elif _path_ends_with(page, _XML_EXTENSIONS):
                            _queue_sitemap(page)
                        else:
                            _add(page)
                        if len(seeds) >= max_urls:
                            break

                    for tf_url in text_file_urls:
                        if budget <= 0 or len(seeds) >= max_urls:
                            break
                        if tf_url in visited_sm:
                            continue
                        visited_sm.add(tf_url)
                        budget -= 1
                        try:
                            tf_resp = await client.get(tf_url)
                            if tf_resp.status_code == 200:
                                for link in _extract_links_from_text(
                                    _decode_sitemap_body(tf_resp), tf_url
                                ):
                                    _add(link)
                        except Exception:
                            logger.debug(
                                "discover_sitemap: text-file fetch failed %s", tf_url
                            )

                    # RSS 2.0 / Atom feeds → page URLs live in <link>, not <loc>.
                    if len(seeds) < max_urls and (xsoup.find("rss") or xsoup.find("feed")):
                        for link_tag in xsoup.find_all("link"):
                            # bs4 returns multi-valued attrs (e.g. rel) as a list.
                            if _attr_str(link_tag.get("rel")).lower() == "self":
                                continue  # Atom self-link points back at the feed
                            href = _attr_str(link_tag.get("href")).strip()  # Atom
                            if not href:
                                href = (link_tag.get_text() or "").strip()  # RSS
                            if href:
                                found_structured = True
                                _add(href)
                                if len(seeds) >= max_urls:
                                    break

                # Plain-text / markdown sitemap (sitemap.txt, llms.txt) — no XML
                # structure. Mine absolute URLs and relative markdown links.
                if not found_structured:
                    for url in _extract_links_from_text(body, sm):
                        _add(url)
                        if len(seeds) >= max_urls:
                            break
    except Exception:
        logger.debug("discover_sitemap: failed for %s", base_url, exc_info=True)

    if seeds:
        logger.info(
            "discover_sitemap: found %d sub-page URL(s) for %s", len(seeds), base_url
        )
    return seeds[:max_urls]


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    base_url = str(args.get("base_url") or "").strip()
    if not base_url:
        return {"error": "Provide a 'base_url' (e.g. https://example.com)."}
    if not base_url.startswith(("http://", "https://")):
        base_url = f"https://{base_url}"

    max_urls = settings.sitemap_max_urls
    requested = args.get("max_urls")
    if isinstance(requested, int) and requested > 0:
        max_urls = min(requested, settings.sitemap_max_urls)

    logger.info("discover_sitemap: %s (max_urls=%d)", base_url, max_urls)
    urls = await _discover_sitemap_urls(
        base_url,
        max_urls=max_urls,
        timeout=settings.sitemap_timeout,
        sitemap_doc_budget=settings.sitemap_doc_budget,
    )

    if not urls:
        return {
            "base_url": base_url,
            "count": 0,
            "urls": [],
            "note": (
                "No sitemap/robots URLs found. The site may not publish a "
                "sitemap — crawl_url the base page and follow its 'links' instead."
            ),
        }
    return {
        "base_url": base_url,
        "count": len(urls),
        "urls": urls,
        "truncated": len(urls) >= max_urls,
    }


discover_sitemap_tool = Tool(
    name="discover_sitemap",
    description=(
        "Enumerate a website's page URLs from its robots.txt and sitemaps, fast "
        "(no browser). Use this FIRST when a site has lots of data across many "
        "pages and you need the full inventory rather than clicking link-by-link: "
        "pass the site's base URL and get back a list of same-host page URLs "
        "(handles XML sitemap indexes, urlsets, RSS/Atom, plain-text/markdown and "
        ".gz sitemaps). Then pick the RELEVANT URLs and read them with crawl_url "
        "or download_file — do NOT crawl every URL returned. Returns an empty list "
        "with a note when the site publishes no sitemap."
    ),
    parameters={
        "type": "object",
        "properties": {
            "base_url": {
                "type": "string",
                "description": (
                    "The site's base URL or homepage (e.g. https://example.com). "
                    "robots.txt and well-known sitemap paths are resolved from it."
                ),
            },
            "max_urls": {
                "type": "integer",
                "description": (
                    "Optional cap on URLs to return (bounded by the server limit). "
                    "Omit for the default."
                ),
            },
        },
        "required": ["base_url"],
    },
    handler=_handle,
    timeout=240.0,
    breakable=True,
)
