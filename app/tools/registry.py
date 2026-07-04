"""Registry of all tools available to the agent executor.

Add new tools here — the executor and function declarations pick them up
automatically (docs/implementation-plan.md §6: extend via new code).
"""

from __future__ import annotations

from app.tools.analyze_image import analyze_image_tool
from app.tools.base import Tool
from app.tools.bash_exec import bash_exec_tool
from app.tools.bm25_search import bm25_search_tool
from app.tools.browser_use_agent import browser_use_tool
from app.tools.crawl_url import crawl_url_tool
from app.tools.deep_think import deep_think_tool
from app.tools.discover_sitemap import discover_sitemap_tool
from app.tools.doc_navigate import doc_navigate_tool
from app.tools.download_file import download_file_tool
from app.tools.gemini_search import gemini_search_tool
from app.tools.parse_document import parse_document_tool, read_artifact_tool
from app.tools.python_exec import python_exec_tool
from app.tools.spawn_subagents import spawn_subagents_tool
from app.tools.web_search import web_search_tool

ALL_TOOLS: list[Tool] = [
    web_search_tool,
    download_file_tool,
    parse_document_tool,
    read_artifact_tool,
    bm25_search_tool,
    doc_navigate_tool,
    python_exec_tool,
    bash_exec_tool,
    crawl_url_tool,
    discover_sitemap_tool,
    analyze_image_tool,
    gemini_search_tool,
    browser_use_tool,
    deep_think_tool,
    spawn_subagents_tool,
]

TOOLS_BY_NAME: dict[str, Tool] = {tool.name: tool for tool in ALL_TOOLS}

__all__ = ["ALL_TOOLS", "TOOLS_BY_NAME"]
