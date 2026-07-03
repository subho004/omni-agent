"""In-session cache for idempotent, expensive read tools.

Dedupes identical tool calls within a session (same tool + same args) so that
replanning does not re-fetch the same URL or re-run the same search. Only
read-only tools are cacheable; stateful ones (python_exec) and dynamic agents
(browser_use) are excluded. State is in-memory, keyed by (session, tool, args).
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

# Tools whose results are safe to reuse for identical arguments in a session.
CACHEABLE_TOOLS = {
    "web_search",
    "gemini_search",
    "download_file",
    "parse_document",
    "read_artifact",
    "bm25_search",
    "doc_navigate",
    "analyze_image",
    "crawl_url",
}

_cache: dict[tuple[UUID, str, str], dict[str, Any]] = {}


def _key(session_id: UUID, tool_name: str, args: dict[str, Any]) -> tuple[UUID, str, str]:
    return (session_id, tool_name, json.dumps(args, sort_keys=True, default=str))


def is_cacheable(tool_name: str) -> bool:
    return tool_name in CACHEABLE_TOOLS


def get(session_id: UUID, tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    return _cache.get(_key(session_id, tool_name, args))


def put(
    session_id: UUID, tool_name: str, args: dict[str, Any], result: dict[str, Any]
) -> None:
    _cache[_key(session_id, tool_name, args)] = result


def reset(session_id: UUID) -> None:
    for key in [k for k in _cache if k[0] == session_id]:
        _cache.pop(key, None)


__all__ = ["CACHEABLE_TOOLS", "is_cacheable", "get", "put", "reset"]
