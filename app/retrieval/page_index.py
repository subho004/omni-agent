"""PageIndex-style section tree for reasoning-based retrieval (Phase 4).

Parses a markdown document into an ordered list of heading-delimited
sections, each with a breadcrumb (its heading path), a short preview, and
its full text. The LLM then reasons over this outline to pick the relevant
sections instead of relying on vector similarity — a vectorless,
structure-aware retrieval approach (see docs/implementation-plan.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
PREVIEW_CHARS = 200


@dataclass
class Section:
    """One heading-delimited section of a document."""

    index: int
    level: int
    title: str
    breadcrumb: str
    text: str

    @property
    def preview(self) -> str:
        body = self.text.strip().replace("\n", " ")
        return body[:PREVIEW_CHARS]


def build_sections(markdown: str) -> list[Section]:
    """Split markdown into sections at headings, tracking heading paths.

    Content before the first heading becomes a leading "(preamble)" section
    so nothing is lost. A document with no headings yields a single section.
    """

    lines = markdown.splitlines()
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title) for the current path
    cur_level = 0
    cur_title = "(preamble)"
    cur_breadcrumb = "(preamble)"
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text or sections:
            sections.append(
                Section(
                    index=len(sections),
                    level=cur_level,
                    title=cur_title,
                    breadcrumb=cur_breadcrumb,
                    text=text,
                )
            )

    for line in lines:
        match = _HEADING_RE.match(line)
        if match is None:
            buf.append(line)
            continue
        # A heading closes the previous section and opens a new one.
        flush()
        buf = [line]
        level = len(match.group(1))
        cur_level = level
        cur_title = match.group(2).strip() or "(untitled)"
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, cur_title))
        cur_breadcrumb = " > ".join(title for _, title in stack)

    flush()
    if not sections:  # empty document
        sections.append(
            Section(index=0, level=0, title="(document)", breadcrumb="(document)", text=markdown.strip())
        )
    return sections


def render_outline(sections: list[Section]) -> str:
    """Render sections as an id-tagged outline for LLM navigation."""

    return "\n".join(
        f"[{s.index}] {'  ' * max(s.level - 1, 0)}{s.breadcrumb} — {s.preview}"
        for s in sections
    )


__all__ = ["Section", "build_sections", "render_outline", "PREVIEW_CHARS"]
