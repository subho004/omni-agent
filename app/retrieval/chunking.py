"""Structure-aware chunking for hybrid corpus retrieval.

Splits a parsed markdown document into overlapping, embed-sized passages while
preserving structure: chunks never span heading boundaries (we chunk within
each PageIndex section) and every chunk carries its heading breadcrumb, so a
retrieved passage still says *where* in the document it came from. This is the
document-side counterpart to ``page_index`` — where ``doc_navigate`` reasons
over whole sections, ``corpus_search`` embeds and ranks these finer chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.page_index import build_sections


@dataclass(frozen=True)
class Chunk:
    """One embed-sized passage of a document, tagged with its location."""

    text: str
    breadcrumb: str
    section_index: int
    offset: int  # char offset of this chunk within its section


def _split_window(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    """Slide a fixed window over ``text``; return (offset, passage) pairs."""

    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [(0, text)]
    step = max(size - overlap, 1)
    windows: list[tuple[int, str]] = []
    for start in range(0, len(text), step):
        piece = text[start : start + size]
        if piece.strip():
            windows.append((start, piece))
        if start + size >= len(text):
            break
    return windows


def chunk_markdown(markdown: str, size: int, overlap: int) -> list[Chunk]:
    """Chunk a markdown document into overlapping, breadcrumb-tagged passages.

    Sections come from the shared PageIndex splitter, so chunks respect heading
    boundaries; each section is then windowed to ``size`` chars with ``overlap``
    carried between adjacent chunks.
    """

    chunks: list[Chunk] = []
    for section in build_sections(markdown):
        for offset, passage in _split_window(section.text, size, overlap):
            chunks.append(
                Chunk(
                    text=passage,
                    breadcrumb=section.breadcrumb,
                    section_index=section.index,
                    offset=offset,
                )
            )
    return chunks


__all__ = ["Chunk", "chunk_markdown"]
