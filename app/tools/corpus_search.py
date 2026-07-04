"""Hybrid cross-document retrieval over the whole session corpus.

``corpus_search`` is the robust, RAG-grade retrieval tool: unlike bm25_search /
doc_navigate (which search *one* artifact), it searches across **every**
uploaded/downloaded document and image in the session at once and returns the
most relevant passages with source attribution — the shape used by production
platforms (OpenAI file_search, Vertex AI RAG Engine, LlamaIndex).

Pipeline per query:

1. **Gather** the corpus — parsed markdown artifacts, image descriptions
   (the vision summary captured at upload), and any not-yet-parsed document
   upload (auto-parsed on the fly, then cached as a parsed artifact).
2. **Chunk** each document structure-aware (``retrieval.chunking``).
3. **Embed** chunks with the embedding model, cached per source artifact on
   disk so repeat queries / later turns never re-embed.
4. **Rank** with dense (embedding cosine) + sparse (BM25) fused via RRF
   (``retrieval.hybrid``).
5. **Rerank** the fused candidates with the LLM for final precision (optional).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.chunking import chunk_markdown
from app.retrieval.hybrid import hybrid_search
from app.schemas.retrieval import RerankSelection
from app.tools.base import Tool, ToolContext, image_mime_for
from app.tools.parse_document import ensure_parsed_markdown

logger = get_logger(__name__)

# Document uploads corpus_search will auto-parse (MarkItDown-supported) when the
# agent hasn't run parse_document on them yet. Images are handled separately via
# their vision summary; anything else is skipped.
_PARSEABLE_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".html", ".htm", ".txt", ".md", ".csv", ".tsv", ".json", ".xml",
    ".rtf", ".epub",
}
_RERANK_PASSAGE_CHARS = 600  # per-candidate text shown to the reranker
MAX_TOP_K = 20

_RERANK_SYSTEM = (
    "You rerank candidate passages for a query. Given a question and numbered "
    "passages (each from a source document), return the ids ordered from most "
    "to least relevant to the question, keeping only genuinely useful ones."
)

# Per-(session, source) locks serialising the two operations that would
# otherwise race when the orchestrator runs sub-agents in parallel and more than
# one calls corpus_search at the same instant: auto-parsing a never-parsed
# upload (else two duplicate parsed artifacts) and first-time embedding of a
# source (else a duplicated embed + concurrent cache-file write). Keyed
# ("parse"|"embed", session_id, source_id); process-local, which is the same
# scope the sub-agents share. Construction is await-free so it's atomic under
# asyncio (no interleave between the get and the set).
_source_locks: dict[tuple[str, str, str], asyncio.Lock] = {}


def _lock_for(key: tuple[str, str, str]) -> asyncio.Lock:
    lock = _source_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _source_locks[key] = lock
    return lock


@dataclass
class _Chunk:
    """A corpus chunk plus its source attribution."""

    text: str
    breadcrumb: str
    source_id: str
    source_name: str


def _cache_path(ctx: ToolContext, source_id: str) -> Path:
    """Per-source embedding cache file, keyed by model + chunking params so a
    settings change transparently invalidates it."""

    stem = (
        f"{source_id}.{settings.embedding_model}.{settings.embedding_dimensions}"
        f".{settings.corpus_chunk_chars}.{settings.corpus_chunk_overlap}.npy"
    ).replace("/", "_")
    return ctx.data_dir / "embeddings" / str(ctx.session_id) / stem


async def _load_cache(path: Path, expected: int) -> list[list[float]] | None:
    """Return cached vectors if the file exists and matches the chunk count."""

    if not path.is_file():
        return None
    cached = await asyncio.to_thread(np.load, path)
    if cached.shape[0] != expected:
        return None
    return [row.tolist() for row in cached]


async def _embed_source(
    ctx: ToolContext, source_id: str, texts: list[str]
) -> list[list[float]]:
    """Embed a source's chunk texts, reusing a valid on-disk cache if present.

    Serialised per (session, source): concurrent sub-agents hitting the same
    not-yet-embedded source wait, and all but the first re-read the cache the
    first one just wrote instead of re-embedding and racing on the file.
    """

    assert ctx.llm is not None
    path = _cache_path(ctx, source_id)
    cached = await _load_cache(path, len(texts))
    if cached is not None:
        return cached

    async with _lock_for(("embed", str(ctx.session_id), source_id)):
        # Re-check inside the lock: a peer may have embedded while we waited.
        cached = await _load_cache(path, len(texts))
        if cached is not None:
            return cached
        vectors = await ctx.llm.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
        if vectors and len(vectors) == len(texts):
            await asyncio.to_thread(_save_vectors, path, vectors)
        return vectors


def _save_vectors(path: Path, vectors: list[list[float]]) -> None:
    """Write vectors atomically (temp file + os.replace) so a reader never sees
    a torn cache file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:  # file handle → np.save won't munge suffix
        np.save(handle, np.asarray(vectors, dtype=np.float32))
    os.replace(tmp, path)


async def _read_text(uri: str) -> str:
    return await asyncio.to_thread(lambda: open(uri, encoding="utf-8").read())


async def _find_parsed_for(ctx: ToolContext, source_name: str) -> Any | None:
    """The parsed artifact produced from ``source_name``, if one exists yet."""

    for artifact in await ctx.artifacts.list_by_session(ctx.session_id):
        if artifact.kind == "parsed" and artifact.name == f"{source_name}.md":
            return artifact
    return None


async def _ensure_parsed_locked(ctx: ToolContext, artifact: Any) -> tuple[Any, str]:
    """Parse an upload to markdown, serialised per (session, source).

    Concurrent sub-agents that both find the source unparsed wait on the lock;
    all but the first re-discover the parsed artifact the first one committed
    (re-checked inside the lock) instead of parsing it a second time.
    """

    async with _lock_for(("parse", str(ctx.session_id), str(artifact.id))):
        existing = await _find_parsed_for(ctx, artifact.name)
        if existing is not None:
            return existing, await _read_text(existing.uri)
        return await ensure_parsed_markdown(ctx, artifact)


async def _gather_corpus(
    ctx: ToolContext, wanted: set[UUID]
) -> tuple[list[_Chunk], list[list[float]], int]:
    """Build the chunk list + aligned embedding matrix across all sources.

    ``wanted`` restricts to specific artifact ids when non-empty. Returns
    (chunks, vectors, documents_searched).
    """

    artifacts = await ctx.artifacts.list_by_session(ctx.session_id)
    parsed_names = {
        a.name for a in artifacts if a.kind == "parsed"
    }  # e.g. "report.pdf.md"

    chunks: list[_Chunk] = []
    vectors: list[list[float]] = []
    documents = 0

    for artifact in artifacts:
        if wanted and artifact.id not in wanted:
            continue

        source_id = str(artifact.id)
        source_name = artifact.name
        markdown: str | None = None

        if artifact.kind == "parsed":
            markdown = await _read_text(artifact.uri)
        elif image_mime_for(artifact.name):
            # Images enter the index via their upload-time vision description.
            summary = (artifact.summary or "").strip()
            if summary and not summary.endswith("bytes)"):
                markdown = f"# Image: {artifact.name}\n\n{summary}"
        elif Path(artifact.name).suffix.lower() in _PARSEABLE_EXTENSIONS:
            if f"{artifact.name}.md" in parsed_names:
                continue  # already represented by its parsed artifact
            try:
                parsed, markdown = await _ensure_parsed_locked(ctx, artifact)
            except Exception:
                logger.exception("corpus_search: failed to parse %s", artifact.name)
                continue
            source_id = str(parsed.id)
            source_name = parsed.name

        if not markdown:
            continue

        source_chunks = chunk_markdown(
            markdown, settings.corpus_chunk_chars, settings.corpus_chunk_overlap
        )
        if not source_chunks:
            continue
        if len(chunks) + len(source_chunks) > settings.corpus_max_chunks:
            logger.warning(
                "corpus_search: chunk cap (%d) reached; skipping remaining sources",
                settings.corpus_max_chunks,
            )
            break

        source_vecs = await _embed_source(
            ctx, source_id, [c.text for c in source_chunks]
        )
        if len(source_vecs) != len(source_chunks):
            logger.warning(
                "corpus_search: embedding count mismatch for %s; skipping",
                source_name,
            )
            continue

        documents += 1
        for chunk, vec in zip(source_chunks, source_vecs, strict=True):
            chunks.append(
                _Chunk(chunk.text, chunk.breadcrumb, source_id, source_name)
            )
            vectors.append(vec)

    return chunks, vectors, documents


async def _rerank(
    ctx: ToolContext, query: str, chunks: list[_Chunk], order: list[int]
) -> list[int]:
    """LLM-rerank fused candidates; fall back to the fused order on any failure."""

    assert ctx.llm is not None
    listing = "\n\n".join(
        f"[{pos}] ({chunks[idx].source_name} — {chunks[idx].breadcrumb})\n"
        f"{chunks[idx].text[:_RERANK_PASSAGE_CHARS]}"
        for pos, idx in enumerate(order)
    )
    try:
        selection, in_tok, out_tok = await ctx.llm.generate_structured(
            prompt=f"Question:\n{query}\n\nCandidate passages:\n{listing}",
            system_instruction=_RERANK_SYSTEM,
            response_schema=RerankSelection,
        )
    except Exception:
        logger.exception("corpus_search: rerank failed; using fused order")
        return order
    if ctx.ledger is not None:
        await ctx.ledger.create(ctx.session_id, ctx.llm.model, in_tok, out_tok)

    if not isinstance(selection, RerankSelection) or not selection.ranked_ids:
        return order
    picked = [order[p] for p in selection.ranked_ids if 0 <= p < len(order)]
    return picked or order


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.llm is None:
        return {"error": "corpus_search unavailable (no LLM client on context)"}

    query = str(args["query"])
    top_k = min(int(args.get("top_k", settings.corpus_top_k)), MAX_TOP_K)
    wanted = {UUID(str(a)) for a in args.get("artifact_ids", []) or []}

    logger.info("corpus_search: %r (filter=%d ids)", query, len(wanted))
    chunks, vectors, documents = await _gather_corpus(ctx, wanted)
    if not chunks:
        return {
            "matches": [],
            "count": 0,
            "documents_searched": 0,
            "note": (
                "No searchable documents found. Upload/parse documents first, "
                "or check the artifact_ids filter."
            ),
        }

    query_vec = (await ctx.llm.embed_texts([query], task_type="RETRIEVAL_QUERY"))
    query_vector = query_vec[0] if query_vec else []

    fused = hybrid_search(
        query=query,
        chunk_texts=[c.text for c in chunks],
        query_vec=query_vector,
        doc_vecs=vectors,
        candidate_k=settings.corpus_candidate_k,
        rrf_k=settings.corpus_rrf_k,
    )
    if not fused:
        return {
            "matches": [],
            "count": 0,
            "documents_searched": documents,
            "chunks_indexed": len(chunks),
        }

    scores = {idx: score for idx, score in fused}
    order = [idx for idx, _ in fused]
    if settings.corpus_rerank:
        order = await _rerank(ctx, query, chunks, order)

    matches = [
        {
            "document": chunks[idx].source_name,
            "artifact_id": chunks[idx].source_id,
            "breadcrumb": chunks[idx].breadcrumb,
            "score": round(scores.get(idx, 0.0), 4),
            "text": chunks[idx].text,
        }
        for idx in order[:top_k]
    ]
    return {
        "matches": matches,
        "count": len(matches),
        "documents_searched": documents,
        "chunks_indexed": len(chunks),
    }


corpus_search_tool = Tool(
    name="corpus_search",
    description=(
        "Search across ALL uploaded/downloaded documents and images in this "
        "session at once and return the most relevant passages with their "
        "source document. This is the best retrieval tool when there are many "
        "files, a large multi-hundred-page document, or the answer may span "
        "several documents — it combines semantic (embedding) and keyword "
        "(BM25) search and reranks the results. Prefer it over bm25_search / "
        "doc_navigate (which search only one document). Optionally pass "
        "artifact_ids to restrict the search to specific files. Not-yet-parsed "
        "documents are parsed automatically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or keywords to search the corpus for.",
            },
            "artifact_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional. Restrict the search to these artifact ids "
                    "(parsed, upload, or image ids from the Files list). Omit "
                    "to search every document in the session."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": (
                    f"Number of passages to return "
                    f"(default {settings.corpus_top_k}, max {MAX_TOP_K})."
                ),
            },
        },
        "required": ["query"],
    },
    handler=_handle,
    timeout=600.0,  # first query may parse + embed the whole corpus
    breakable=True,
)
