"""Hybrid dense + sparse ranking with Reciprocal Rank Fusion.

The ranking core behind ``corpus_search``. Given a set of document chunks it
combines two complementary signals over the *same* candidate pool:

* **dense** — cosine similarity of embedding vectors (semantic match: finds
  "credential recovery" for a "reset my password" query),
* **sparse** — Okapi BM25 over tokens (exact/lexical match: names, ids, rare
  terms embeddings blur together).

The two ranked lists are merged with **Reciprocal Rank Fusion (RRF)** — the
tuning-free fusion used by Elasticsearch, OpenSearch, and most production RAG
stacks: an item's fused score is ``sum(1 / (k + rank))`` over the lists it
appears in, so agreement near the top of either list wins without needing the
two score scales to be comparable. Everything here is pure (vectors/text in,
ranked indices out) so it can be unit-tested offline.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Plus

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def dense_ranking(query_vec: list[float], doc_vecs: list[list[float]]) -> list[int]:
    """Chunk indices ranked by cosine similarity to the query, best first."""

    if not doc_vecs or not query_vec:
        return []
    matrix = np.asarray(doc_vecs, dtype=np.float32)
    query = np.asarray(query_vec, dtype=np.float32)
    doc_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    denom = doc_norms * query_norm
    scores = np.divide(
        matrix @ query, denom, out=np.zeros(len(doc_vecs), dtype=np.float32),
        where=denom > 0,
    )
    return [int(i) for i in np.argsort(-scores)]


def sparse_ranking(query: str, chunk_texts: list[str]) -> list[int]:
    """Chunk indices ranked by BM25 relevance to the query, best first.

    Only chunks that share at least one query term are ranked; the rest are
    dropped (BM25Plus gives every chunk a positive baseline, so score alone
    can't gate them out on a small single-corpus index).
    """

    if not chunk_texts:
        return []
    tokenized = [_tokenize(t) for t in chunk_texts]
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []
    bm25 = BM25Plus(tokenized)
    scores = bm25.get_scores(list(query_tokens))
    ranked = sorted(range(len(chunk_texts)), key=lambda i: scores[i], reverse=True)
    return [i for i in ranked if query_tokens & set(tokenized[i])]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Fuse several ranked index lists into one, best first.

    ``rankings`` is a list of ranked-index lists (e.g. the dense list and the
    sparse list). Returns ``(index, fused_score)`` pairs sorted by descending
    fused score, where ``fused_score = sum(1 / (k + rank))`` over every list the
    index appears in (rank is 0-based).
    """

    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_search(
    query: str,
    chunk_texts: list[str],
    query_vec: list[float],
    doc_vecs: list[list[float]],
    candidate_k: int,
    rrf_k: int,
) -> list[tuple[int, float]]:
    """Rank chunks by dense+sparse RRF; return the top ``candidate_k`` fused hits."""

    dense = dense_ranking(query_vec, doc_vecs)
    sparse = sparse_ranking(query, chunk_texts)
    fused = reciprocal_rank_fusion([dense, sparse], rrf_k)
    return fused[:candidate_k]


__all__ = [
    "dense_ranking",
    "sparse_ranking",
    "reciprocal_rank_fusion",
    "hybrid_search",
]
