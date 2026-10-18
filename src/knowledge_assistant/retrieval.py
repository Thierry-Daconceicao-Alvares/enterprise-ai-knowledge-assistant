"""Find the passages that are relevant to a question.

- SemanticRetriever (used by the application): cosine similarity between the
  embedding of the question and the embeddings of the passages, in Chroma.
  It matches meaning, so it works with paraphrases and across languages.
- BM25Retriever (baseline for the evaluation): classic keyword ranking. It
  needs no API but only finds passages that share words with the question.
- HybridRetriever (experiment): merges the two rankings with reciprocal rank
  fusion. It is evaluated, not used by default.

Superseded documents are excluded unless `include_superseded` is set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import Embedder
from .text import tokenize
from .vector_store import ChromaStore, StoredChunk


@dataclass(frozen=True)
class SearchHit:
    """A retrieved passage. `score` orders the results of one retriever;
    `similarity` is the cosine similarity with the question when the passage
    was found by semantic search (None for a keyword-only match)."""

    chunk: StoredChunk
    score: float
    rank: int
    similarity: float | None = None


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int, include_superseded: bool = False) -> list[SearchHit]: ...


def _allowed_keys(store: ChromaStore, include_superseded: bool) -> list[str] | None:
    if include_superseded:
        return None
    current = store.current_doc_keys()
    return None if len(current) == len(store.documents()) else current


class SemanticRetriever:
    name = "semantic"

    def __init__(self, store: ChromaStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, query: str, k: int, include_superseded: bool = False) -> list[SearchHit]:
        if self.store.chunk_count() == 0:
            return []
        vector = self.embedder.embed_query(query)
        results = self.store.query(vector, k, _allowed_keys(self.store, include_superseded))
        return [
            SearchHit(chunk=chunk, score=round(score, 4), rank=rank, similarity=round(score, 4))
            for rank, (chunk, score) in enumerate(results, start=1)
        ]


class BM25Retriever:
    """Okapi BM25 over the stored chunks, rebuilt when the index changes."""

    name = "bm25"

    def __init__(self, store: ChromaStore, k1: float = 1.5, b: float = 0.75) -> None:
        self.store = store
        self._k1, self._b = k1, b
        self._cache: dict[bool, tuple[int, list[StoredChunk], BM25Okapi | None]] = {}

    def _index(self, include_superseded: bool) -> tuple[list[StoredChunk], BM25Okapi | None]:
        cached = self._cache.get(include_superseded)
        if cached and cached[0] == self.store.revision:
            return cached[1], cached[2]
        keys = None if include_superseded else self.store.current_doc_keys()
        chunks = self.store.chunks(keys)
        corpus = [tokenize(f"{chunk.title} {chunk.section_label} {chunk.text}") for chunk in chunks]
        bm25 = BM25Okapi(corpus, k1=self._k1, b=self._b) if chunks else None
        self._cache[include_superseded] = (self.store.revision, chunks, bm25)
        return chunks, bm25

    def search(self, query: str, k: int, include_superseded: bool = False) -> list[SearchHit]:
        chunks, bm25 = self._index(include_superseded)
        tokens = tokenize(query)
        if bm25 is None or not tokens:
            return []
        scores = np.asarray(bm25.get_scores(tokens), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        hits: list[SearchHit] = []
        for position in order[:k]:
            if scores[position] <= 0:  # no word in common with the question
                break
            hits.append(
                SearchHit(
                    chunk=chunks[position],
                    score=round(float(scores[position]), 4),
                    rank=len(hits) + 1,
                )
            )
        return hits


class HybridRetriever:
    """Reciprocal rank fusion: each passage scores the sum of 1 / (rrf_k + rank)
    over the rankings it appears in. Only ranks are used, which avoids
    calibrating BM25 scores against cosine similarities."""

    name = "hybrid"

    def __init__(
        self, retrievers: Sequence[Retriever], rrf_k: int = 60, candidates: int = 20
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever.")
        self._retrievers = list(retrievers)
        self._rrf_k = rrf_k
        self._candidates = candidates

    def search(self, query: str, k: int, include_superseded: bool = False) -> list[SearchHit]:
        fused: dict[str, float] = {}
        chunks: dict[str, StoredChunk] = {}
        similarities: dict[str, float] = {}
        for retriever in self._retrievers:
            for hit in retriever.search(query, max(k, self._candidates), include_superseded):
                key = hit.chunk.chunk_id
                fused[key] = fused.get(key, 0.0) + 1.0 / (self._rrf_k + hit.rank)
                chunks[key] = hit.chunk
                if hit.similarity is not None:
                    similarities[key] = hit.similarity
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [
            SearchHit(
                chunk=chunks[key],
                score=round(score, 6),
                rank=rank,
                similarity=similarities.get(key),
            )
            for rank, (key, score) in enumerate(ranked, start=1)
        ]
