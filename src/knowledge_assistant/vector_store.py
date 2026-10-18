"""Persistent storage of the passages, their vectors and their metadata (Chroma).

One Chroma collection holds the chunks of all documents. Each chunk carries
the metadata of its document (file name, title, document ID, version, status)
and its position (section, pages). The list of documents is derived from these
metadata, so there is a single source of truth.

Version handling: a document is treated as superseded when it says so in its
header ("Status: Superseded"), or when another indexed document has the same
document ID and a higher version. Superseded documents stay in the index for
traceability but are excluded from retrieval by default.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from .chunking import Chunk
from .errors import IndexMismatchError
from .extraction import SUPERSEDED_STATUSES, ExtractedDocument, version_key

CURRENT = "current"
SUPERSEDED = "superseded"
_BATCH = 500


@dataclass(frozen=True)
class StoredChunk:
    """A chunk as stored in the index, with the metadata needed to cite it."""

    chunk_id: str
    doc_key: str
    text: str
    file_name: str
    file_type: str
    title: str
    doc_id: str | None
    version: str | None
    section: str | None
    page_start: int | None
    page_end: int | None
    chunk_index: int
    flagged: bool

    @property
    def pages_label(self) -> str | None:
        if self.page_start is None:
            return None
        if self.page_end is None or self.page_end == self.page_start:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    @property
    def section_label(self) -> str:
        return self.section or "Document header"

    @classmethod
    def from_record(cls, chunk_id: str, text: str, meta: dict[str, Any]) -> StoredChunk:
        return cls(
            chunk_id=chunk_id,
            doc_key=str(meta["doc_key"]),
            text=text,
            file_name=str(meta.get("file_name", "")),
            file_type=str(meta.get("file_type", "")),
            title=str(meta.get("title", "")),
            doc_id=meta.get("doc_id"),
            version=meta.get("version"),
            section=meta.get("section"),
            page_start=meta.get("page_start"),
            page_end=meta.get("page_end"),
            chunk_index=int(meta.get("chunk_index", 0)),
            flagged=bool(meta.get("flagged", False)),
        )


@dataclass(frozen=True)
class DocumentInfo:
    """One indexed document, as shown in the document list."""

    doc_key: str
    file_name: str
    file_type: str
    title: str
    doc_id: str | None
    version: str | None
    effective_date: str | None
    declared_status: str | None
    page_count: int | None
    size_bytes: int
    indexed_at: str
    chunk_count: int = 0
    flagged_chunks: int = 0
    status: str = CURRENT
    status_reason: str | None = None


def resolve_versions(documents: Sequence[DocumentInfo]) -> list[DocumentInfo]:
    """Mark the documents that are superseded, and say why."""
    by_id: dict[str, list[DocumentInfo]] = defaultdict(list)
    for document in documents:
        if document.doc_id:
            by_id[document.doc_id.upper()].append(document)

    def rank(document: DocumentInfo) -> tuple[tuple[int, ...], str]:
        return version_key(document.version), document.effective_date or ""

    resolved = []
    for document in documents:
        status, reason = CURRENT, None
        if (document.declared_status or "") in SUPERSEDED_STATUSES:
            status = SUPERSEDED
            reason = f"marked '{document.declared_status}' in the document"
        elif document.doc_id:
            latest = max(by_id[document.doc_id.upper()], key=rank)
            if rank(latest) > rank(document):
                status = SUPERSEDED
                version = f"version {latest.version}" if latest.version else "a newer version"
                reason = f"replaced by {version} ({latest.file_name})"
        resolved.append(replace(document, status=status, status_reason=reason))
    return resolved


def _clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be strings, numbers or booleans."""
    return {key: value for key, value in meta.items() if value is not None}


class ChromaStore:
    """Chunks, vectors and metadata in one Chroma collection.

    The collection records the embedding model that produced its vectors.
    Opening it with another model raises IndexMismatchError instead of
    silently comparing incompatible vectors.
    """

    def __init__(self, client: Any, name: str, signature: str, dimensions: int) -> None:
        self._client = client
        self.name = name
        self.signature = signature
        self.dimensions = dimensions
        self._lock = threading.RLock()
        self._documents: list[DocumentInfo] | None = None
        self.revision = 0  # incremented at every change, used to refresh caches
        self._collection = client.get_or_create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_signature": signature, "dimensions": dimensions},
            embedding_function=None,
        )
        stored = (self._collection.metadata or {}).get("embedding_signature")
        if stored and stored != signature:
            raise IndexMismatchError(
                f"The index '{name}' was built with {stored}, but the current embedding "
                f"model is {signature}. Rebuild the index (see the README)."
            )

    # ------------------------------------------------------------------ factories

    @classmethod
    def persistent(cls, path: Path, name: str, signature: str, dimensions: int) -> ChromaStore:
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        return cls(client, name, signature, dimensions)

    @classmethod
    def in_memory(cls, name: str, signature: str, dimensions: int) -> ChromaStore:
        client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
        return cls(client, name, signature, dimensions)

    # --------------------------------------------------------------------- writes

    def add_document(
        self,
        document: ExtractedDocument,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        flags: Sequence[bool],
    ) -> None:
        if len(chunks) != len(vectors) or len(chunks) != len(flags):
            raise ValueError("chunks, vectors and flags must have the same length")
        if vectors.ndim != 2 or (len(vectors) and vectors.shape[1] != self.dimensions):
            raise IndexMismatchError(
                f"Vectors have shape {vectors.shape}, expected {self.dimensions} dimensions."
            )
        meta = document.metadata
        base = {
            "doc_key": document.doc_key,
            "file_name": document.file_name,
            "file_type": document.file_type,
            "title": meta.title,
            "doc_id": meta.doc_id,
            "version": meta.version,
            "effective_date": meta.effective_date,
            "declared_status": meta.status,
            "owner": meta.owner,
            "page_count": document.page_count,
            "size_bytes": document.size_bytes,
            "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "chunk_count": len(chunks),
        }
        records = [
            (
                chunk.chunk_id,
                vector.tolist(),
                chunk.text,
                _clean_metadata(
                    {
                        **base,
                        "chunk_index": chunk.index,
                        "section": chunk.section,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "flagged": bool(flag),
                    }
                ),
            )
            for chunk, vector, flag in zip(chunks, vectors, flags, strict=True)
        ]
        with self._lock:
            for start in range(0, len(records), _BATCH):
                batch = records[start : start + _BATCH]
                self._collection.upsert(
                    ids=[record[0] for record in batch],
                    embeddings=[record[1] for record in batch],
                    documents=[record[2] for record in batch],
                    metadatas=[record[3] for record in batch],
                )
            self._changed()

    def delete_document(self, doc_key: str) -> None:
        with self._lock:
            self._collection.delete(where={"doc_key": doc_key})
            self._changed()

    def reset(self) -> None:
        """Remove every chunk (the collection and its settings are kept)."""
        with self._lock:
            keys = [document.doc_key for document in self.documents()]
            for key in keys:
                self._collection.delete(where={"doc_key": key})
            self._changed()

    def copy_to(self, other: ChromaStore) -> None:
        """Copy all chunks and vectors into another store (used to give each
        visitor of the public demo a private copy of the sample index)."""
        if other.signature != self.signature:
            raise IndexMismatchError("Cannot copy between indexes built with different models.")
        data = self._collection.get(include=["embeddings", "documents", "metadatas"])
        ids = data["ids"]
        with other._lock:
            for start in range(0, len(ids), _BATCH):
                end = start + _BATCH
                other._collection.upsert(
                    ids=ids[start:end],
                    embeddings=data["embeddings"][start:end],
                    documents=data["documents"][start:end],
                    metadatas=data["metadatas"][start:end],
                )
            other._changed()

    def drop(self) -> None:
        """Delete the collection itself (used for temporary demo sessions)."""
        with self._lock:
            self._client.delete_collection(self.name)
            self._changed()

    def _changed(self) -> None:
        self._documents = None
        self.revision += 1

    # ---------------------------------------------------------------------- reads

    def chunk_count(self) -> int:
        return self._collection.count()

    def documents(self) -> list[DocumentInfo]:
        """All indexed documents, with their resolved status, sorted by file name."""
        with self._lock:
            if self._documents is None:
                data = self._collection.get(include=["metadatas"])
                grouped: dict[str, DocumentInfo] = {}
                counts: dict[str, int] = defaultdict(int)
                flagged: dict[str, int] = defaultdict(int)
                for meta in data["metadatas"] or []:
                    key = str(meta["doc_key"])
                    counts[key] += 1
                    flagged[key] += int(bool(meta.get("flagged", False)))
                    if key not in grouped:
                        grouped[key] = DocumentInfo(
                            doc_key=key,
                            file_name=str(meta.get("file_name", "")),
                            file_type=str(meta.get("file_type", "")),
                            title=str(meta.get("title", "")),
                            doc_id=meta.get("doc_id"),
                            version=meta.get("version"),
                            effective_date=meta.get("effective_date"),
                            declared_status=meta.get("declared_status"),
                            page_count=meta.get("page_count"),
                            size_bytes=int(meta.get("size_bytes", 0)),
                            indexed_at=str(meta.get("indexed_at", "")),
                        )
                documents = [
                    replace(doc, chunk_count=counts[key], flagged_chunks=flagged[key])
                    for key, doc in grouped.items()
                ]
                resolved = resolve_versions(documents)
                self._documents = sorted(resolved, key=lambda doc: doc.file_name.lower())
            return list(self._documents)

    def document(self, doc_key: str) -> DocumentInfo | None:
        return next((doc for doc in self.documents() if doc.doc_key == doc_key), None)

    def has_document(self, doc_key: str) -> bool:
        return self.document(doc_key) is not None

    def doc_keys_for_file(self, file_name: str) -> list[str]:
        return [doc.doc_key for doc in self.documents() if doc.file_name == file_name]

    def current_doc_keys(self) -> list[str]:
        return [doc.doc_key for doc in self.documents() if doc.status == CURRENT]

    def chunks(self, doc_keys: Sequence[str] | None = None) -> list[StoredChunk]:
        where = {"doc_key": {"$in": list(doc_keys)}} if doc_keys is not None else None
        if doc_keys is not None and not doc_keys:
            return []
        data = self._collection.get(where=where, include=["documents", "metadatas"])
        chunks = [
            StoredChunk.from_record(chunk_id, text or "", meta or {})
            for chunk_id, text, meta in zip(
                data["ids"], data["documents"], data["metadatas"], strict=True
            )
        ]
        return sorted(chunks, key=lambda chunk: (chunk.file_name, chunk.chunk_index))

    def query(
        self, vector: np.ndarray, k: int, doc_keys: Sequence[str] | None = None
    ) -> list[tuple[StoredChunk, float]]:
        """The k chunks closest to the vector, with their cosine similarity
        (1 = same direction, 0 = unrelated). `doc_keys` restricts the search to
        some documents."""
        if doc_keys is not None and not doc_keys:
            return []
        total = self.chunk_count()
        if total == 0:
            return []
        where = {"doc_key": {"$in": list(doc_keys)}} if doc_keys is not None else None
        result = self._collection.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
            n_results=min(k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            # Chroma's cosine distance is 1 - cosine similarity.
            hits.append((StoredChunk.from_record(chunk_id, text or "", meta or {}), 1.0 - distance))
        return hits
