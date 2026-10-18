"""Document ingestion: validate, extract, chunk, embed, store.

Re-ingestion rules, so that repeated uploads never create duplicates:
- the same content (same SHA-256), whatever the file name: skipped;
- a new content under a file name that is already indexed: the new version
  replaces the old one (its chunks are deleted after the new ones are ready);
- a new file with the same document ID as an indexed one: both are kept, and
  the older version is treated as superseded (see vector_store.py).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_document, embedding_text
from .config import Settings
from .embeddings import Embedder
from .errors import DocumentError
from .extraction import SUPPORTED_EXTENSIONS, extract_document, safe_file_name
from .guardrails import looks_like_injection
from .logging_setup import log_event
from .vector_store import ChromaStore

logger = logging.getLogger(__name__)

INDEXED = "indexed"
REPLACED = "replaced"
DUPLICATE = "duplicate"
REJECTED = "rejected"


@dataclass
class IngestResult:
    file_name: str
    status: str
    message: str
    doc_key: str | None = None
    title: str | None = None
    pages: int | None = None
    chunks: int = 0
    flagged_chunks: int = 0
    error_code: str | None = None
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in (INDEXED, REPLACED, DUPLICATE)


class DocumentStorage:
    """Keeps the original files, so that citations can link to them."""

    def save(self, doc_key: str, file_name: str, data: bytes) -> None: ...

    def load(self, doc_key: str) -> tuple[str, bytes] | None: ...

    def delete(self, doc_key: str) -> None: ...


class FileStorage(DocumentStorage):
    """Originals on disk: <directory>/<doc_key>/<file name>."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, doc_key: str, file_name: str, data: bytes) -> None:
        folder = self.directory / doc_key
        folder.mkdir(parents=True, exist_ok=True)
        (folder / safe_file_name(file_name)).write_bytes(data)

    def load(self, doc_key: str) -> tuple[str, bytes] | None:
        folder = self.directory / doc_key
        if not folder.is_dir():
            return None
        files = sorted(path for path in folder.iterdir() if path.is_file())
        return (files[0].name, files[0].read_bytes()) if files else None

    def delete(self, doc_key: str) -> None:
        folder = self.directory / doc_key
        if folder.is_dir():
            for path in folder.iterdir():
                path.unlink()
            folder.rmdir()


class MemoryStorage(DocumentStorage):
    """Originals in memory (public demo sessions, tests)."""

    def __init__(self) -> None:
        self._files: dict[str, tuple[str, bytes]] = {}

    def save(self, doc_key: str, file_name: str, data: bytes) -> None:
        self._files[doc_key] = (safe_file_name(file_name), data)

    def load(self, doc_key: str) -> tuple[str, bytes] | None:
        return self._files.get(doc_key)

    def delete(self, doc_key: str) -> None:
        self._files.pop(doc_key, None)


class IngestionService:
    def __init__(
        self,
        store: ChromaStore,
        embedder: Embedder,
        settings: Settings,
        storage: DocumentStorage | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.storage = storage or MemoryStorage()

    def ingest_bytes(self, file_name: str, data: bytes) -> IngestResult:
        """Ingest one file. Problems with the file itself are returned as a
        rejected result; API errors (embedding) are raised to the caller."""
        start = time.perf_counter()
        name = safe_file_name(file_name)
        try:
            document = extract_document(
                name, data, self.settings.max_file_bytes, self.settings.max_pages
            )
        except DocumentError as exc:
            result = IngestResult(name, REJECTED, str(exc), error_code=exc.code)
            return self._finish(result, start)

        if self.store.has_document(document.doc_key):
            existing = self.store.document(document.doc_key)
            same_as = existing.file_name if existing else name
            result = IngestResult(
                name,
                DUPLICATE,
                f"Already indexed (identical to {same_as}).",
                doc_key=document.doc_key,
                title=document.metadata.title,
            )
            return self._finish(result, start)

        chunks = chunk_document(document, self.settings.chunk_size, self.settings.chunk_overlap)
        meta = document.metadata
        vectors = self.embedder.embed_documents(
            [embedding_text(chunk, meta.title, meta.doc_id, meta.version) for chunk in chunks]
        )
        flags = [looks_like_injection(chunk.text) for chunk in chunks]

        previous = self.store.doc_keys_for_file(name)
        self.store.add_document(document, chunks, vectors, flags)
        for doc_key in previous:
            self.store.delete_document(doc_key)
            self.storage.delete(doc_key)
        self.storage.save(document.doc_key, name, data)

        warnings = list(document.warnings)
        if any(flags):
            warnings.append(
                f"{sum(flags)} passage(s) contain text that looks like instructions to an AI "
                "system. They are indexed but labelled as untrusted."
            )
        status = REPLACED if previous else INDEXED
        pages = f", {document.page_count} pages" if document.page_count else ""
        message = f"Indexed {len(chunks)} passages{pages}."
        if previous:
            message = f"Replaced the previous version. {message}"
        result = IngestResult(
            name,
            status,
            message,
            doc_key=document.doc_key,
            title=meta.title,
            pages=document.page_count,
            chunks=len(chunks),
            flagged_chunks=sum(flags),
            warnings=warnings,
        )
        return self._finish(result, start)

    def ingest_paths(self, paths: Iterable[Path]) -> list[IngestResult]:
        return [self.ingest_bytes(path.name, path.read_bytes()) for path in paths]

    def ingest_folder(self, folder: Path) -> list[IngestResult]:
        """Ingest every supported file of a folder (not its subfolders)."""
        paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        return self.ingest_paths(paths)

    def delete(self, doc_key: str) -> None:
        self.store.delete_document(doc_key)
        self.storage.delete(doc_key)

    def rebuild(self, extra_folder: Path | None = None) -> list[IngestResult]:
        """Empty the index and ingest again all the stored originals (and the
        files of `extra_folder`), for example after changing the chunk size."""
        originals = []
        for document in self.store.documents():
            stored = self.storage.load(document.doc_key)
            if stored:
                originals.append(stored)
        self.store.reset()
        results = [self.ingest_bytes(name, data) for name, data in originals]
        if extra_folder is not None:
            results += self.ingest_folder(extra_folder)
        return results

    def _finish(self, result: IngestResult, start: float) -> IngestResult:
        result.duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log_event(
            logger,
            "ingest",
            file_name=result.file_name,
            status=result.status,
            error_code=result.error_code,
            doc_key=result.doc_key,
            chunks=result.chunks,
            flagged_chunks=result.flagged_chunks,
            duration_ms=result.duration_ms,
        )
        return result
