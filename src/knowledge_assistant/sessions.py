"""Private indexes for the visitors of the public demonstration.

In public demo mode, the sample documents are indexed once, in a base index.
Each browser session receives its own in-memory copy of that index (vectors
are copied, not recomputed), so that a visitor's uploads are only visible to
that visitor and disappear with the session. Copies that have not been used
for a while are deleted.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from .factory import Components, build_components
from .ingestion import MemoryStorage
from .vector_store import ChromaStore


class SessionIndexes:
    def __init__(self, base: Components, ttl_s: float = 3600, max_sessions: int = 50) -> None:
        self.base = base
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._stores: dict[str, tuple[ChromaStore, float]] = {}
        self._lock = threading.Lock()

    def create(self) -> tuple[str, Components]:
        """A new private copy of the base index, with its own ingestion service."""
        self.cleanup()
        name = f"session-{uuid.uuid4().hex[:12]}"
        embedder = self.base.embedder
        store = ChromaStore.in_memory(name, embedder.signature, embedder.dimensions)
        self.base.store.copy_to(store)
        components = build_components(
            self.base.settings,
            store=store,
            storage=MemoryStorage(),
            client=self.base.client,
            embedder=embedder,
        )
        with self._lock:
            self._stores[name] = (store, time.monotonic())
        return name, components

    def touch(self, name: str) -> bool:
        """Mark a session as active. False if it was already deleted."""
        with self._lock:
            if name not in self._stores:
                return False
            self._stores[name] = (self._stores[name][0], time.monotonic())
            return True

    def cleanup(self) -> int:
        """Delete idle sessions, and the oldest ones beyond the maximum."""
        now = time.monotonic()
        with self._lock:
            by_age = sorted(self._stores.items(), key=lambda item: item[1][1])
            expired = [name for name, (_, seen) in by_age if now - seen > self.ttl_s]
            overflow = max(0, len(by_age) - len(expired) - self.max_sessions + 1)
            expired += [name for name, _ in by_age if name not in expired][:overflow]
            removed = [self._stores.pop(name)[0] for name in expired]
        for store in removed:
            store.drop()
        return len(removed)

    def __len__(self) -> int:
        return len(self._stores)


def load_original(
    components: Components, doc_key: str, sample_dir: Path
) -> tuple[str, bytes] | None:
    """The original file of a document: from the storage of the ingestion
    service, or from the sample folder for the preloaded sample documents."""
    stored = components.ingestion.storage.load(doc_key)
    if stored:
        return stored
    document = components.store.document(doc_key)
    if document:
        path = sample_dir / document.file_name
        if path.is_file():
            return path.name, path.read_bytes()
    return None
