"""Assemble the components from the settings."""

from __future__ import annotations

import re
from dataclasses import dataclass

from google import genai

from .assistant import KnowledgeAssistant
from .config import Settings
from .embeddings import Embedder, GeminiEmbedder, LocalHashEmbedder
from .gemini import make_client
from .ingestion import DocumentStorage, FileStorage, IngestionService
from .llm import GeminiGenerator
from .retrieval import BM25Retriever, HybridRetriever, Retriever, SemanticRetriever
from .vector_store import ChromaStore


def collection_name(settings: Settings, embedder: Embedder) -> str:
    """One collection per embedding model: vectors of different models are
    never mixed. Chroma names allow letters, digits, dots, dashes and underscores."""
    raw = f"{settings.collection_name}-{embedder.signature}"
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")
    return name[:60] or "documents"


def build_client(settings: Settings) -> genai.Client | None:
    """A Gemini client, or None when no API key is configured."""
    return make_client(settings) if settings.gemini_api_key else None


def build_embedder(settings: Settings, client: genai.Client | None = None) -> Embedder:
    if settings.embedding_provider == "local":
        return LocalHashEmbedder()
    client = client or make_client(settings)  # raises ConfigurationError without a key
    return GeminiEmbedder(client, settings.embedding_model, settings.embedding_dimensions)


def build_store(settings: Settings, embedder: Embedder, persistent: bool = True) -> ChromaStore:
    name = collection_name(settings, embedder)
    if persistent:
        return ChromaStore.persistent(
            settings.index_dir, name, embedder.signature, embedder.dimensions
        )
    return ChromaStore.in_memory(name, embedder.signature, embedder.dimensions)


def build_retriever(name: str, store: ChromaStore, embedder: Embedder) -> Retriever:
    """'semantic' (embeddings), 'bm25' (keywords) or 'hybrid' (both, fused)."""
    if name == "bm25":
        return BM25Retriever(store)
    semantic = SemanticRetriever(store, embedder)
    if name == "hybrid":
        return HybridRetriever([semantic, BM25Retriever(store)])
    return semantic


@dataclass
class Components:
    settings: Settings
    embedder: Embedder
    store: ChromaStore
    ingestion: IngestionService
    assistant: KnowledgeAssistant
    client: genai.Client | None = None


def build_components(
    settings: Settings | None = None,
    *,
    store: ChromaStore | None = None,
    storage: DocumentStorage | None = None,
    client: genai.Client | None = None,
    embedder: Embedder | None = None,
) -> Components:
    """Build the ingestion service and the assistant, sharing one store.

    Without an API key, the assistant can still be built with the local
    embedder (search only); asking a question then raises ConfigurationError.
    """
    settings = settings or Settings.from_env()
    client = client or build_client(settings)
    embedder = embedder or build_embedder(settings, client)
    store = store or build_store(settings, embedder)
    storage = storage or FileStorage(settings.storage_dir)
    generator = (
        GeminiGenerator(
            client,
            settings.generation_model,
            temperature=settings.temperature,
            thinking_level=settings.thinking_level,
        )
        if client is not None
        else None
    )
    embedding_model = settings.embedding_model if isinstance(embedder, GeminiEmbedder) else None
    assistant = KnowledgeAssistant(
        store,
        build_retriever(settings.retriever, store, embedder),
        generator,
        settings,
        embedding_model=embedding_model,
    )
    return Components(
        settings=settings,
        embedder=embedder,
        store=store,
        ingestion=IngestionService(store, embedder, settings, storage),
        assistant=assistant,
        client=client,
    )
