"""Shared test helpers. Every test runs offline: no API key, no network.

The sample documents are indexed once per test session with the local
hashing embedder; tests that modify an index build their own.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from knowledge_assistant.assistant import KnowledgeAssistant
from knowledge_assistant.config import PROJECT_ROOT, Settings
from knowledge_assistant.embeddings import LocalHashEmbedder
from knowledge_assistant.ingestion import IngestionService, MemoryStorage
from knowledge_assistant.llm import AnswerPayload, GenerationResult, Statement, TokenUsage
from knowledge_assistant.retrieval import BM25Retriever, SemanticRetriever
from knowledge_assistant.vector_store import ChromaStore

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_documents"
EVAL_SET = PROJECT_ROOT / "eval" / "dataset.jsonl"
FIXTURES = PROJECT_ROOT / "eval" / "fixtures"


def make_settings(tmp_path: Path | None = None, **overrides) -> Settings:
    base = Settings(embedding_provider="local", data_dir=tmp_path or PROJECT_ROOT / "data")
    return replace(base, **overrides)


def new_store(embedder: LocalHashEmbedder | None = None) -> ChromaStore:
    embedder = embedder or LocalHashEmbedder()
    return ChromaStore.in_memory(
        f"test-{uuid.uuid4().hex[:10]}", embedder.signature, embedder.dimensions
    )


@pytest.fixture(scope="session")
def embedder() -> LocalHashEmbedder:
    return LocalHashEmbedder()


@pytest.fixture(scope="session")
def sample_store(embedder: LocalHashEmbedder) -> ChromaStore:
    """The 32 sample documents, indexed once. Do not modify it in tests."""
    store = new_store(embedder)
    results = IngestionService(store, embedder, make_settings(), MemoryStorage()).ingest_folder(
        SAMPLE_DIR
    )
    assert all(result.ok for result in results)
    return store


@pytest.fixture
def empty_store(embedder: LocalHashEmbedder) -> ChromaStore:
    return new_store(embedder)


@pytest.fixture
def ingestion(empty_store: ChromaStore, embedder: LocalHashEmbedder) -> IngestionService:
    return IngestionService(empty_store, embedder, make_settings(), MemoryStorage())


@pytest.fixture(scope="session")
def bm25(sample_store: ChromaStore) -> BM25Retriever:
    return BM25Retriever(sample_store)


Response = AnswerPayload | Exception | Callable[[str, str], AnswerPayload]


class FakeGenerator:
    """Returns scripted answers (or raises scripted errors) and records prompts."""

    def __init__(self, *responses: Response, model: str = "gemini-3.5-flash-lite") -> None:
        self.model = model
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        self.calls.append((system_prompt, user_prompt))
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(response, Exception):
            raise response
        payload = response(system_prompt, user_prompt) if callable(response) else response
        return GenerationResult(
            payload=payload,
            usage=TokenUsage(prompt_tokens=1200, output_tokens=80, thinking_tokens=20),
            model=self.model,
            latency_ms=5.0,
        )


def answered(*statements: tuple[str, list[int]] | tuple[str, list[int], str], note: str = ""):
    return AnswerPayload(
        status="answered",
        statements=[
            Statement(
                text=item[0], sources=item[1], support=item[2] if len(item) > 2 else "explicit"
            )
            for item in statements
        ],
        note=note,
    )


def abstained(note: str = "The documents do not cover this.") -> AnswerPayload:
    return AnswerPayload(status="insufficient_evidence", statements=[], note=note)


def make_assistant(
    store: ChromaStore, *responses: Response, retriever=None, **settings_overrides
) -> tuple[KnowledgeAssistant, FakeGenerator]:
    generator = FakeGenerator(*responses)
    settings = make_settings(top_k=3, **settings_overrides)
    retriever = retriever or SemanticRetriever(store, LocalHashEmbedder())
    return KnowledgeAssistant(store, retriever, generator, settings), generator
