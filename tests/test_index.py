"""Vector store, ingestion and retrieval, with the local embedder."""

import pytest
from conftest import FIXTURES, SAMPLE_DIR, make_settings, new_store

from knowledge_assistant.embeddings import LocalHashEmbedder
from knowledge_assistant.errors import IndexMismatchError
from knowledge_assistant.ingestion import (
    DUPLICATE,
    INDEXED,
    REJECTED,
    REPLACED,
    FileStorage,
    IngestionService,
    MemoryStorage,
)
from knowledge_assistant.retrieval import BM25Retriever, HybridRetriever, SemanticRetriever
from knowledge_assistant.vector_store import ChromaStore

DOA_V2 = "NX-PUR-002_delegation-of-authority_v2.0.pdf"
DOA_V1 = "NX-PUR-002_delegation-of-authority_v1.0_2023.pdf"


def test_sample_documents_are_indexed(sample_store):
    documents = sample_store.documents()
    assert len(documents) == 32
    assert sample_store.chunk_count() == sum(doc.chunk_count for doc in documents)
    falcon = next(doc for doc in documents if doc.file_name.startswith("falcon"))
    assert falcon.flagged_chunks >= 1
    assert falcon.doc_id is None


def test_superseded_versions_are_detected(sample_store):
    status = {doc.file_name: doc for doc in sample_store.documents()}
    assert status[DOA_V1].status == "superseded"
    assert "marked 'superseded'" in status[DOA_V1].status_reason
    # The 2021 password policy still says "Current", but a newer version exists.
    old_password = status["NX-IT-002_password-policy_2021.txt"]
    assert old_password.status == "superseded"
    assert "version 2.0" in old_password.status_reason
    assert (
        status[DOA_V2].status
        == status["NX-IT-002_password-and-account-security_v2.0.txt"].status
        == "current"
    )


def test_superseded_documents_are_excluded_from_retrieval(sample_store, embedder):
    retriever = SemanticRetriever(sample_store, embedder)
    question = "approval matrix Plant Manager Managing Director Group CEO purchase order value"
    default = {hit.chunk.file_name for hit in retriever.search(question, 10)}
    assert DOA_V1 not in default
    included = {
        hit.chunk.file_name for hit in retriever.search(question, 10, include_superseded=True)
    }
    assert DOA_V1 in included


def test_bm25_and_hybrid_find_the_card_limits(sample_store, embedder, bm25):
    question = "What is the single transaction limit of the purchasing card (P-card)?"
    for retriever in (bm25, HybridRetriever([SemanticRetriever(sample_store, embedder), bm25])):
        hits = retriever.search(question, 5)
        assert any(hit.chunk.section == "3. Card limits" for hit in hits), retriever.name
        assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_bm25_excludes_superseded_documents(bm25):
    hits = bm25.search("Plant Manager Managing Director Group CEO", 10)
    assert all(hit.chunk.file_name != DOA_V1 for hit in hits)


def test_semantic_scores_are_cosine_similarities(sample_store, embedder):
    hits = SemanticRetriever(sample_store, embedder).search("password length characters", 3)
    assert all(-1.0 <= hit.score <= 1.0 for hit in hits)
    assert hits == sorted(hits, key=lambda hit: -hit.score)


def test_reingesting_the_same_content_is_a_no_op(ingestion):
    data = (SAMPLE_DIR / DOA_V2).read_bytes()
    first = ingestion.ingest_bytes(DOA_V2, data)
    second = ingestion.ingest_bytes("copy-with-another-name.pdf", data)
    assert (first.status, second.status) == (INDEXED, DUPLICATE)
    assert len(ingestion.store.documents()) == 1


def test_new_content_under_the_same_name_replaces_the_old_version(ingestion):
    first = ingestion.ingest_bytes("notes.txt", b"Version one of the canteen notes: open at 11:30.")
    second = ingestion.ingest_bytes(
        "notes.txt", b"Version two of the canteen notes: open at 12:00."
    )
    assert second.status == REPLACED
    documents = ingestion.store.documents()
    assert [doc.doc_key for doc in documents] == [second.doc_key]
    assert ingestion.storage.load(first.doc_key) is None
    assert ingestion.storage.load(second.doc_key)[1].startswith(b"Version two")


def test_rejected_files_are_reported_not_raised(ingestion):
    result = ingestion.ingest_bytes("empty.txt", (FIXTURES / "empty.txt").read_bytes())
    assert result.status == REJECTED
    assert result.error_code == "empty_file"
    assert ingestion.store.chunk_count() == 0


def test_injection_like_passages_are_flagged(ingestion):
    result = ingestion.ingest_bytes(
        "falcon.pdf", (SAMPLE_DIR / "falcon-logistics_service-overview.pdf").read_bytes()
    )
    assert result.flagged_chunks >= 1
    assert result.warnings
    assert any(chunk.flagged for chunk in ingestion.store.chunks())


def test_delete_and_rebuild(tmp_path, embedder):
    store = new_store(embedder)
    service = IngestionService(
        store, embedder, make_settings(tmp_path), FileStorage(tmp_path / "files")
    )
    a = service.ingest_bytes("a.txt", b"Alpha procedure: the alpha form is signed by the manager.")
    service.ingest_bytes("b.txt", b"Beta procedure: the beta form is signed by the director.")
    service.delete(a.doc_key)
    assert [doc.file_name for doc in store.documents()] == ["b.txt"]
    results = service.rebuild()
    assert [result.file_name for result in results] == ["b.txt"]
    assert store.chunk_count() == 1


def test_persistent_store_survives_a_restart(tmp_path, embedder):
    first = ChromaStore.persistent(
        tmp_path / "index", "docs", embedder.signature, embedder.dimensions
    )
    IngestionService(first, embedder, make_settings(tmp_path)).ingest_bytes(
        "a.txt", b"The canteen opens at 11:30 every working day."
    )
    reopened = ChromaStore.persistent(
        tmp_path / "index", "docs", embedder.signature, embedder.dimensions
    )
    assert [doc.file_name for doc in reopened.documents()] == ["a.txt"]


def test_an_index_built_with_another_model_is_refused(tmp_path):
    ChromaStore.persistent(tmp_path, "docs", "local-hash-512", 512)
    with pytest.raises(IndexMismatchError, match="Rebuild"):
        ChromaStore.persistent(tmp_path, "docs", "gemini-embedding-001-768", 768)


def test_copy_to_gives_an_independent_copy(sample_store, embedder):
    copy = new_store(embedder)
    sample_store.copy_to(copy)
    assert copy.chunk_count() == sample_store.chunk_count()
    IngestionService(copy, embedder, make_settings(), MemoryStorage()).ingest_bytes(
        "extra.txt", b"An extra document that only exists in the copy of the index."
    )
    assert copy.chunk_count() == sample_store.chunk_count() + 1
    assert len(sample_store.documents()) == 32


def test_empty_store_returns_no_hits(empty_store):
    assert SemanticRetriever(empty_store, LocalHashEmbedder()).search("anything", 5) == []
    assert BM25Retriever(empty_store).search("anything", 5) == []
