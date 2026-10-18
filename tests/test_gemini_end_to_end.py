"""End-to-end tests through the real Gemini SDK, against a local fake API server."""

from dataclasses import replace

import pytest
from conftest import SAMPLE_DIR, make_settings
from fake_gemini_server import FakeGemini

from knowledge_assistant.embeddings import GeminiEmbedder
from knowledge_assistant.errors import LLMRequestError, LLMUnavailableError
from knowledge_assistant.factory import build_components
from knowledge_assistant.ingestion import MemoryStorage


@pytest.fixture
def fake_api():
    with FakeGemini() as server:
        yield server


def _components(fake_api, tmp_path, **overrides):
    options = {
        "embedding_provider": "gemini",
        "gemini_api_key": "test-key",
        "gemini_base_url": fake_api.url,
        "max_retries": 1,
        "request_timeout_s": 10,
        **overrides,
    }
    settings = replace(make_settings(tmp_path), **options)
    return build_components(settings, storage=MemoryStorage())


def test_ingest_and_answer_through_the_sdk(fake_api, tmp_path):
    components = _components(fake_api, tmp_path)
    assert isinstance(components.embedder, GeminiEmbedder)
    result = components.ingestion.ingest_paths(
        [SAMPLE_DIR / "NX-PUR-004_purchasing-card-policy.docx"]
    )[0]
    assert result.status == "indexed"

    answer = components.assistant.ask("What is the single transaction limit of the P-card?")
    assert answer.status == "answered"
    assert answer.citations and answer.citations[0].file_name.startswith("NX-PUR-004")
    assert answer.usage.prompt_tokens > 0
    assert answer.model == "gemini-3.5-flash-lite"

    paths = [path for path, _ in fake_api.requests]
    assert any(path.endswith("gemini-embedding-001:batchEmbedContents") for path in paths)
    assert any(path.endswith("gemini-3.5-flash-lite:generateContent") for path in paths)
    generate = next(body for path, body in fake_api.requests if path.endswith(":generateContent"))
    config = generate["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert "temperature" not in config  # Gemini 3 defaults are kept
    assert "statements" in str(config.get("responseSchema") or config.get("responseJsonSchema"))


def test_embedding_task_types_and_dimensions(fake_api, tmp_path):
    components = _components(fake_api, tmp_path)
    components.ingestion.ingest_bytes("a.txt", b"The canteen opens at 11:30 every working day.")
    components.assistant.retriever.search("When does the canteen open?", 3)
    batch = next(body for path, body in fake_api.requests if path.endswith(":batchEmbedContents"))
    task_types = {request.get("taskType") for request in batch["requests"]}
    assert task_types == {"RETRIEVAL_DOCUMENT"}
    assert {request.get("outputDimensionality") for request in batch["requests"]} == {768}
    query = [body for path, body in fake_api.requests if "embed" in path.lower()][-1]
    assert "RETRIEVAL_QUERY" in str(query)


def test_temporary_errors_are_retried(fake_api, tmp_path):
    components = _components(fake_api, tmp_path)
    fake_api.fail_next = [503]
    result = components.ingestion.ingest_bytes(
        "a.txt", b"The canteen opens at 11:30 every working day."
    )
    assert result.status == "indexed"


def test_rate_limit_after_retries_is_a_temporary_error(fake_api, tmp_path):
    components = _components(fake_api, tmp_path, max_retries=0)
    fake_api.fail_next = [429]
    with pytest.raises(LLMUnavailableError, match="429"):
        components.ingestion.ingest_bytes("a.txt", b"The canteen opens at 11:30 every working day.")


def test_rejected_key_is_a_configuration_error(fake_api, tmp_path):
    components = _components(fake_api, tmp_path, max_retries=0)
    fake_api.fail_next = [400]
    with pytest.raises(LLMRequestError, match="API key"):
        components.ingestion.ingest_bytes("a.txt", b"The canteen opens at 11:30 every working day.")
