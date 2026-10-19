"""End-to-end tests through the real Gemini SDK, against a local fake API server."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from conftest import SAMPLE_DIR, make_settings
from fake_gemini_server import FakeGemini, quota_error

from knowledge_assistant import gemini
from knowledge_assistant.embeddings import GeminiEmbedder
from knowledge_assistant.errors import (
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMUnavailableError,
)
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


def test_a_per_minute_rate_limit_is_retried_after_the_delay_the_api_asks(
    fake_api, tmp_path, monkeypatch
):
    waits: list[float] = []
    monkeypatch.setattr(gemini, "_sleep", waits.append)
    components = _components(fake_api, tmp_path, max_retries=2)
    fake_api.fail_next = [
        quota_error("EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier", retry_delay="7s")
    ]
    result = components.ingestion.ingest_bytes(
        "a.txt", b"The canteen opens at 11:30 every working day."
    )
    assert result.status == "indexed"
    assert waits == [7.0]


def test_a_used_up_daily_quota_fails_at_once_and_says_when_it_resets(
    fake_api, tmp_path, monkeypatch
):
    """Observed in CI: retrying a used-up daily quota only delayed the failure."""
    waits: list[float] = []
    monkeypatch.setattr(gemini, "_sleep", waits.append)
    components = _components(fake_api, tmp_path, max_retries=3)
    fake_api.fail_next = [
        quota_error("EmbedContentRequestsPerDayPerProjectPerModel-FreeTier", retry_delay="36s")
    ] * 4
    with pytest.raises(LLMQuotaExhaustedError) as caught:
        components.ingestion.ingest_bytes("a.txt", b"The canteen opens at 11:30 every working day.")
    assert len(fake_api.requests) == 1 and waits == []  # no retry
    message = str(caught.value)
    assert "daily quota" in message and "gemini-embedding-001" in message
    assert "limit 1000 per day" in message and "UTC" in message
    assert caught.value.resets_at is not None and caught.value.resets_at > datetime.now(UTC)


def test_a_rate_limit_that_persists_is_reported_with_its_quota(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(gemini, "_sleep", lambda _: None)
    components = _components(fake_api, tmp_path, max_retries=1)
    fake_api.fail_next = [
        quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", retry_delay="20s")
    ] * 2
    with pytest.raises(LLMRateLimitError, match=r"PerMinute.*Retry in 20 s") as caught:
        components.ingestion.ingest_bytes("a.txt", b"The canteen opens at 11:30 every working day.")
    assert caught.value.retry_after_s == 20.0
    assert len(fake_api.requests) == 2


def test_rejected_key_is_a_configuration_error(fake_api, tmp_path):
    components = _components(fake_api, tmp_path, max_retries=0)
    fake_api.fail_next = [400]
    with pytest.raises(LLMRequestError, match="API key"):
        components.ingestion.ingest_bytes("a.txt", b"The canteen opens at 11:30 every working day.")


def test_a_safety_block_is_reported_clearly(fake_api, tmp_path):
    """Observed in the evaluation: the provider refused a question about a phishing incident."""
    components = _components(fake_api, tmp_path)
    components.ingestion.ingest_bytes("a.txt", b"Report phishing with the Report phishing button.")
    fake_api.block_next = True
    with pytest.raises(LLMResponseError, match="safety filter refused this request"):
        components.assistant.ask("What should I do after clicking a phishing link?")


def test_daily_quotas_reset_at_midnight_pacific_time():
    summer = datetime(2026, 10, 19, 11, 57, tzinfo=UTC)
    assert gemini.next_daily_reset(summer) == datetime(2026, 10, 20, 7, 0, tzinfo=UTC)
    winter = datetime(2026, 12, 1, 12, 0, tzinfo=UTC)
    assert gemini.next_daily_reset(winter) == datetime(2026, 12, 2, 8, 0, tzinfo=UTC)
    just_after = datetime(2026, 10, 20, 7, 30, tzinfo=UTC)
    assert gemini.next_daily_reset(just_after) == datetime(2026, 10, 21, 7, 0, tzinfo=UTC)


def test_the_daily_quota_is_reported_when_several_quotas_are_exceeded():
    _, body = quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", limit="15")
    body["error"]["details"][0]["violations"].append(
        {
            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            "quotaDimensions": {"model": "gemini-3.5-flash-lite"},
            "quotaValue": "20",
        }
    )
    violation = gemini.parse_quota_violation(body)
    assert violation.daily and violation.limit == "20"
    assert violation.model == "gemini-3.5-flash-lite" and violation.retry_after_s == 1.0
    assert gemini.parse_quota_violation({"error": {"code": 429}}) == gemini.QuotaViolation()
    assert gemini.parse_quota_violation("not JSON") == gemini.QuotaViolation()
