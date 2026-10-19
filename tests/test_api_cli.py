"""HTTP API and command-line interface."""

from datetime import UTC, datetime, timedelta

import pytest
from conftest import FIXTURES, SAMPLE_DIR, FakeGenerator, answered, make_settings, new_store
from fastapi.testclient import TestClient

from knowledge_assistant import api, cli
from knowledge_assistant.errors import (
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from knowledge_assistant.factory import build_components
from knowledge_assistant.ingestion import MemoryStorage


@pytest.fixture
def client(tmp_path, embedder):
    components = build_components(
        make_settings(tmp_path),
        store=new_store(embedder),
        storage=MemoryStorage(),
        embedder=embedder,
    )
    components.assistant.generator = FakeGenerator(answered(("The limit is €1,500.", [1])))
    api.app.dependency_overrides[api.get_components] = lambda: components
    yield TestClient(api.app), components
    api.app.dependency_overrides.clear()


def test_health(client):
    test_client, _ = client
    assert test_client.get("/health").json()["status"] == "ok"


def test_upload_list_and_ask(client):
    test_client, _ = client
    data = (SAMPLE_DIR / "NX-PUR-004_purchasing-card-policy.docx").read_bytes()
    response = test_client.post("/documents", files={"file": ("card.docx", data)})
    assert response.status_code == 201
    assert response.json()["status"] == "indexed"
    again = test_client.post("/documents", files={"file": ("card.docx", data)})
    assert again.status_code == 200 and again.json()["status"] == "duplicate"

    listing = test_client.get("/documents").json()
    assert [doc["file_name"] for doc in listing["documents"]] == ["card.docx"]

    answer = test_client.post("/ask", json={"question": "What is the P-card limit?"}).json()
    assert answer["status"] == "answered"
    assert answer["citations"][0]["file_name"] == "card.docx"
    assert answer["answer_text"].startswith("The limit is €1,500.")


def test_rejected_upload(client):
    test_client, _ = client
    response = test_client.post("/documents", files={"file": ("budget.xlsx", b"PK\x03\x04")})
    assert response.status_code == 422
    assert response.json()["error_code"] == "unsupported_file_type"


def test_api_errors_are_mapped(client):
    test_client, components = client
    components.ingestion.ingest_bytes("a.txt", b"The canteen opens at 11:30 every working day.")
    response = test_client.post("/ask", json={"question": "?!"})
    assert response.status_code == 422 and response.json()["error"] == "invalid_question"
    components.assistant.generator = FakeGenerator(LLMUnavailableError("Gemini is down"))
    response = test_client.post("/ask", json={"question": "When does the canteen open?"})
    assert response.status_code == 503 and response.json()["error"] == "llm_unavailable"


def test_quota_errors_are_http_429_with_retry_after(client):
    test_client, components = client
    components.ingestion.ingest_bytes("a.txt", b"The canteen opens at 11:30 every working day.")
    question = {"question": "When does the canteen open?"}

    components.assistant.generator = FakeGenerator(LLMRateLimitError("slow down", 12.4))
    response = test_client.post("/ask", json=question)
    assert response.status_code == 429 and response.json()["error"] == "llm_rate_limited"
    assert response.headers["Retry-After"] == "13"

    resets_at = datetime.now(UTC) + timedelta(hours=2)
    components.assistant.generator = FakeGenerator(LLMQuotaExhaustedError("used up", resets_at))
    response = test_client.post("/ask", json=question)
    assert response.status_code == 429 and response.json()["error"] == "llm_quota_exhausted"
    assert 7000 < int(response.headers["Retry-After"]) <= 7200


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    sample = tmp_path / "sample_documents"
    sample.mkdir()
    for name in ["NX-PUR-004_purchasing-card-policy.docx", "NX-IT-002_password-policy_2021.txt"]:
        (sample / name).write_bytes((SAMPLE_DIR / name).read_bytes())
    return tmp_path


def test_cli_ingest_documents_and_search(cli_env, capsys):
    assert cli.main(["ingest", "--samples"]) == 0
    out = capsys.readouterr().out
    assert "indexed" in out and "Index: 2 documents" in out
    assert cli.main(["documents"]) == 0
    assert "NX-PUR-004" in capsys.readouterr().out
    assert cli.main(["search", "purchasing card transaction limit", "-k", "2"]) == 0
    assert "NX-PUR-004_purchasing-card-policy.docx" in capsys.readouterr().out


def test_cli_rejected_file_sets_the_exit_code(cli_env, capsys):
    assert cli.main(["ingest", str(FIXTURES / "empty.txt")]) == 1
    assert "rejected" in capsys.readouterr().out


def test_cli_ask_without_a_key_explains_what_to_do(cli_env, capsys):
    cli.main(["ingest", "--samples"])
    assert cli.main(["ask", "What is the P-card limit?"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_cli_retrieval_evaluation(cli_env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "build_eval_index",
        lambda store, ingestion, folder: ingestion.ingest_folder(SAMPLE_DIR),
    )
    output = tmp_path / "report"
    code = cli.main(
        ["evaluate", "--retrieval-only", "--retrievers", "bm25,semantic", "--output", str(output)]
    )
    assert code == 0
    assert (output / "report.md").exists()
    assert "bm25" in capsys.readouterr().out


def test_cli_evaluation_stopped_by_the_daily_quota(cli_env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "build_eval_index",
        lambda store, ingestion, folder: ingestion.ingest_folder(SAMPLE_DIR),
    )
    build_components = cli.build_components

    def components_with_quota_error(*args, **kwargs):
        components = build_components(*args, **kwargs)
        components.assistant.generator = FakeGenerator(LLMQuotaExhaustedError("quota used up"))
        return components

    monkeypatch.setattr(cli, "build_components", components_with_quota_error)
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef")
    output = tmp_path / "report"
    arguments = ["--retrievers", "bm25", "--answer-retrievers", "bm25", "--output", str(output)]
    assert cli.main(["evaluate", *arguments]) == 1
    assert "Error (evaluation_incomplete)" in capsys.readouterr().err
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Run incomplete" in report and "Commit: 0123456789ab" in report
