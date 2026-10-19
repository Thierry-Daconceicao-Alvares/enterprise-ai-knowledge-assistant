"""Interface tests with Streamlit's AppTest, against the fake Gemini API."""

import pytest
from conftest import PROJECT_ROOT
from fake_gemini_server import FakeGemini, quota_error
from streamlit.testing.v1 import AppTest

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")


@pytest.fixture
def fake_api():
    with FakeGemini() as server:
        yield server


@pytest.fixture
def app_env(monkeypatch, tmp_path, fake_api):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_BASE_URL", fake_api.url)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_RETRIES", "0")
    sample = tmp_path / "sample_documents"
    sample.mkdir()
    for name in [
        "NX-PUR-004_purchasing-card-policy.docx",
        "NX-PUR-002_delegation-of-authority_v2.0.pdf",
    ]:
        source = PROJECT_ROOT / "data" / "sample_documents" / name
        (sample / name).write_bytes(source.read_bytes())
    return tmp_path


def _text(app: AppTest) -> str:
    parts = [element.value for element in app.markdown] + [element.value for element in app.caption]
    return "\n".join(str(part) for part in parts)


def test_missing_key_shows_setup_instructions(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert any("GEMINI_API_KEY" in error.value for error in app.error)
    assert "aistudio.google.com" in _text(app)


def test_load_samples_and_ask_a_question(app_env):
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert any("No document is indexed yet" in info.value for info in app.info)

    load = next(button for button in app.sidebar.button if "sample documents" in button.label)
    load.click().run()
    assert not app.exception
    assert app.sidebar.metric[0].value == "2"

    app.chat_input[0].set_value("What is the single transaction limit of the P-card?").run()
    assert not app.exception
    assert app.session_state.history[-1]["answer"].status == "answered"
    assert "Sources" in _text(app)
    assert any("relevance" in expander.label for expander in app.expander)


def test_invalid_question_shows_a_warning(app_env):
    app = AppTest.from_file(APP, default_timeout=60).run()
    next(b for b in app.sidebar.button if "sample documents" in b.label).click().run()
    app.chat_input[0].set_value("?!?! ...").run()
    assert not app.exception
    assert any("words" in warning.value for warning in app.warning)


def test_public_demo_mode_isolates_sessions(app_env, monkeypatch):
    monkeypatch.setenv("PUBLIC_DEMO", "true")
    first = AppTest.from_file(APP, default_timeout=60).run()
    assert not first.exception
    assert any("Public demo" in warning.value for warning in first.warning)
    assert first.sidebar.metric[0].value == "2"  # sample documents preloaded
    second = AppTest.from_file(APP, default_timeout=60).run()
    assert first.session_state.index_name != second.session_state.index_name


def test_public_demo_explains_a_used_up_daily_quota(app_env, monkeypatch, fake_api):
    monkeypatch.setenv("PUBLIC_DEMO", "true")
    # In-memory Chroma collections live as long as the process: use a new one.
    monkeypatch.setenv("COLLECTION_NAME", "quota-test")
    fake_api.fail_next = [quota_error("EmbedContentRequestsPerDayPerProjectPerModel-FreeTier")]
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    message = " ".join(error.value for error in app.error)
    assert "public demo has used up the daily quota" in message and "UTC" in message
    assert "github.com" in _text(app)  # where to see the screenshots meanwhile
