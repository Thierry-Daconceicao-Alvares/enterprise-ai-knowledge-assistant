"""Runtime settings, read from environment variables and an optional `.env` file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigurationError


def _find_project_root() -> Path:
    """The repository root (the folder with pyproject.toml and src/), or the
    current directory when the package is installed elsewhere."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()

EMBEDDING_PROVIDERS = ("gemini", "local")
RETRIEVERS = ("semantic", "hybrid", "bm25")
THINKING_LEVELS = ("minimal", "low", "medium", "high")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _read(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _read_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = _read(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def _read_float(name: str, default: float | None, minimum: float, maximum: float) -> float | None:
    raw = _read(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}.") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def _read_bool(name: str, default: bool) -> bool:
    raw = _read(name)
    if raw is None:
        return default
    if raw.lower() in _TRUE:
        return True
    if raw.lower() in _FALSE:
        return False
    raise ConfigurationError(f"{name} must be true or false, got {raw!r}.")


def _read_choice(name: str, default: str | None, choices: tuple[str, ...]) -> str | None:
    raw = _read(name, default)
    if raw is None:
        return None
    value = raw.lower()
    if value not in choices:
        raise ConfigurationError(f"{name} must be one of {', '.join(choices)}, got {raw!r}.")
    return value


@dataclass(frozen=True)
class Settings:
    """All tunable parameters. Each field can be set with the environment
    variable of the same name in upper case, for example `CHUNK_SIZE=800`."""

    # Models
    gemini_api_key: str | None = None
    generation_model: str = "gemini-3.5-flash-lite"
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    # None keeps the model defaults, as Google recommends for Gemini 3 models.
    temperature: float | None = None
    thinking_level: str | None = None
    request_timeout_s: float = 60.0
    max_retries: int = 3
    gemini_base_url: str | None = None

    # Chunking and retrieval
    retriever: str = "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    # Semantic hits below this cosine similarity are not sent to the model.
    min_relevance: float = 0.0

    # Limits
    max_file_mb: float = 10.0
    max_pages: int = 300
    max_question_chars: int = 1000

    # Storage
    data_dir: Path = PROJECT_ROOT / "data"
    collection_name: str = "nexora"

    # Public demonstration mode: uploads stay in the visitor's session
    public_demo: bool = False
    demo_max_questions: int = 30
    demo_max_uploads: int = 3

    # Logging
    log_level: str = "INFO"
    log_content: bool = False

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def storage_dir(self) -> Path:
        return self.data_dir / "storage"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "sample_documents"

    @property
    def max_file_bytes(self) -> int:
        return int(self.max_file_mb * 1024 * 1024)

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        """Build settings from the environment. Values already set in the
        environment take precedence over the `.env` file."""
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
        settings = cls(
            gemini_api_key=_read("GEMINI_API_KEY"),
            generation_model=_read("GENERATION_MODEL", cls.generation_model),
            embedding_provider=_read_choice(
                "EMBEDDING_PROVIDER", cls.embedding_provider, EMBEDDING_PROVIDERS
            ),
            embedding_model=_read("EMBEDDING_MODEL", cls.embedding_model),
            embedding_dimensions=_read_int("EMBEDDING_DIMENSIONS", 768, 128, 3072),
            temperature=_read_float("TEMPERATURE", None, 0.0, 2.0),
            thinking_level=_read_choice("THINKING_LEVEL", None, THINKING_LEVELS),
            request_timeout_s=_read_float("REQUEST_TIMEOUT_S", 60.0, 1.0, 600.0),
            max_retries=_read_int("MAX_RETRIES", 3, 0, 10),
            gemini_base_url=_read("GEMINI_BASE_URL"),
            retriever=_read_choice("RETRIEVER", cls.retriever, RETRIEVERS),
            chunk_size=_read_int("CHUNK_SIZE", 1000, 200, 8000),
            chunk_overlap=_read_int("CHUNK_OVERLAP", 150, 0, 2000),
            top_k=_read_int("TOP_K", 5, 1, 20),
            min_relevance=_read_float("MIN_RELEVANCE", 0.0, -1.0, 1.0),
            max_file_mb=_read_float("MAX_FILE_MB", 10.0, 0.1, 200.0),
            max_pages=_read_int("MAX_PAGES", 300, 1, 5000),
            max_question_chars=_read_int("MAX_QUESTION_CHARS", 1000, 20, 10000),
            data_dir=Path(_read("DATA_DIR", str(cls.data_dir))),
            collection_name=_read("COLLECTION_NAME", cls.collection_name),
            public_demo=_read_bool("PUBLIC_DEMO", False),
            demo_max_questions=_read_int("DEMO_MAX_QUESTIONS", 30, 1, 10000),
            demo_max_uploads=_read_int("DEMO_MAX_UPLOADS", 3, 0, 100),
            log_level=(_read("LOG_LEVEL", "INFO") or "INFO").upper(),
            log_content=_read_bool("LOG_CONTENT", False),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.chunk_overlap * 2 > self.chunk_size:
            raise ConfigurationError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be at most half of "
                f"CHUNK_SIZE ({self.chunk_size})."
            )

    def require_api_key(self) -> str:
        """Return the Gemini API key, or explain how to set it."""
        if not self.gemini_api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Create a free key in Google AI Studio "
                "(https://aistudio.google.com/apikey) and add it to your environment "
                "or to a .env file (see .env.example)."
            )
        return self.gemini_api_key
