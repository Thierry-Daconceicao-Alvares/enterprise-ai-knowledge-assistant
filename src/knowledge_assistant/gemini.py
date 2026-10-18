"""Gemini API client and translation of its errors into the assistant's own errors."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .config import Settings
from .errors import LLMRequestError, LLMUnavailableError

RETRYABLE_STATUS_CODES = [408, 429, 500, 502, 503, 504]

# The SDK uses httpx; accept the exceptions of its httpx2 fork as well when installed.
try:
    import httpx2

    _TIMEOUT_ERRORS: tuple[type[Exception], ...] = (
        httpx.TimeoutException,
        httpx2.TimeoutException,
    )
    _TRANSPORT_ERRORS: tuple[type[Exception], ...] = (httpx.TransportError, httpx2.TransportError)
except ImportError:  # pragma: no cover - depends on the environment
    _TIMEOUT_ERRORS = (httpx.TimeoutException,)
    _TRANSPORT_ERRORS = (httpx.TransportError,)


def make_client(settings: Settings) -> genai.Client:
    """Gemini client with a request timeout and automatic retries with
    exponential backoff on rate limits and server errors."""
    return genai.Client(
        api_key=settings.require_api_key(),
        http_options=types.HttpOptions(
            base_url=settings.gemini_base_url,
            timeout=int(settings.request_timeout_s * 1000),
            retry_options=types.HttpRetryOptions(
                attempts=settings.max_retries + 1,
                initial_delay=2.0,
                max_delay=30.0,
                exp_base=2.0,
                jitter=1.0,
                http_status_codes=RETRYABLE_STATUS_CODES,
            ),
        ),
    )


@contextmanager
def gemini_errors(operation: str) -> Iterator[None]:
    """Translate SDK and network exceptions into LLMError subclasses with a
    message that says what to do."""
    try:
        yield
    except genai_errors.ClientError as exc:
        detail = getattr(exc, "message", "") or str(exc)
        if exc.code == 429:
            raise LLMUnavailableError(
                f"Gemini {operation}: rate limit or quota exceeded (HTTP 429). "
                "Wait a minute and retry, or check your quota in Google AI Studio."
            ) from exc
        if exc.code in (401, 403) or "api key" in detail.lower():
            raise LLMRequestError(
                f"Gemini {operation}: the API key was rejected (HTTP {exc.code}). "
                "Check GEMINI_API_KEY."
            ) from exc
        if exc.code == 404:
            raise LLMRequestError(
                f"Gemini {operation}: model not found (HTTP 404). "
                "Check GENERATION_MODEL and EMBEDDING_MODEL."
            ) from exc
        raise LLMRequestError(
            f"Gemini {operation}: request rejected (HTTP {exc.code}): {detail}"
        ) from exc
    except genai_errors.ServerError as exc:
        raise LLMUnavailableError(
            f"Gemini {operation}: server error (HTTP {exc.code}). Retry later."
        ) from exc
    except _TIMEOUT_ERRORS as exc:
        raise LLMUnavailableError(f"Gemini {operation}: the request timed out.") from exc
    except _TRANSPORT_ERRORS as exc:
        raise LLMUnavailableError(
            f"Gemini {operation}: network error ({exc.__class__.__name__})."
        ) from exc
