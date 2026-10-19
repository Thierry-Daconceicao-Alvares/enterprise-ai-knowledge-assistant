"""Gemini API client and translation of its errors into the assistant's own errors."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as clock_time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .config import Settings
from .errors import (
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMRequestError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

# Retried by the SDK with exponential backoff. HTTP 429 is retried by `call`
# instead, which can tell a per-minute rate limit (worth waiting for) from a
# used-up daily quota (not worth waiting for).
RETRYABLE_STATUS_CODES = [408, 500, 502, 503, 504]
MAX_RATE_LIMIT_WAIT_S = 30.0

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

_sleep = time.sleep  # replaced in the tests


def make_client(settings: Settings) -> genai.Client:
    """Gemini client with a request timeout and automatic retries with
    exponential backoff on timeouts and server errors."""
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


# ------------------------------------------------------------------------ quotas


@dataclass(frozen=True)
class QuotaViolation:
    """What an HTTP 429 response of the Gemini API says about the exceeded quota.

    The response lists the violated quotas, such as
    `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, and how long to wait.
    For a daily quota, that wait is not reliable: the quota only comes back
    when it resets.
    """

    quota_id: str = ""
    limit: str = ""
    model: str = ""
    retry_after_s: float | None = None

    @property
    def daily(self) -> bool:
        return "PerDay" in self.quota_id


def _seconds(duration: object) -> float | None:
    """A protobuf Duration in JSON, such as "36s" or "1.5s", in seconds."""
    if isinstance(duration, str) and duration.endswith("s"):
        try:
            return float(duration[:-1])
        except ValueError:
            return None
    return None


def parse_quota_violation(details: object) -> QuotaViolation:
    """Read the QuotaFailure and RetryInfo details of a 429 error response.
    A daily quota is reported first when several quotas are exceeded."""
    error = details.get("error", details) if isinstance(details, dict) else {}
    items = error.get("details") if isinstance(error, dict) else None
    violations: list[dict] = []
    retry_after_s = None
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("@type", ""))
        if kind.endswith("QuotaFailure"):
            violations += [v for v in item.get("violations") or [] if isinstance(v, dict)]
        elif kind.endswith("RetryInfo"):
            retry_after_s = _seconds(item.get("retryDelay"))
    violations.sort(key=lambda v: "PerDay" not in str(v.get("quotaId", "")))
    if not violations:
        return QuotaViolation(retry_after_s=retry_after_s)
    first = violations[0]
    dimensions = first.get("quotaDimensions")
    return QuotaViolation(
        quota_id=str(first.get("quotaId", "")),
        limit=str(first.get("quotaValue", "")),
        model=str(dimensions.get("model", "")) if isinstance(dimensions, dict) else "",
        retry_after_s=retry_after_s,
    )


def next_daily_reset(now: datetime | None = None) -> datetime | None:
    """When the daily quotas of the Gemini API reset: at midnight Pacific time.
    None when the system has no time zone database."""
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:  # pragma: no cover - depends on the environment
        return None
    local = (now or datetime.now(UTC)).astimezone(pacific)
    midnight = datetime.combine(local.date() + timedelta(days=1), clock_time(0), tzinfo=pacific)
    return midnight.astimezone(UTC)


def _rate_limit_error(operation: str, details: object) -> LLMUnavailableError:
    violation = parse_quota_violation(details)
    if violation.daily:
        resets_at = next_daily_reset()
        model = f" for {violation.model}" if violation.model else ""
        limit = f" (limit {violation.limit} per day)" if violation.limit else ""
        when = f"at {resets_at:%H:%M} UTC" if resets_at else "at midnight Pacific time"
        return LLMQuotaExhaustedError(
            f"Gemini {operation}: the daily quota of this API key{model} is used up{limit}. "
            f"It resets {when}. To continue before then, use a key from another Google Cloud "
            "project or enable billing.",
            resets_at=resets_at,
        )
    quota = f" ({violation.quota_id})" if violation.quota_id else ""
    wait = (
        f"Retry in {violation.retry_after_s:.0f} s."
        if violation.retry_after_s is not None
        else "Wait a minute and retry."
    )
    return LLMRateLimitError(
        f"Gemini {operation}: rate limit reached{quota} (HTTP 429). {wait}",
        retry_after_s=violation.retry_after_s,
    )


# ------------------------------------------------------------------------ errors


@contextmanager
def gemini_errors(operation: str) -> Iterator[None]:
    """Translate SDK and network exceptions into LLMError subclasses with a
    message that says what to do."""
    try:
        yield
    except genai_errors.ClientError as exc:
        detail = getattr(exc, "message", "") or str(exc)
        if exc.code == 429:
            raise _rate_limit_error(operation, exc.details) from exc
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


def call[T](operation: str, request: Callable[[], T], max_retries: int) -> T:
    """Send a request and translate its errors.

    After a per-minute rate limit, wait as long as the API asks (at most
    MAX_RATE_LIMIT_WAIT_S) and send it again, up to `max_retries` times. A
    used-up daily quota fails at once: waiting would not help before it resets.
    """
    attempt = 0
    while True:
        try:
            with gemini_errors(operation):
                return request()
        except LLMRateLimitError as exc:
            if attempt >= max_retries:
                raise
            asked = exc.retry_after_s if exc.retry_after_s is not None else 2.0 * 2**attempt
            wait = min(max(asked, 1.0), MAX_RATE_LIMIT_WAIT_S)
            attempt += 1
            logger.warning("%s Waiting %.0f s (retry %d/%d).", exc, wait, attempt, max_retries)
            _sleep(wait)
