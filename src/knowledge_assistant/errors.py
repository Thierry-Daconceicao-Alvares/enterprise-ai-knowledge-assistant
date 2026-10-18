"""Errors raised by the assistant.

Each error has a stable `code`, used by the API responses, the user interface
and the evaluation, and a message written for the person who has to act on it.
"""

from __future__ import annotations


class AssistantError(Exception):
    """Base class for all errors of the application."""

    code = "error"


class ConfigurationError(AssistantError):
    """A setting is missing or invalid, for example the API key."""

    code = "configuration_error"


# ----------------------------------------------------------------------- documents


class DocumentError(AssistantError):
    """An uploaded file cannot be ingested."""

    code = "document_error"


class UnsupportedFileTypeError(DocumentError):
    code = "unsupported_file_type"


class FileTooLargeError(DocumentError):
    code = "file_too_large"


class EmptyFileError(DocumentError):
    code = "empty_file"


class MalformedDocumentError(DocumentError):
    """The file is corrupted, protected by a password, or not what its extension says."""

    code = "malformed_document"


class NoExtractableTextError(DocumentError):
    """The file is valid but contains no text, for example a scanned PDF."""

    code = "no_extractable_text"


# ------------------------------------------------------------------- questions/index


class InvalidQuestionError(AssistantError):
    """The question is empty, too short, too long or contains no words."""

    code = "invalid_question"


class IndexMismatchError(AssistantError):
    """The index was built with another embedding model and must be rebuilt."""

    code = "index_mismatch"


# --------------------------------------------------------------------------- models


class LLMError(AssistantError):
    """Base class for errors raised while calling the model API."""

    code = "llm_error"


class LLMUnavailableError(LLMError):
    """Temporary failure: timeout, network error, rate limit or server error."""

    code = "llm_unavailable"


class LLMRequestError(LLMError):
    """The request was rejected (invalid key, unknown model, bad request).
    Retrying will not help until the configuration is fixed."""

    code = "llm_request_rejected"


class LLMResponseError(LLMError):
    """The model answered, but the answer cannot be used (blocked, truncated or
    not matching the expected JSON schema)."""

    code = "llm_invalid_response"
