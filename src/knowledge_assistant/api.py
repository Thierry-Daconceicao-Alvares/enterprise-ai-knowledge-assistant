"""HTTP API. Run with `uvicorn knowledge_assistant.api:app`.

There is no authentication in this version: run the API on a trusted network
or behind an authenticating gateway (see the security section of the README).
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .errors import (
    AssistantError,
    ConfigurationError,
    IndexMismatchError,
    InvalidQuestionError,
    LLMRequestError,
    LLMResponseError,
    LLMUnavailableError,
)
from .factory import Components, build_components
from .logging_setup import configure_logging

app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    version=__version__,
    description=(
        "Answers questions from the indexed documents of Nexora Industries (a fictional "
        "company), with the passages, pages and sections it relies on."
    ),
)

_components: Components | None = None
_lock = threading.Lock()


def get_components() -> Components:
    """Built on the first request and then reused; a failed build is retried
    at the next request."""
    global _components
    if _components is None:
        with _lock:
            if _components is None:
                settings = Settings.from_env()
                configure_logging(settings.log_level)
                _components = build_components(settings)
    return _components


ERROR_STATUS: dict[type[AssistantError], int] = {
    InvalidQuestionError: 422,
    ConfigurationError: 503,
    IndexMismatchError: 503,
    LLMUnavailableError: 503,
    LLMResponseError: 502,
    LLMRequestError: 500,
}


async def _handle_error(_: Request, exc: Exception) -> JSONResponse:
    status = next((code for error, code in ERROR_STATUS.items() if isinstance(exc, error)), 500)
    code = getattr(exc, "code", "error")
    return JSONResponse(status_code=status, content={"error": code, "detail": str(exc)})


app.add_exception_handler(AssistantError, _handle_error)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=5000,
        examples=["What is the procedure to approve a purchase request of more than €50,000?"],
    )
    include_superseded: bool = False


ComponentsDep = Annotated[Components, Depends(get_components)]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/documents")
def documents(components: ComponentsDep) -> dict:
    docs = components.store.documents()
    return {
        "documents": [asdict(doc) for doc in docs],
        "passages": components.store.chunk_count(),
    }


@app.post("/documents")
async def upload(components: ComponentsDep, file: Annotated[UploadFile, File()]) -> JSONResponse:
    limit = components.settings.max_file_bytes
    data = await file.read(limit + 1)  # never read more than the limit
    result = components.ingestion.ingest_bytes(file.filename or "upload", data)
    status = {"indexed": 201, "replaced": 201, "duplicate": 200}.get(result.status, 422)
    return JSONResponse(status_code=status, content=asdict(result))


@app.post("/ask")
def ask(request: AskRequest, components: ComponentsDep) -> dict:
    answer = components.assistant.ask(request.question, request.include_superseded)
    return answer.to_dict()
