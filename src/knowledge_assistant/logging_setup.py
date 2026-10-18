"""Structured logs: one JSON object per event.

Logs record what happened (status, sources, timings, token counts) but not
the content of the questions, answers or documents, which may be
confidential. Set LOG_CONTENT=true to include questions and answers when
debugging locally.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log an event with structured fields."""
    logger.log(level, event, extra={"fields": {"event": event, **fields}})


def configure_logging(level: str = "INFO") -> None:
    """Send JSON logs to standard error. Safe to call several times."""
    root = logging.getLogger("knowledge_assistant")
    root.setLevel(level)
    if not any(getattr(handler, "_knowledge_assistant", False) for handler in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        handler._knowledge_assistant = True  # type: ignore[attr-defined]
        root.addHandler(handler)
        root.propagate = False
