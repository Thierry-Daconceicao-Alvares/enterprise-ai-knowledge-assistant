"""Checks applied before and after the model call.

None of these checks is a guarantee on its own; together with the prompt
rules and the evaluation they reduce the risk of misleading answers:

- the question is validated before any processing;
- passages that look like instructions aimed at an AI system are flagged at
  ingestion and labelled as untrusted in the prompt;
- every statement of the answer must cite at least one excerpt that was
  actually given to the model, otherwise it is removed;
- numbers in a statement must appear in its cited excerpts (or the question),
  otherwise the statement is flagged;
- a secret marker ("canary") in the system prompt reveals answers that leak it.
"""

from __future__ import annotations

import re
import secrets
import unicodedata
from collections.abc import Iterable

from .errors import InvalidQuestionError
from .text import numbers_in

MIN_QUESTION_CHARS = 3

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏ -‮⁠-⁯]")

_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier|preceding)\s+(instructions|rules|prompts?)",
    r"\bdisregard\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\b",
    r"\b(reveal|print|show|repeat|output|display)\s+(me\s+)?(your|the)\s+(full\s+|complete\s+|entire\s+)?(system\s+)?(prompt|instructions)",
    r"\bsystem\s+prompt\b",
    r"\binstructions?\s+for\s+(ai|llm|language\s+model|chatbot)s?\b",
    r"\b(note|message)\s+to\s+(ai|llm)\s+assistants?\b",
    r"\byou\s+are\s+now\s+(in\s+)?(a|an|developer|admin|dan)\b",
    r"\bdeveloper\s+mode\b",
    r"\bnew\s+instructions\s*:",
]
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def validate_question(raw: object, max_chars: int = 1000) -> str:
    """Return the cleaned question, or raise InvalidQuestionError with a
    message that can be shown to the user."""
    if not isinstance(raw, str):
        raise InvalidQuestionError("The question must be text.")
    text = _CONTROL.sub(" ", unicodedata.normalize("NFC", raw))
    text = " ".join(text.split())
    if len(text) < MIN_QUESTION_CHARS:
        raise InvalidQuestionError("Please type a question (at least 3 characters).")
    if len(text) > max_chars:
        raise InvalidQuestionError(
            f"The question is {len(text)} characters long; the limit is {max_chars}."
        )
    if sum(char.isalpha() for char in text) < 2:
        raise InvalidQuestionError("The question must contain words.")
    return text


def looks_like_injection(text: str) -> bool:
    """True when a passage contains phrases typical of prompt-injection
    attempts. This is a heuristic: it flags known patterns and cannot catch
    every rephrasing, so flagged passages are labelled, not trusted."""
    return bool(_INJECTION.search(text))


def make_canary() -> str:
    """A random marker placed in the system prompt; it must never appear in
    an answer."""
    return f"NXC-{secrets.token_hex(6).upper()}"


def contains_canary(texts: Iterable[str], canary: str) -> bool:
    return any(canary.lower() in text.lower() for text in texts)


def unverified_numbers(statement: str, evidence: Iterable[str], question: str) -> list[str]:
    """Numbers stated in the answer that appear neither in the cited excerpts
    nor in the question (for example an invented amount or deadline)."""
    allowed = numbers_in(question)
    for text in evidence:
        allowed |= numbers_in(text)
    missing = sorted(numbers_in(statement) - allowed)
    return [f"{value:g}" for value in missing]
