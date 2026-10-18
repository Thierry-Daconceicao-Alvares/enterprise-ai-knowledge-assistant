"""Answer generation with a language model, constrained to a JSON schema.

Asking for structured output makes the answer machine-checkable: each
statement carries its support type and the numbers of the excerpts it relies
on, which the application then verifies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Protocol

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from .errors import LLMResponseError
from .gemini import gemini_errors

ANSWERED = "answered"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CLARIFICATION_NEEDED = "clarification_needed"


class Statement(BaseModel):
    text: str = Field(description="One sentence of the answer, without excerpt numbers.")
    support: Literal["explicit", "inferred"] = Field(
        description=(
            "'explicit' when an excerpt states it directly; 'inferred' when it follows "
            "from combining or applying excerpts."
        )
    )
    sources: list[int] = Field(description="Numbers of the excerpts that support the sentence.")


class AnswerPayload(BaseModel):
    """The structure the model must return."""

    status: Literal["answered", "insufficient_evidence", "clarification_needed"] = Field(
        description=(
            "'answered' when the excerpts answer the question; 'insufficient_evidence' when "
            "they do not; 'clarification_needed' when the question is too vague."
        )
    )
    statements: list[Statement] = Field(description="The answer, one statement per sentence.")
    note: str = Field(
        description=(
            "What is missing (insufficient_evidence), the clarifying question "
            "(clarification_needed), or a detail the excerpts do not cover (answered). "
            "Empty when there is nothing to add."
        )
    )


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens + self.thinking_tokens


@dataclass(frozen=True)
class GenerationResult:
    payload: AnswerPayload
    usage: TokenUsage
    model: str
    latency_ms: float


class AnswerGenerator(Protocol):
    """Anything that can turn a prompt into an AnswerPayload."""

    model: str

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult: ...


def _usage(response: types.GenerateContentResponse) -> TokenUsage:
    metadata = response.usage_metadata
    if metadata is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=metadata.prompt_token_count or 0,
        output_tokens=metadata.candidates_token_count or 0,
        thinking_tokens=metadata.thoughts_token_count or 0,
    )


def _parse_payload(response: types.GenerateContentResponse) -> AnswerPayload:
    if isinstance(response.parsed, AnswerPayload):
        return response.parsed
    if not response.candidates:
        feedback = response.prompt_feedback
        reason = feedback.block_reason if feedback else None
        raise LLMResponseError(f"The model returned no answer (block reason: {reason}).")
    finish_reason = response.candidates[0].finish_reason
    text = response.text
    if not text:
        raise LLMResponseError(f"The model returned an empty answer ({finish_reason}).")
    try:
        return AnswerPayload.model_validate_json(text)
    except ValidationError as exc:
        raise LLMResponseError(
            f"The answer does not match the expected JSON schema ({finish_reason})."
        ) from exc


class GeminiGenerator:
    """Generate grounded answers with a Gemini model.

    Sampling parameters and the thinking level are only sent when they are set:
    for Gemini 3 models Google recommends keeping the default temperature, and
    the output length is not capped, so that the answer is not cut short when
    the model spends tokens on reasoning.
    """

    def __init__(
        self,
        client: genai.Client,
        model: str,
        temperature: float | None = None,
        thinking_level: str | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self._temperature = temperature
        self._thinking_level = thinking_level

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        thinking = (
            types.ThinkingConfig(thinking_level=self._thinking_level.upper())
            if self._thinking_level
            else None
        )
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
            response_mime_type="application/json",
            response_schema=AnswerPayload,
            thinking_config=thinking,
        )
        start = time.perf_counter()
        with gemini_errors("generation"):
            response = self._client.models.generate_content(
                model=self.model, contents=user_prompt, config=config
            )
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResult(
            payload=_parse_payload(response),
            usage=_usage(response),
            model=response.model_version or self.model,
            latency_ms=latency_ms,
        )
