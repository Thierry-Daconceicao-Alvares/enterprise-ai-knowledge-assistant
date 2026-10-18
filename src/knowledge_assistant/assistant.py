"""The question-answering pipeline: validate, retrieve, generate, then verify.

Every answer goes through the same steps:

1. The question is validated (length, content).
2. The most relevant passages are retrieved; superseded documents are excluded.
3. If no passage is relevant enough, the assistant abstains without calling the model.
4. The model answers from the numbered passages, as a list of statements.
5. Each statement is checked: it must cite passages that were given to the
   model, and its numbers must appear in those passages. Unsupported
   statements are removed; an answer left without support is withheld.
6. Sources are attached from the index metadata (file, page, section, score).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field

from .config import Settings
from .costs import embedding_cost, estimate_tokens, generation_cost
from .errors import ConfigurationError
from .guardrails import contains_canary, make_canary, unverified_numbers, validate_question
from .llm import (
    ANSWERED,
    CLARIFICATION_NEEDED,
    INSUFFICIENT_EVIDENCE,
    AnswerGenerator,
    AnswerPayload,
    TokenUsage,
)
from .logging_setup import log_event
from .prompts import build_user_prompt, system_prompt
from .retrieval import Retriever, SearchHit
from .vector_store import SUPERSEDED, ChromaStore

logger = logging.getLogger(__name__)

EMPTY_INDEX_MESSAGE = (
    "No documents are indexed yet. Upload documents or load the sample set, then ask again."
)
NO_RELEVANT_MESSAGE = (
    "No passage of the indexed documents is relevant enough to answer this question."
)
NOT_COVERED_MESSAGE = "The documents do not contain the information needed to answer this question."
UNSUPPORTED_MESSAGE = (
    "The model's answer could not be verified against the documents, so it is not shown."
)
LEAK_MESSAGE = "The answer was withheld because it contained confidential instructions."

_MARKERS = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class SourcePassage:
    """A retrieved passage, as shown to the user and given to the model."""

    number: int
    chunk_id: str
    doc_key: str
    file_name: str
    file_type: str
    title: str
    reference: str | None
    section: str
    pages: str | None
    page_start: int | None
    score: float
    similarity: float | None
    text: str
    superseded: bool
    flagged: bool

    @classmethod
    def from_hit(cls, number: int, hit: SearchHit, superseded: bool) -> SourcePassage:
        chunk = hit.chunk
        reference = None
        if chunk.doc_id:
            reference = chunk.doc_id + (f" v{chunk.version}" if chunk.version else "")
        return cls(
            number=number,
            chunk_id=chunk.chunk_id,
            doc_key=chunk.doc_key,
            file_name=chunk.file_name,
            file_type=chunk.file_type,
            title=chunk.title,
            reference=reference,
            section=chunk.section_label,
            pages=chunk.pages_label,
            page_start=chunk.page_start,
            score=hit.score,
            similarity=hit.similarity,
            text=chunk.text,
            superseded=superseded,
            flagged=chunk.flagged,
        )

    @property
    def location(self) -> str:
        parts = [self.file_name]
        if self.pages:
            parts.append(self.pages)
        parts.append(self.section)
        return " · ".join(parts)


@dataclass(frozen=True)
class AnswerStatement:
    text: str
    support: str  # "explicit" or "inferred"
    sources: list[int]
    warnings: list[str] = field(default_factory=list)


@dataclass
class AssistantAnswer:
    request_id: str
    question: str
    status: str
    statements: list[AnswerStatement]
    note: str
    citations: list[SourcePassage]
    retrieved: list[SourcePassage]
    model: str
    retriever: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    estimated_cost_usd: float | None = None
    removed_statements: list[AnswerStatement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def cited_chunk_ids(self) -> list[str]:
        return [citation.chunk_id for citation in self.citations]

    @property
    def answer_text(self) -> str:
        """Statements with their source markers, followed by the note."""
        sentences = [
            statement.text + "".join(f" [{number}]" for number in statement.sources)
            for statement in self.statements
        ]
        text = " ".join(sentences)
        if self.note:
            text = f"{text}\n\nNote: {self.note}" if text else self.note
        return text

    def to_dict(self) -> dict:
        data = asdict(self)
        data["usage"]["total_tokens"] = self.usage.total_tokens
        data["answer_text"] = self.answer_text
        return data


def _clean_statement(text: str, sources: list[int]) -> tuple[str, list[int]]:
    """Move citation markers written in the text, such as "[2]", to the sources."""
    found = [int(n) for group in _MARKERS.findall(text) for n in group.split(",")]
    return _MARKERS.sub("", text).strip(), list(dict.fromkeys([*sources, *found]))


class KnowledgeAssistant:
    """Answer questions from the indexed documents, with verifiable sources."""

    def __init__(
        self,
        store: ChromaStore,
        retriever: Retriever,
        generator: AnswerGenerator | None,
        settings: Settings,
        embedding_model: str | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.generator = generator
        self.settings = settings
        self.embedding_model = embedding_model
        self.canary = make_canary()

    @property
    def model_name(self) -> str:
        return self.generator.model if self.generator else "none"

    def _result(
        self, question: str, status: str, note: str, start: float, **extra
    ) -> AssistantAnswer:
        answer = AssistantAnswer(
            request_id=extra.pop("request_id"),
            question=question,
            status=status,
            statements=extra.pop("statements", []),
            note=note,
            citations=extra.pop("citations", []),
            retrieved=extra.pop("retrieved", []),
            model=extra.pop("model", self.model_name),
            retriever=self.retriever.name,
            total_ms=round((time.perf_counter() - start) * 1000, 1),
            **extra,
        )
        log_event(
            logger,
            "answer",
            request_id=answer.request_id,
            status=answer.status,
            retriever=answer.retriever,
            model=answer.model,
            retrieved=[(p.chunk_id, p.score) for p in answer.retrieved],
            cited=answer.cited_chunk_ids,
            removed_statements=len(answer.removed_statements),
            warnings=answer.warnings,
            prompt_tokens=answer.usage.prompt_tokens,
            output_tokens=answer.usage.output_tokens + answer.usage.thinking_tokens,
            retrieval_ms=answer.retrieval_ms,
            generation_ms=answer.generation_ms,
            total_ms=answer.total_ms,
            question_chars=len(question),
            **(
                {"question": question, "answer": answer.answer_text}
                if self.settings.log_content
                else {}
            ),
        )
        return answer

    def ask(
        self, question: str, include_superseded: bool = False, top_k: int | None = None
    ) -> AssistantAnswer:
        """Answer a question. `include_superseded` also searches outdated
        documents; `top_k` overrides the number of passages retrieved."""
        start = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        question = validate_question(question, self.settings.max_question_chars)

        if self.store.chunk_count() == 0:
            return self._result(
                question,
                INSUFFICIENT_EVIDENCE,
                EMPTY_INDEX_MESSAGE,
                start,
                request_id=request_id,
                warnings=["empty_index"],
            )
        if self.generator is None:
            raise ConfigurationError(
                "Answers need a language model: set GEMINI_API_KEY (see .env.example)."
            )

        hits = self.retriever.search(question, top_k or self.settings.top_k, include_superseded)
        retrieval_ms = round((time.perf_counter() - start) * 1000, 1)
        superseded_keys = frozenset(
            doc.doc_key for doc in self.store.documents() if doc.status == SUPERSEDED
        )
        # Passages found by semantic search must reach the relevance threshold.
        hits = [
            hit
            for hit in hits
            if hit.similarity is None or hit.similarity >= self.settings.min_relevance
        ]
        retrieved = [
            SourcePassage.from_hit(number, hit, hit.chunk.doc_key in superseded_keys)
            for number, hit in enumerate(hits, start=1)
        ]
        query_cost = (
            embedding_cost(self.embedding_model, estimate_tokens(question))
            if self.embedding_model
            else 0.0
        )
        if not hits:
            return self._result(
                question,
                INSUFFICIENT_EVIDENCE,
                NO_RELEVANT_MESSAGE,
                start,
                request_id=request_id,
                retrieval_ms=retrieval_ms,
                estimated_cost_usd=query_cost,
                warnings=["no_relevant_passage"],
            )

        result = self.generator.generate(
            system_prompt(self.canary), build_user_prompt(question, hits, superseded_keys)
        )
        generation = generation_cost(result.model, result.usage)
        if generation is None:
            generation = generation_cost(self.generator.model, result.usage)
        cost = None if generation is None or query_cost is None else query_cost + generation
        common = {
            "request_id": request_id,
            "retrieved": retrieved,
            "model": result.model,
            "usage": result.usage,
            "retrieval_ms": retrieval_ms,
            "generation_ms": round(result.latency_ms, 1),
            "estimated_cost_usd": cost,
        }
        return self._verify(question, result.payload, hits, retrieved, start, common)

    def _verify(
        self,
        question: str,
        payload: AnswerPayload,
        hits: list[SearchHit],
        retrieved: list[SourcePassage],
        start: float,
        common: dict,
    ) -> AssistantAnswer:
        outputs = [payload.note, *(statement.text for statement in payload.statements)]
        if contains_canary(outputs, self.canary):
            return self._result(
                question,
                INSUFFICIENT_EVIDENCE,
                LEAK_MESSAGE,
                start,
                warnings=["system_prompt_leak"],
                **common,
            )

        warnings: list[str] = []
        kept: list[AnswerStatement] = []
        removed: list[AnswerStatement] = []
        for raw in payload.statements:
            text, sources = _clean_statement(raw.text, raw.sources)
            if not text:
                continue
            valid = [n for n in dict.fromkeys(sources) if 1 <= n <= len(hits)]
            notes = [f"invalid_source:{n}" for n in sources if not 1 <= n <= len(hits)]
            warnings.extend(notes)
            if not valid:
                removed.append(AnswerStatement(text, raw.support, [], [*notes, "no_valid_source"]))
                continue
            evidence = []
            for number in valid:
                chunk = hits[number - 1].chunk
                evidence.append(
                    f"{chunk.title} {chunk.doc_id or ''} {chunk.section_label} {chunk.text}"
                )
            unverified = unverified_numbers(text, evidence, question)
            if unverified:
                notes.append("unverified_numbers:" + ",".join(unverified))
                warnings.append(f"unverified_numbers:{','.join(unverified)}")
            kept.append(AnswerStatement(text, raw.support, valid, notes))
        if removed:
            warnings.append(f"removed_unsupported_statements:{len(removed)}")

        status, note = payload.status, payload.note.strip()
        if status == ANSWERED and not kept:
            status, note = INSUFFICIENT_EVIDENCE, UNSUPPORTED_MESSAGE
            warnings.append("answer_without_valid_citation")
        elif status == INSUFFICIENT_EVIDENCE and not note:
            note = NOT_COVERED_MESSAGE
        elif status == CLARIFICATION_NEEDED and not note:
            note = "Could you make the question more specific?"

        cited_numbers = list(dict.fromkeys(n for statement in kept for n in statement.sources))
        citations = [retrieved[n - 1] for n in cited_numbers]
        return self._result(
            question,
            status,
            note,
            start,
            statements=kept,
            citations=citations,
            removed_statements=removed,
            warnings=warnings,
            **common,
        )
