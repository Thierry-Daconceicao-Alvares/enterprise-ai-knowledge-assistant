"""Measure retrieval and answer quality on a labelled evaluation set.

The checks are deterministic, so two runs on the same answers give the same
scores and every number can be traced back to a case:

- retrieval: is a relevant passage among the first results, and how high?
- answers: does the answer behave as expected (answer, abstain, ask for
  clarification, reject the input), contain the expected facts, avoid the
  forbidden ones, and cite relevant passages?

Labels refer to document sections ("NX-PUR-002 §2"), not to chunk IDs, so
they stay valid when the chunk size changes. Key facts are matched as
phrases after normalisation (numbers, plurals, French and English number
words); this is stricter than a human reader, so failures are listed in the
report for manual review.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .assistant import AssistantAnswer, KnowledgeAssistant
from .config import PROJECT_ROOT
from .errors import DocumentError, InvalidQuestionError, LLMError, LLMRequestError
from .ingestion import REJECTED, IngestionService
from .llm import ANSWERED, CLARIFICATION_NEEDED
from .retrieval import Retriever
from .text import answer_verdict, contains_fact
from .vector_store import ChromaStore, StoredChunk

logger = logging.getLogger(__name__)

CATEGORIES = (
    "single_document",
    "multi_passage",
    "multi_document",
    "no_answer",
    "ambiguous",
    "contradictory",
    "citation",
    "misleading_instructions",
    "superseded",
    "malformed_input",
)
BEHAVIORS = ("answer", "abstain", "clarify", "reject")
K_VALUES = (1, 3, 5)
ERROR = "error"
REJECTED_STATUS = "rejected"


class EvalSetError(ValueError):
    """The evaluation set is malformed."""


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    expected_behavior: str
    question: str | None = None
    upload: str | None = None
    relevant: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    key_facts: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    verdict: str | None = None
    min_facts: int | None = None
    expected_error: str | None = None
    reference_answer: str = ""

    @property
    def checks(self) -> list[str]:
        verdict = [f"verdict: {self.verdict}"] if self.verdict else []
        return verdict + list(self.key_facts)

    @property
    def labels(self) -> set[str]:
        return set(self.relevant) | set(self.related)


def parse_label(label: str) -> tuple[str, str | None]:
    """'NX-PUR-002 §2' -> ('NX-PUR-002', '2'); 'file.pdf' -> ('file.pdf', None)."""
    reference, _, section = label.partition("§")
    return reference.strip(), section.strip() or None


def chunk_matches(chunk: StoredChunk, label: str) -> bool:
    reference, section = parse_label(label)
    if reference not in (chunk.doc_id, chunk.file_name):
        return False
    if section is None:
        return True
    return bool(chunk.section) and chunk.section.split(".")[0].strip() == section


def load_eval_set(path: Path) -> list[EvalCase]:
    """Read and validate the JSONL evaluation set."""
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            case = EvalCase(
                id=raw["id"],
                category=raw["category"],
                expected_behavior=raw["expected_behavior"],
                question=raw.get("question"),
                upload=raw.get("upload"),
                relevant=tuple(raw.get("relevant", [])),
                related=tuple(raw.get("related", [])),
                key_facts=tuple(raw.get("key_facts", [])),
                forbidden=tuple(raw.get("forbidden", [])),
                verdict=raw.get("verdict"),
                min_facts=raw.get("min_facts"),
                expected_error=raw.get("expected_error"),
                reference_answer=raw.get("reference_answer", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EvalSetError(f"{path.name}, line {line_number}: {exc}") from exc
        problems = []
        if case.id in seen:
            problems.append("duplicate id")
        if case.category not in CATEGORIES:
            problems.append(f"unknown category {case.category!r}")
        if case.expected_behavior not in BEHAVIORS:
            problems.append(f"unknown behavior {case.expected_behavior!r}")
        if (case.question is None) == (case.upload is None):
            problems.append("needs either a question or an upload")
        if case.expected_behavior == "answer" and not (case.relevant and case.key_facts):
            problems.append("an answer case needs relevant sources and key facts")
        if case.expected_behavior == "reject" and not case.expected_error:
            problems.append("a reject case needs expected_error")
        if case.verdict not in (None, "yes", "no"):
            problems.append("verdict must be 'yes' or 'no'")
        if problems:
            raise EvalSetError(f"{case.id}: {'; '.join(problems)}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise EvalSetError(f"{path.name} contains no case")
    return cases


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _percentile(values: Sequence[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 1)


# --------------------------------------------------------------------------- retrieval


@dataclass
class RetrievalCase:
    id: str
    category: str
    retrieved: list[str]  # "doc_id §section" of each result
    first_relevant_rank: int | None
    labels_found: int
    labels_total: int
    top_score: float | None


@dataclass
class RetrievalReport:
    retriever: str
    cases: list[RetrievalCase]
    metrics: dict[str, float | None]
    by_category: dict[str, dict[str, float | None]]


def _retrieval_metrics(cases: Sequence[RetrievalCase]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {"cases": float(len(cases))}
    for k in K_VALUES:
        metrics[f"hit@{k}"] = _mean(
            [1.0 if c.first_relevant_rank and c.first_relevant_rank <= k else 0.0 for c in cases]
        )
    metrics["recall@5"] = _mean([c.labels_found / c.labels_total for c in cases if c.labels_total])
    metrics["mrr"] = _mean(
        [1.0 / c.first_relevant_rank if c.first_relevant_rank else 0.0 for c in cases]
    )
    return metrics


def _describe(chunk: StoredChunk) -> str:
    section = chunk.section.split(".")[0] if chunk.section else "header"
    return f"{chunk.doc_id or chunk.file_name} §{section}"


def evaluate_retrieval(
    retriever: Retriever, cases: Sequence[EvalCase], k: int = 5
) -> RetrievalReport:
    """Hit@k, recall@5 and MRR on the cases that have labelled sources."""
    results: list[RetrievalCase] = []
    for case in cases:
        if not case.question or not case.relevant:
            continue
        hits = retriever.search(case.question, max(k, max(K_VALUES)))
        rank = next(
            (
                position
                for position, hit in enumerate(hits, start=1)
                if any(chunk_matches(hit.chunk, label) for label in case.labels)
            ),
            None,
        )
        found = sum(
            any(chunk_matches(hit.chunk, label) for hit in hits[:5]) for label in case.relevant
        )
        results.append(
            RetrievalCase(
                id=case.id,
                category=case.category,
                retrieved=[_describe(hit.chunk) for hit in hits],
                first_relevant_rank=rank,
                labels_found=found,
                labels_total=len(case.relevant),
                top_score=(hits[0].similarity if hits[0].similarity is not None else hits[0].score)
                if hits
                else None,
            )
        )
    grouped: dict[str, list[RetrievalCase]] = defaultdict(list)
    for result in results:
        grouped[result.category].append(result)
    return RetrievalReport(
        retriever=retriever.name,
        cases=results,
        metrics=_retrieval_metrics(results),
        by_category={name: _retrieval_metrics(group) for name, group in sorted(grouped.items())},
    )


# ----------------------------------------------------------------------------- answers


@dataclass
class AnswerCase:
    id: str
    category: str
    expected_behavior: str
    question: str
    status: str
    answer: str = ""
    passed: bool = False
    checks: list[str] = field(default_factory=list)
    facts_found: list[bool] = field(default_factory=list)
    forbidden_found: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    citation_hit: bool | None = None
    citation_precision: float | None = None
    superseded_citations: int = 0
    removed_statements: int = 0
    unverified_numbers: int = 0
    latency_ms: float | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    top_similarity: float | None = None
    retrieved: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def missing(self) -> list[str]:
        return [
            check for check, found in zip(self.checks, self.facts_found, strict=False) if not found
        ]

    @property
    def fact_recall(self) -> float:
        if not self.facts_found:
            return 0.0
        return sum(self.facts_found) / len(self.facts_found)


def score_answer(case: EvalCase, answer: AssistantAnswer) -> AnswerCase:
    """Compare an answer with the labels of its case."""
    text = answer.answer_text
    result = AnswerCase(
        id=case.id,
        category=case.category,
        expected_behavior=case.expected_behavior,
        question=case.question or "",
        status=answer.status,
        answer=text,
        checks=case.checks,
        cited=[f"{c.reference or c.file_name} {c.section}" for c in answer.citations],
        superseded_citations=sum(c.superseded for c in answer.citations),
        removed_statements=len(answer.removed_statements),
        unverified_numbers=sum(
            any(w.startswith("unverified_numbers") for w in s.warnings) for s in answer.statements
        ),
        latency_ms=answer.total_ms,
        prompt_tokens=answer.usage.prompt_tokens,
        output_tokens=answer.usage.output_tokens + answer.usage.thinking_tokens,
        cost_usd=answer.estimated_cost_usd,
        top_similarity=max(
            (p.similarity for p in answer.retrieved if p.similarity is not None), default=None
        ),
        retrieved=[f"{p.reference or p.file_name} {p.section}" for p in answer.retrieved],
        warnings=list(answer.warnings),
    )
    verdict = [answer_verdict(text) == case.verdict] if case.verdict else []
    result.facts_found = verdict + [contains_fact(text, fact) for fact in case.key_facts]
    result.forbidden_found = [fact for fact in case.forbidden if contains_fact(text, fact)]

    if answer.citations and case.labels:
        matches = [
            any(_passage_matches(citation, label) for label in case.labels)
            for citation in answer.citations
        ]
        result.citation_hit = any(
            any(_passage_matches(citation, label) for label in case.relevant)
            for citation in answer.citations
        )
        result.citation_precision = sum(matches) / len(matches)
    elif case.relevant and answer.status == ANSWERED:
        result.citation_hit, result.citation_precision = False, 0.0

    safe = not result.forbidden_found and result.superseded_citations == 0
    behavior = case.expected_behavior
    if behavior == "answer":
        result.passed = answer.status == ANSWERED and all(result.facts_found) and safe
    elif behavior == "abstain":
        result.passed = answer.status != ANSWERED and safe
    elif behavior == "clarify":
        # Without listed interpretations, only a clarifying question passes.
        needed = min(case.min_facts or 2, len(result.facts_found))
        covered = bool(result.facts_found) and sum(result.facts_found) >= needed
        result.passed = safe and (
            answer.status == CLARIFICATION_NEEDED or (answer.status == ANSWERED and covered)
        )
    return result


def _passage_matches(citation, label: str) -> bool:
    reference, section = parse_label(label)
    doc_reference = (citation.reference or "").split(" v")[0]
    if reference not in (doc_reference, citation.file_name):
        return False
    if section is None:
        return True
    return citation.section.split(".")[0].strip() == section


@dataclass
class AnswerReport:
    model: str
    retriever: str
    cases: list[AnswerCase]
    metrics: dict[str, float | int | None]
    by_category: dict[str, dict[str, float | int | None]]
    aborted: str | None = None


def _answer_metrics(cases: Sequence[AnswerCase]) -> dict[str, float | int | None]:
    def of(behavior: str) -> list[AnswerCase]:
        return [c for c in cases if c.expected_behavior == behavior]

    answerable = of("answer")
    answered = [c for c in answerable if c.status == ANSWERED]
    unanswerable = [c for c in of("abstain") if c.category == "no_answer"]
    completed = [c for c in cases if c.latency_ms is not None and c.status not in (ERROR,)]
    with_cost = [c.cost_usd for c in completed if c.cost_usd is not None]
    return {
        "cases": len(cases),
        "pass_rate": _mean([1.0 if c.passed else 0.0 for c in cases]),
        "errors": sum(c.status == ERROR for c in cases),
        "error_rate": _mean([1.0 if c.status == ERROR else 0.0 for c in cases]),
        "answer_rate": _mean([1.0 if c.status == ANSWERED else 0.0 for c in answerable]),
        "key_fact_recall": _mean(
            [c.fact_recall if c.status == ANSWERED else 0.0 for c in answerable]
        ),
        "fully_correct": _mean([1.0 if c.passed else 0.0 for c in answerable]),
        "citation_hit_rate": _mean([1.0 if c.citation_hit else 0.0 for c in answered]),
        "citation_precision": _mean([c.citation_precision or 0.0 for c in answered]),
        "abstention_rate": _mean([1.0 if c.status != ANSWERED else 0.0 for c in unanswerable]),
        "unsupported_answer_rate": _mean(
            [1.0 if c.status == ANSWERED else 0.0 for c in unanswerable]
        ),
        "forbidden_content": sum(bool(c.forbidden_found) for c in cases),
        "superseded_citations": sum(c.superseded_citations for c in cases),
        "removed_statements": sum(c.removed_statements for c in cases),
        "latency_p50_ms": _percentile([c.latency_ms or 0.0 for c in completed], 50),
        "latency_p95_ms": _percentile([c.latency_ms or 0.0 for c in completed], 95),
        "avg_prompt_tokens": _mean([float(c.prompt_tokens) for c in completed if c.prompt_tokens]),
        "avg_output_tokens": _mean([float(c.output_tokens) for c in completed if c.prompt_tokens]),
        "avg_cost_usd": round(statistics.fmean(with_cost), 6) if with_cost else None,
        "total_cost_usd": round(sum(with_cost), 6) if with_cost else None,
    }


def evaluate_answers(
    assistant: KnowledgeAssistant,
    cases: Sequence[EvalCase],
    ingestion: IngestionService | None = None,
    delay_s: float = 0.0,
    max_consecutive_errors: int = 5,
    progress: Callable[[int, int, AnswerCase], None] | None = None,
) -> AnswerReport:
    """Run every case and score it.

    A temporary API error is recorded and the run continues. A configuration
    error (invalid key or model) stops the run immediately, and several
    consecutive errors stop it too: the report then covers the cases run so far.
    """
    results: list[AnswerCase] = []
    consecutive_errors = 0
    aborted: str | None = None
    model = assistant.model_name
    for index, case in enumerate(cases, start=1):
        called_model = False
        if case.upload:
            result = _run_upload_case(case, ingestion)
        else:
            try:
                answer = assistant.ask(case.question or "")
            except InvalidQuestionError as exc:
                result = AnswerCase(
                    case.id,
                    case.category,
                    case.expected_behavior,
                    case.question or "",
                    REJECTED_STATUS,
                    answer=str(exc),
                    error=exc.code,
                    passed=case.expected_behavior == "reject" and exc.code == case.expected_error,
                )
            except LLMRequestError:
                raise
            except LLMError as exc:
                consecutive_errors += 1
                result = AnswerCase(
                    case.id,
                    case.category,
                    case.expected_behavior,
                    case.question or "",
                    ERROR,
                    error=str(exc),
                    checks=case.checks,
                    facts_found=[False] * len(case.checks),
                )
                if consecutive_errors >= max_consecutive_errors:
                    results.append(result)
                    aborted = f"stopped after {consecutive_errors} consecutive errors ({exc})"
                    logger.error("Evaluation %s", aborted)
                    break
            else:
                consecutive_errors = 0
                called_model = answer.usage.prompt_tokens > 0
                model = answer.model if called_model else model
                result = score_answer(case, answer)
        results.append(result)
        if progress:
            progress(index, len(cases), result)
        if delay_s and called_model and index < len(cases):
            time.sleep(delay_s)

    grouped: dict[str, list[AnswerCase]] = defaultdict(list)
    for result in results:
        grouped[result.category].append(result)
    return AnswerReport(
        model=model,
        retriever=assistant.retriever.name,
        cases=results,
        metrics=_answer_metrics(results),
        by_category={name: _answer_metrics(group) for name, group in sorted(grouped.items())},
        aborted=aborted,
    )


def _run_upload_case(case: EvalCase, ingestion: IngestionService | None) -> AnswerCase:
    """Malformed uploads are tested on the ingestion service of a scratch index."""
    if ingestion is None:
        raise ValueError("Upload cases need an ingestion service.")
    path = PROJECT_ROOT / (case.upload or "")
    try:
        outcome = ingestion.ingest_bytes(path.name, path.read_bytes())
        status, code, message = outcome.status, outcome.error_code, outcome.message
        if outcome.doc_key and outcome.status != REJECTED:
            ingestion.delete(outcome.doc_key)  # keep the evaluation index unchanged
    except DocumentError as exc:  # pragma: no cover - ingest_bytes returns rejections
        status, code, message = REJECTED, exc.code, str(exc)
    return AnswerCase(
        case.id,
        case.category,
        case.expected_behavior,
        f"[upload] {path.name}",
        REJECTED_STATUS if status == REJECTED else status,
        answer=message,
        error=code,
        passed=case.expected_behavior == "reject"
        and status == REJECTED
        and code == case.expected_error,
    )


# ------------------------------------------------------------------------------ report


def _pct(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _num(value: float | int | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _retrieval_section(reports: Sequence[RetrievalReport]) -> list[str]:
    cases = int(reports[0].metrics["cases"] or 0)
    lines = [
        "## Retrieval",
        "",
        f"{cases} questions with labelled relevant sections. Hit@k: a relevant passage is in "
        "the first k results. Recall@5: share of the required sections found in the first 5 "
        "results. MRR: mean of 1 / rank of the first relevant passage.",
        "",
        "| Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        m = report.metrics
        lines.append(
            f"| {report.retriever} | {_pct(m['hit@1'])} | {_pct(m['hit@3'])} | {_pct(m['hit@5'])} "
            f"| {_pct(m['recall@5'])} | {_num(m['mrr'], 2)} |"
        )
    categories = sorted({name for report in reports for name in report.by_category})
    lines += [
        "",
        "Hit@5 by category:",
        "",
        "| Category | " + " | ".join(report.retriever for report in reports) + " |",
        "|---|" + "---:|" * len(reports),
    ]
    for name in categories:
        cells = [_pct(report.by_category.get(name, {}).get("hit@5")) for report in reports]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _answer_section(report: AnswerReport) -> list[str]:
    m = report.metrics
    lines = [
        f"## Answers ({report.retriever} retrieval)",
        "",
        f"Model `{report.model}`, {m['cases']} cases, {m['errors']} API errors.",
        "",
        *([f"**Run incomplete: {report.aborted}.**", ""] if report.aborted else []),
        "| Metric | Value | Definition |",
        "|---|---:|---|",
        f"| Cases passed | {_pct(m['pass_rate'])} | Expected behaviour (answer, abstain, "
        "clarify, reject) with all checks passed |",
        f"| Fully correct answers | {_pct(m['fully_correct'])} | Answerable questions answered "
        "with every key fact and the right yes/no verdict |",
        f"| Key-fact recall | {_pct(m['key_fact_recall'])} | Share of expected facts present "
        "(abstentions count as 0) |",
        f"| Answer rate | {_pct(m['answer_rate'])} | Answerable questions that were answered |",
        f"| Citation hit rate | {_pct(m['citation_hit_rate'])} | Answers citing at least one "
        "required section |",
        f"| Citation precision | {_pct(m['citation_precision'])} | Share of citations pointing "
        "to a labelled (required or related) section |",
        f"| Correct abstentions | {_pct(m['abstention_rate'])} | Questions not covered by the "
        "documents that were not answered |",
        f"| Unsupported answers | {_pct(m['unsupported_answer_rate'])} | Questions not covered "
        "by the documents that were answered anyway |",
        f"| Forbidden content | {m['forbidden_content']} | Answers containing a forbidden "
        "statement (injected instruction, superseded value) |",
        f"| Superseded citations | {m['superseded_citations']} | Citations of superseded "
        "documents |",
        f"| Statements removed | {m['removed_statements']} | Statements without a valid source, "
        "removed by the guardrail |",
        f"| Error rate | {_pct(m['error_rate'])} | Cases that failed with an API error |",
        f"| Latency p50 / p95 | {_num(m['latency_p50_ms'])} / {_num(m['latency_p95_ms'])} ms "
        "| End to end, retrieval and generation |",
        f"| Tokens per question | {_num(m['avg_prompt_tokens'])} in / "
        f"{_num(m['avg_output_tokens'])} out | Average; output includes thinking tokens |",
        f"| Estimated cost | ${_num(m['avg_cost_usd'], 5)} per question, "
        f"${_num(m['total_cost_usd'], 4)} in total | Paid-tier list prices (see costs.py) |",
        "",
        "| Category | Cases | Passed | Key-fact recall | Citation precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CATEGORIES:
        cm = report.by_category.get(name)
        if not cm:
            continue
        lines.append(
            f"| {name} | {cm['cases']} | {_pct(cm['pass_rate'])} | "
            f"{_pct(cm['key_fact_recall'])} | {_pct(cm['citation_precision'])} |"
        )
    lines += _similarity_lines(report)
    failures = [case for case in report.cases if not case.passed]
    lines += ["", f"### Cases to review ({len(failures)})", ""]
    if not failures:
        lines.append("None.")
    for case in failures:
        reasons = []
        if case.status == ERROR:
            reasons.append(f"error: {case.error}")
        elif case.expected_behavior == "answer" and case.status != ANSWERED:
            reasons.append(f"expected an answer, got {case.status}")
        elif case.expected_behavior == "abstain" and case.status == ANSWERED:
            reasons.append("answered although the documents do not cover it")
        elif case.expected_behavior == "clarify":
            reasons.append(f"expected a clarification or a complete overview, got {case.status}")
        elif case.expected_behavior == "reject":
            reasons.append(f"expected a rejection, got {case.status} ({case.error})")
        if case.missing and case.status == ANSWERED:
            reasons.append("missing: " + ", ".join(f"`{item}`" for item in case.missing))
        if case.forbidden_found:
            reasons.append("forbidden: " + ", ".join(f"`{item}`" for item in case.forbidden_found))
        if case.superseded_citations:
            reasons.append(f"{case.superseded_citations} superseded citation(s)")
        answer = " ".join(case.answer.split())
        if len(answer) > 400:
            answer = answer[:397] + "..."
        lines.append(f"- **{case.id}** ({case.category}): {case.question}")
        lines.append(f"  - {'; '.join(reasons) or 'see results.json'}")
        if answer:
            lines.append(f"  - Answer: {answer}")
    return lines


def _similarity_lines(report: AnswerReport) -> list[str]:
    """Top-1 cosine similarity of answerable questions and of questions the
    documents do not cover: the data needed to choose MIN_RELEVANCE."""
    groups = {
        "Answerable questions": [
            c.top_similarity
            for c in report.cases
            if c.expected_behavior == "answer" and c.top_similarity is not None
        ],
        "Questions not covered by the documents": [
            c.top_similarity
            for c in report.cases
            if c.category == "no_answer" and c.top_similarity is not None
        ],
    }
    if not any(groups.values()):
        return []
    lines = [
        "",
        "Top-1 relevance (cosine similarity of the best passage), used to calibrate "
        "`MIN_RELEVANCE`:",
        "",
        "| Questions | Count | Min | Median | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in groups.items():
        if values:
            lines.append(
                f"| {name} | {len(values)} | {min(values):.2f} | "
                f"{statistics.median(values):.2f} | {max(values):.2f} |"
            )
    return lines


def write_report(
    output_dir: Path,
    retrieval_reports: Sequence[RetrievalReport],
    answer_reports: Sequence[AnswerReport] = (),
    context: dict | None = None,
) -> Path:
    """Write `report.md` (for people) and `results.json` (for tools)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    context = {"generated": generated, **(context or {})}
    lines = ["# Evaluation report", ""]
    lines += [f"- {key.replace('_', ' ').capitalize()}: {value}" for key, value in context.items()]
    lines.append("")
    if retrieval_reports:
        lines += [*_retrieval_section(retrieval_reports), ""]
    for report in answer_reports:
        lines += [*_answer_section(report), ""]
    path = output_dir / "report.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    results = {
        "context": context,
        "retrieval": [asdict(report) for report in retrieval_reports],
        "answers": [asdict(report) for report in answer_reports],
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def build_eval_index(store: ChromaStore, ingestion: IngestionService, folder: Path) -> None:
    """Index the sample documents into a fresh store."""
    results = ingestion.ingest_folder(folder)
    rejected = [r for r in results if r.status == REJECTED]
    if rejected:
        names = ", ".join(r.file_name for r in rejected)
        raise RuntimeError(f"Sample documents were rejected: {names}")
