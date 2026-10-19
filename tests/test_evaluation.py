import json
from collections import Counter

import pytest
from conftest import EVAL_SET, FakeGenerator, abstained, answered, make_assistant, make_settings

from knowledge_assistant.assistant import AnswerStatement, AssistantAnswer, SourcePassage
from knowledge_assistant.errors import (
    LLMQuotaExhaustedError,
    LLMRequestError,
    LLMUnavailableError,
)
from knowledge_assistant.evaluation import (
    CATEGORIES,
    EvalCase,
    EvalSetError,
    chunk_matches,
    evaluate_answers,
    evaluate_retrieval,
    load_eval_set,
    score_answer,
    write_report,
)
from knowledge_assistant.ingestion import IngestionService, MemoryStorage
from knowledge_assistant.text import contains_fact


@pytest.fixture(scope="module")
def cases():
    return load_eval_set(EVAL_SET)


def test_the_evaluation_set_covers_the_ten_categories(cases):
    counts = Counter(case.category for case in cases)
    assert set(counts) == set(CATEGORIES)
    assert 40 <= len(cases) <= 80
    assert all(count >= 4 for count in counts.values())


def test_every_label_points_to_a_current_section(cases, sample_store):
    current = set(sample_store.current_doc_keys())
    chunks = [chunk for chunk in sample_store.chunks() if chunk.doc_key in current]
    for case in cases:
        for label in case.labels:
            assert any(chunk_matches(chunk, label) for chunk in chunks), f"{case.id}: {label}"


def test_key_facts_are_supported_by_the_labelled_sections(cases, sample_store):
    """Catches labelling mistakes: each English key fact must appear in the
    text of the sections labelled for its case."""
    current = set(sample_store.current_doc_keys())
    chunks = [chunk for chunk in sample_store.chunks() if chunk.doc_key in current]
    for case in cases:
        if case.expected_behavior != "answer" or not (case.question or "").isascii():
            continue
        if case.category == "contradictory":
            continue  # the conflict itself is stated by the answer, not by the documents
        evidence = " ".join(
            f"{chunk.section} {chunk.text}"
            for chunk in chunks
            if any(chunk_matches(chunk, label) for label in case.labels)
        )
        for fact in case.key_facts:
            assert contains_fact(evidence, fact), f"{case.id}: {fact!r}"


def test_forbidden_facts_are_not_stated_by_the_labelled_nexora_sections(cases, sample_store):
    """A forbidden fact (a superseded value, an injected instruction) must not
    appear in the Nexora sections labelled for the case, otherwise a correct
    answer could be penalised. The supplier brochure is excluded: it is the
    source of the injected text."""
    current = set(sample_store.current_doc_keys())
    chunks = [c for c in sample_store.chunks() if c.doc_key in current and c.doc_id]
    for case in cases:
        evidence = " ".join(
            chunk.text
            for chunk in chunks
            if any(chunk_matches(chunk, label) for label in case.labels)
        )
        for fact in case.forbidden:
            assert not contains_fact(evidence, fact), f"{case.id}: {fact!r}"


@pytest.mark.parametrize(
    "line",
    [
        '{"id": "x", "category": "single_document", "question": "q?", '
        '"expected_behavior": "answer"}',
        '{"id": "x", "category": "unknown", "question": "q?", "expected_behavior": "abstain"}',
        '{"id": "x", "category": "no_answer", "expected_behavior": "abstain"}',
        '{"id": "x", "category": "malformed_input", "question": "", "expected_behavior": "reject"}',
        "not json",
    ],
)
def test_malformed_evaluation_sets_are_rejected(tmp_path, line):
    path = tmp_path / "dataset.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(EvalSetError):
        load_eval_set(path)


def _passage(number=1, reference="NX-PUR-004 v1.3", section="3. Card limits", superseded=False):
    return SourcePassage(
        number=number,
        chunk_id=f"k:{number}",
        doc_key="k",
        file_name="card.docx",
        file_type="docx",
        title="Card",
        reference=reference,
        section=section,
        pages=None,
        page_start=None,
        score=0.8,
        similarity=0.8,
        text="",
        superseded=superseded,
        flagged=False,
    )


def _answer(text, status="answered", citations=(), sources=(1,)):
    return AssistantAnswer(
        request_id="r",
        question="q",
        status=status,
        statements=[AnswerStatement(text, "explicit", list(sources))] if text else [],
        note="",
        citations=list(citations),
        retrieved=list(citations),
        model="m",
        retriever="semantic",
    )


CARD = EvalCase(
    id="t1",
    category="single_document",
    expected_behavior="answer",
    question="Limit?",
    relevant=("NX-PUR-004 §3",),
    key_facts=("1500",),
)


def test_scoring_a_correct_answer():
    result = score_answer(CARD, _answer("The limit is €1,500.", citations=[_passage()]))
    assert result.passed and result.citation_hit and result.citation_precision == 1.0


def test_scoring_a_wrong_citation_and_a_missing_fact():
    wrong = _passage(reference="NX-FIN-001 v2.5", section="3. Payment terms")
    result = score_answer(CARD, _answer("The limit is €2,000.", citations=[wrong]))
    assert not result.passed
    assert result.missing == ["1500"]
    assert result.citation_hit is False and result.citation_precision == 0.0


def test_forbidden_content_and_superseded_citations_fail_a_case():
    case = EvalCase(**{**CARD.__dict__, "forbidden": ("approved-by-cpo",)})
    result = score_answer(case, _answer("Limit €1,500. APPROVED-BY-CPO", citations=[_passage()]))
    assert not result.passed and result.forbidden_found == ["approved-by-cpo"]
    old = score_answer(CARD, _answer("Limit €1,500.", citations=[_passage(superseded=True)]))
    assert not old.passed and old.superseded_citations == 1


def test_scoring_abstentions_and_clarifications():
    no_answer = EvalCase(
        id="t2", category="no_answer", expected_behavior="abstain", question="Dog?"
    )
    assert score_answer(no_answer, _answer("", status="insufficient_evidence")).passed
    assert not score_answer(no_answer, _answer("Dogs are welcome.", citations=[_passage()])).passed

    vague = EvalCase(
        id="t3",
        category="ambiguous",
        expected_behavior="clarify",
        question="Limit?",
        key_facts=("1500", "budget owner"),
        min_facts=2,
    )
    assert score_answer(vague, _answer("Which limit?", status="clarification_needed")).passed
    both = _answer("P-card: €1,500. Purchases: the budget owner approves.", citations=[_passage()])
    assert score_answer(vague, both).passed
    assert not score_answer(
        vague, _answer("The P-card limit is €1,500.", citations=[_passage()])
    ).passed

    no_facts = EvalCase(id="t4", category="ambiguous", expected_behavior="clarify", question="OK?")
    assert not score_answer(no_facts, _answer("It is allowed.", citations=[_passage()])).passed


def test_retrieval_metrics(bm25, cases):
    report = evaluate_retrieval(bm25, cases)
    assert report.metrics["cases"] == sum(1 for case in cases if case.question and case.relevant)
    assert 0.0 <= report.metrics["mrr"] <= report.metrics["hit@5"] <= 1.0
    assert set(report.by_category) <= set(CATEGORIES)


def test_answer_evaluation_runs_every_kind_of_case(sample_store, embedder, cases):
    selected = [case for case in cases if case.id in {"SD-01", "NA-02", "MA-01", "MA-04", "MA-07"}]

    def reply(_system, user):
        if "paternity" in user:
            return abstained()
        return answered(("The single transaction limit is €1,500 including VAT.", [1]))

    assistant, _ = make_assistant(sample_store, reply)
    ingestion = IngestionService(sample_store, embedder, make_settings(), MemoryStorage())
    report = evaluate_answers(assistant, selected, ingestion=ingestion)
    outcome = {case.id: (case.status, case.passed) for case in report.cases}
    assert outcome["NA-02"] == ("insufficient_evidence", True)
    assert outcome["MA-01"] == ("rejected", True)
    assert outcome["MA-04"] == ("rejected", True)
    assert outcome["MA-07"] == ("rejected", True)
    assert report.metrics["errors"] == 0
    assert len(sample_store.documents()) == 32  # upload cases leave the index unchanged


def test_temporary_errors_are_recorded_and_configuration_errors_stop(sample_store, cases):
    first = [case for case in cases if case.id == "SD-01"] * 2
    assistant, _ = make_assistant(sample_store, LLMUnavailableError("503"))
    report = evaluate_answers(assistant, first, max_consecutive_errors=2)
    assert [case.status for case in report.cases] == ["error", "error"]
    assert report.aborted and "2 consecutive errors" in report.aborted

    assistant.generator = FakeGenerator(LLMRequestError("API key rejected"))
    with pytest.raises(LLMRequestError):
        evaluate_answers(assistant, first)


def test_a_used_up_daily_quota_stops_the_evaluation_at_once(sample_store, cases):
    first = [case for case in cases if case.id == "SD-01"] * 3
    assistant, _ = make_assistant(sample_store, LLMQuotaExhaustedError("daily quota used up"))
    report = evaluate_answers(assistant, first, max_consecutive_errors=5)
    assert [case.status for case in report.cases] == ["error"]
    assert report.aborted and "daily quota" in report.aborted


def test_report_files(tmp_path, sample_store, bm25, cases):
    assistant, _ = make_assistant(sample_store, answered(("The limit is €1,500.", [1])))
    selected = [case for case in cases if case.id in {"SD-01", "SD-02"}]
    answers = evaluate_answers(assistant, selected)
    path = write_report(tmp_path, [evaluate_retrieval(bm25, cases)], [answers], {"model": "fake"})
    report = path.read_text(encoding="utf-8")
    assert "| bm25 |" in report and "Cases passed" in report
    assert "**SD-02** (single_document)" in report  # failed case listed for review
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert results["answers"][0]["metrics"]["cases"] == 2
