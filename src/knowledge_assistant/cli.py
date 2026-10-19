"""Command-line interface: `knowledge-assistant <command>` or `python -m knowledge_assistant`.

Commands:
  ingest     index files or folders (--samples for the synthetic Nexora documents)
  documents  list the indexed documents and their status
  rebuild    empty the index and index the stored originals again
  search     show the passages retrieved for a question (no model call)
  ask        answer a question with cited sources
  evaluate   run the evaluation set and write a report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from . import __version__
from .config import PROJECT_ROOT, Settings
from .errors import AssistantError
from .evaluation import (
    AnswerCase,
    build_eval_index,
    evaluate_answers,
    evaluate_retrieval,
    load_eval_set,
    write_report,
)
from .factory import build_client, build_components, build_embedder, build_retriever
from .ingestion import IngestionService, MemoryStorage
from .logging_setup import configure_logging
from .vector_store import ChromaStore

DEFAULT_EVAL_SET = PROJECT_ROOT / "eval" / "dataset.jsonl"
DEFAULT_EVAL_OUTPUT = PROJECT_ROOT / "eval" / "results" / "latest"
RETRIEVERS = ("semantic", "bm25", "hybrid")


def _print_answer(answer) -> None:
    labels = {
        "answered": "ANSWER",
        "insufficient_evidence": "NOT ENOUGH EVIDENCE",
        "clarification_needed": "CLARIFICATION NEEDED",
    }
    print(f"\n[{labels.get(answer.status, answer.status.upper())}]")
    for statement in answer.statements:
        markers = "".join(f"[{n}]" for n in statement.sources)
        tag = " (inferred)" if statement.support == "inferred" else ""
        print(f"- {statement.text} {markers}{tag}")
    if answer.note:
        print(f"\n{answer.note}")
    if answer.citations:
        print("\nSources:")
        for citation in answer.citations:
            print(
                f"  [{citation.number}] {citation.title} | {citation.location} "
                f"| relevance {citation.similarity if citation.similarity is not None else '-'}"
            )
    for warning in answer.warnings:
        print(f"  ! {warning}")
    cost = f", ~${answer.estimated_cost_usd:.5f}" if answer.estimated_cost_usd is not None else ""
    print(
        f"\n{answer.retriever} retrieval {answer.retrieval_ms:.0f} ms, generation "
        f"{answer.generation_ms:.0f} ms, {answer.usage.total_tokens} tokens{cost}, "
        f"model {answer.model}"
    )


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    components = build_components(settings)
    paths: list[Path] = [settings.sample_dir] if args.samples else []
    paths += [Path(item) for item in args.paths]
    if not paths:
        print("Nothing to ingest: give files or folders, or use --samples.", file=sys.stderr)
        return 2
    results = []
    for path in paths:
        if path.is_dir():
            results += components.ingestion.ingest_folder(path)
        elif path.is_file():
            results += components.ingestion.ingest_paths([path])
        else:
            print(f"Not found: {path}", file=sys.stderr)
            return 2
    for result in results:
        print(f"{result.status:<10} {result.file_name}: {result.message}")
        for warning in result.warnings:
            print(f"           ! {warning}")
    documents = components.store.documents()
    print(f"\nIndex: {len(documents)} documents, {components.store.chunk_count()} passages.")
    return 1 if any(result.status == "rejected" for result in results) else 0


def cmd_documents(_args: argparse.Namespace, settings: Settings) -> int:
    components = build_components(settings)
    documents = components.store.documents()
    if not documents:
        print("No document indexed. Run: knowledge-assistant ingest --samples")
        return 0
    for doc in documents:
        reference = f"{doc.doc_id} v{doc.version}" if doc.doc_id else "-"
        pages = f"{doc.page_count} p." if doc.page_count else doc.file_type
        status = doc.status + (f" ({doc.status_reason})" if doc.status_reason else "")
        print(
            f"{doc.file_name:<62} {reference:<16} {pages:>6} {doc.chunk_count:>3} chunks  {status}"
        )
    return 0


def cmd_rebuild(args: argparse.Namespace, settings: Settings) -> int:
    components = build_components(settings)
    results = components.ingestion.rebuild(settings.sample_dir if args.samples else None)
    print(
        f"Rebuilt the index: {sum(r.ok for r in results)} documents, "
        f"{components.store.chunk_count()} passages."
    )
    return 0


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    components = build_components(settings)
    retriever = build_retriever(args.retriever, components.store, components.embedder)
    hits = retriever.search(args.question, args.k, args.include_superseded)
    if not hits:
        print("No matching passage.")
    for hit in hits:
        chunk = hit.chunk
        where = " · ".join(part for part in (chunk.pages_label, chunk.section_label) if part)
        print(f"{hit.rank}. [{hit.score:.3f}] {chunk.file_name} · {where}")
        preview = " ".join(chunk.text.split())
        print(f"   {preview[:160]}{'...' if len(preview) > 160 else ''}")
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    components = build_components(settings)
    answer = components.assistant.ask(args.question, include_superseded=args.include_superseded)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_answer(answer)
    return 0


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    """Index the sample documents in a fresh in-memory store, then evaluate."""
    cases = load_eval_set(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    client = build_client(settings)
    embedder = build_embedder(settings, client)
    store = ChromaStore.in_memory(
        f"eval-{uuid.uuid4().hex[:8]}", embedder.signature, embedder.dimensions
    )
    ingestion = IngestionService(store, embedder, settings, MemoryStorage())
    build_eval_index(store, ingestion, settings.sample_dir)
    print(f"Indexed {len(store.documents())} documents ({store.chunk_count()} passages).")

    retrieval_reports = []
    for name in args.retrievers:
        report = evaluate_retrieval(build_retriever(name, store, embedder), cases)
        retrieval_reports.append(report)
        m = report.metrics
        print(
            f"{name:>8}: hit@1 {m['hit@1']:.2f}  hit@5 {m['hit@5']:.2f}  "
            f"recall@5 {m['recall@5']:.2f}  MRR {m['mrr']:.2f}"
        )

    context = {
        "dataset": f"{args.dataset.name} ({len(cases)} cases)",
        "corpus": f"{len(store.documents())} documents, {store.chunk_count()} passages",
        "embeddings": embedder.signature,
        "chunking": f"{settings.chunk_size} characters, overlap {settings.chunk_overlap}",
        "top_k": settings.top_k,
        "min_relevance": settings.min_relevance,
    }
    if os.environ.get("GITHUB_SHA"):  # the evaluated commit, when run in GitHub Actions
        context["commit"] = os.environ["GITHUB_SHA"][:12]
    answer_reports = []
    if not args.retrieval_only:
        components = build_components(
            settings, store=store, client=client, embedder=embedder, storage=MemoryStorage()
        )
        context["generation_model"] = settings.generation_model

        def progress(index: int, total: int, case: AnswerCase) -> None:
            outcome = "PASS" if case.passed else "FAIL"
            detail = f" ({case.error})" if case.error else ""
            print(f"[{index:>2}/{total}] {case.id:<7} {outcome} {case.status}{detail}", flush=True)

        for name in args.answer_retrievers:
            assistant = components.assistant
            assistant.retriever = build_retriever(name, store, embedder)
            print(f"\nAnswers with {name} retrieval:")
            report = evaluate_answers(
                assistant,
                cases,
                ingestion=components.ingestion,
                delay_s=args.delay,
                progress=progress,
            )
            answer_reports.append(report)
            m = report.metrics
            print(f"passed {m['pass_rate']:.0%}, errors {m['errors']}")
            if report.aborted:
                break

    path = write_report(args.output, retrieval_reports, answer_reports, context=context)
    print(f"\nReport written to {path}")
    aborted = next((report.aborted for report in answer_reports if report.aborted), None)
    if aborted:
        print(f"Error (evaluation_incomplete): the evaluation {aborted}", file=sys.stderr)
        return 1
    return 0


def _choices(raw: str) -> list[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    unknown = [value for value in values if value not in RETRIEVERS]
    if unknown or not values:
        raise argparse.ArgumentTypeError(f"choose among {', '.join(RETRIEVERS)}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge-assistant",
        description="Answer questions from company documents with cited sources.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show detailed logs")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index files or folders")
    ingest.add_argument("paths", nargs="*", help="files or folders to index")
    ingest.add_argument("--samples", action="store_true", help="index the sample documents")

    commands.add_parser("documents", help="list the indexed documents")

    rebuild = commands.add_parser("rebuild", help="rebuild the index from the stored originals")
    rebuild.add_argument("--samples", action="store_true", help="also index the sample documents")

    search = commands.add_parser("search", help="show the passages retrieved for a question")
    search.add_argument("question")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--retriever", choices=RETRIEVERS, default="semantic")
    search.add_argument("--include-superseded", action="store_true")

    ask = commands.add_parser("ask", help="answer a question with cited sources")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="print the full result as JSON")
    ask.add_argument("--include-superseded", action="store_true")

    evaluate = commands.add_parser("evaluate", help="run the evaluation set")
    evaluate.add_argument("--dataset", type=Path, default=DEFAULT_EVAL_SET)
    evaluate.add_argument("--output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    evaluate.add_argument(
        "--retrievers",
        type=_choices,
        default=list(RETRIEVERS),
        help="retrievers compared on retrieval metrics (comma-separated)",
    )
    evaluate.add_argument(
        "--answer-retrievers",
        type=_choices,
        default=["semantic"],
        help="retrievers used for the answer evaluation (comma-separated)",
    )
    evaluate.add_argument("--retrieval-only", action="store_true", help="no model call")
    evaluate.add_argument("--limit", type=int, default=0, help="only the first N cases")
    evaluate.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="seconds to wait between model calls (free-tier rate limits)",
    )
    return parser


COMMANDS = {
    "ingest": cmd_ingest,
    "documents": cmd_documents,
    "rebuild": cmd_rebuild,
    "search": cmd_search,
    "ask": cmd_ask,
    "evaluate": cmd_evaluate,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        configure_logging("DEBUG" if args.verbose else "WARNING")
        return COMMANDS[args.command](args, settings)
    except AssistantError as exc:
        print(f"Error ({exc.code}): {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
