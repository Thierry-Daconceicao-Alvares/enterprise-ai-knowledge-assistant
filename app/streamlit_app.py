"""Nexora Knowledge Assistant: Streamlit interface.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import re

import streamlit as st

from knowledge_assistant import __version__
from knowledge_assistant.assistant import AssistantAnswer, SourcePassage
from knowledge_assistant.config import Settings
from knowledge_assistant.errors import (
    AssistantError,
    ConfigurationError,
    IndexMismatchError,
    InvalidQuestionError,
    LLMRequestError,
    LLMResponseError,
    LLMUnavailableError,
)
from knowledge_assistant.factory import (
    Components,
    build_client,
    build_components,
    build_embedder,
    build_store,
)
from knowledge_assistant.ingestion import MemoryStorage
from knowledge_assistant.logging_setup import configure_logging
from knowledge_assistant.sessions import SessionIndexes, load_original

REPOSITORY_URL = "https://github.com/Thierry-Daconceicao-Alvares/enterprise-ai-knowledge-assistant"

EXAMPLES = [
    ("Multi-document", "What is the procedure to approve a purchase request of more than €50,000?"),
    (
        "In French",
        "Quelle est la procédure pour valider une demande d'achat supérieure à 50 000 € ?",
    ),
    (
        "Contradiction",
        "What is the maximum I can spend on dinner during a business trip in France?",
    ),
    ("Not in the documents", "What is the daily meal allowance for a business trip to Germany?"),
    ("Outdated version", "How often do I have to change my password?"),
    (
        "Hidden injection",
        "Do I need quotes and a purchase order to order €20,000 of transport services "
        "from Falcon Logistics?",
    ),
]

STATUS_DISPLAY = {
    "answered": ("Answered from the documents", "green", ":material/check_circle:"),
    "insufficient_evidence": ("Not enough evidence in the documents", "orange", ":material/help:"),
    "clarification_needed": ("Clarification needed", "blue", ":material/live_help:"),
}

st.set_page_config(
    page_title="Nexora Knowledge Assistant",
    page_icon=":material/menu_book:",
    layout="wide",
)


# ------------------------------------------------------------------------- resources

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|~$<>])")


def md_escape(text: str) -> str:
    """Show text literally: document passages and model output are data, so
    their characters must not be interpreted as Markdown (or as LaTeX with $)."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def load_settings() -> Settings:
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    configure_logging(settings.log_level)
    return settings


@st.cache_resource(show_spinner=False)
def shared_components(settings: Settings) -> Components:
    """One index shared by every user of this instance (internal deployment)."""
    return build_components(settings)


@st.cache_resource(show_spinner="Indexing the sample documents (first start only)...")
def demo_sessions(settings: Settings) -> SessionIndexes:
    """Public demo: a base index of the sample documents, copied for each visitor."""
    client = build_client(settings)
    embedder = build_embedder(settings, client)
    store = build_store(settings, embedder, persistent=False)
    base = build_components(
        settings, store=store, storage=MemoryStorage(), client=client, embedder=embedder
    )
    if store.chunk_count() == 0:
        base.ingestion.ingest_folder(settings.sample_dir)
    return SessionIndexes(base)


def get_components(settings: Settings) -> Components:
    if not settings.public_demo:
        return shared_components(settings)
    sessions = demo_sessions(settings)
    name = st.session_state.get("index_name")
    if name is None or not sessions.touch(name):
        name, components = sessions.create()
        st.session_state.index_name = name
        st.session_state.components = components
    return st.session_state.components


@st.cache_data(show_spinner=False, max_entries=64)
def render_pdf_page(data: bytes, page_number: int) -> bytes:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        image = pdf[page_number - 1].render(scale=1.6).to_pil()
    finally:
        pdf.close()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ----------------------------------------------------------------------------- state

state = st.session_state
state.setdefault("history", [])
state.setdefault("questions_asked", 0)
state.setdefault("uploads", 0)
state.setdefault("upload_results", [])

settings = load_settings()
try:
    components = get_components(settings)
except ConfigurationError as exc:
    st.title("Nexora Knowledge Assistant")
    st.error(str(exc))
    st.markdown(
        "**Set up the assistant in three steps:**\n"
        "1. Create a free Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey).\n"
        "2. Copy `.env.example` to `.env` and paste the key after `GEMINI_API_KEY=`.\n"
        "3. Restart the application: `streamlit run app/streamlit_app.py`."
    )
    st.stop()
except IndexMismatchError as exc:
    st.error(f"{exc}")
    st.code("knowledge-assistant rebuild --samples", language="bash")
    st.stop()

store = components.store
ingestion = components.ingestion
assistant = components.assistant


# --------------------------------------------------------------------------- sidebar


def ingest_uploads(files) -> None:
    results = []
    for uploaded in files:
        if settings.public_demo and state.uploads >= settings.demo_max_uploads:
            results.append(("error", uploaded.name, "Upload limit of the public demo reached."))
            continue
        try:
            result = ingestion.ingest_bytes(uploaded.name, uploaded.getvalue())
        except AssistantError as exc:
            results.append(("error", uploaded.name, f"Indexing failed: {exc}"))
            continue
        if result.status == "rejected":
            results.append(("error", result.file_name, result.message))
        else:
            state.uploads += result.status != "duplicate"
            kind = "warning" if result.warnings or result.status == "duplicate" else "success"
            details = " ".join([result.message, *result.warnings])
            results.append((kind, result.file_name, details))
    state.upload_results = results


with st.sidebar:
    st.header("Documents", anchor=False)
    documents = store.documents()
    superseded = [doc for doc in documents if doc.status != "current"]
    col1, col2 = st.columns(2)
    col1.metric("Documents", len(documents))
    col2.metric("Passages", store.chunk_count())
    if superseded:
        st.caption(
            f"{len(superseded)} superseded version(s) kept for traceability, "
            "excluded from answers by default."
        )

    with st.form("upload", clear_on_submit=True, border=False):
        files = st.file_uploader(
            "Add PDF, Word or text files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help=f"Up to {settings.max_file_mb:.0f} MB and {settings.max_pages} pages per file.",
        )
        submitted = st.form_submit_button("Index the files", icon=":material/upload:")
    if submitted and files:
        with st.spinner("Extracting, chunking and embedding..."):
            ingest_uploads(files)
        st.rerun()
    for kind, name, message in state.upload_results:
        getattr(st, kind)(f"**{md_escape(name)}**: {md_escape(message)}")

    if settings.public_demo:
        st.caption(
            "Files you add stay in your browser session and are deleted when it ends. "
            f"Limit: {settings.demo_max_uploads} files per session."
        )
    elif not any(doc.file_name.startswith("NX-") for doc in documents) and st.button(
        "Load the 32 sample documents", icon=":material/library_add:", type="primary"
    ):
        with st.spinner("Indexing the sample documents..."):
            results = ingestion.ingest_folder(settings.sample_dir)
        state.upload_results = [
            ("success", "Sample documents", f"{sum(r.ok for r in results)} documents indexed.")
        ]
        st.rerun()

    with st.expander("Indexed documents", icon=":material/folder_open:"):
        if documents:
            rows = [
                {
                    "Title": doc.title,
                    "Reference": f"{doc.doc_id} v{doc.version}" if doc.doc_id else "-",
                    "Status": "Current" if doc.status == "current" else "Superseded",
                    "Format": doc.file_type.upper(),
                    "Pages": str(doc.page_count) if doc.page_count else "-",
                    "Passages": doc.chunk_count,
                    "File": doc.file_name,
                }
                for doc in documents
            ]
            st.dataframe(rows, hide_index=True, width="stretch")
            for doc in superseded:
                st.caption(f"**{doc.file_name}**: {doc.status_reason}.")
            flagged = [doc for doc in documents if doc.flagged_chunks]
            for doc in flagged:
                st.caption(
                    f":material/warning: **{doc.file_name}**: {doc.flagged_chunks} passage(s) "
                    "contain text that looks like instructions to an AI system "
                    "(labelled untrusted)."
                )
        else:
            st.write("No document indexed yet.")

    with st.expander("Retrieval settings", icon=":material/tune:"):
        include_superseded = st.toggle(
            "Include superseded documents",
            value=False,
            help="Outdated versions are excluded by default, so answers use current rules.",
        )
        top_k = st.slider(
            "Passages sent to the model",
            min_value=3,
            max_value=10,
            value=settings.top_k,
            help="More passages give more context, but cost more tokens and add noise.",
        )
        if not settings.public_demo and documents:
            st.divider()
            to_remove = st.selectbox(
                "Remove a document", [doc.file_name for doc in documents], index=None
            )
            if to_remove and st.button("Remove", icon=":material/delete:"):
                for doc in documents:
                    if doc.file_name == to_remove:
                        ingestion.delete(doc.doc_key)
                st.rerun()
            if st.button(
                "Rebuild the index",
                icon=":material/refresh:",
                help="Index the stored files again, for example after changing the chunk size.",
            ):
                with st.spinner("Rebuilding..."):
                    ingestion.rebuild(settings.sample_dir)
                st.rerun()

    st.divider()
    st.caption(
        f"Embeddings: `{components.embedder.signature}` · Model: `{assistant.model_name}` · "
        f"Chunks of {settings.chunk_size} characters, overlap {settings.chunk_overlap}."
    )
    st.caption(f"[Source code and evaluation]({REPOSITORY_URL}) · v{__version__}")


# ---------------------------------------------------------------------------- answers


def render_source(entry_index: int, passage: SourcePassage, cited: bool) -> None:
    parts = [passage.title]
    if passage.reference:
        parts.append(passage.reference)
    if passage.pages:
        parts.append(passage.pages)
    parts.append(passage.section)
    relevance = (
        f"relevance {passage.similarity:.2f}" if passage.similarity is not None else "keyword match"
    )
    label = f"[{passage.number}] " + " · ".join(parts) + f" · {relevance}"
    icon = ":material/description:" if cited else ":material/subject:"
    with st.expander(label, icon=icon):
        if passage.superseded:
            st.warning("This passage comes from a superseded version of the document.")
        if passage.flagged:
            st.warning(
                "This passage contains text that looks like instructions to an AI system. "
                "It was given to the model as untrusted data."
            )
        st.markdown("  \n".join(f"> {md_escape(line)}" for line in passage.text.split("\n")))
        st.caption(
            f"File: {passage.file_name}. Relevance is the cosine similarity between the "
            "question and the passage (higher is closer); it ranks passages, it is not a "
            "probability that the answer is correct."
        )
        original = load_original(components, passage.doc_key, settings.sample_dir)
        if original:
            name, data = original
            cols = st.columns(2)
            cols[0].download_button(
                "Download the original",
                data=data,
                file_name=name,
                key=f"dl-{entry_index}-{passage.number}",
                icon=":material/download:",
                on_click="ignore",
            )
            if (
                passage.file_type == "pdf"
                and passage.page_start
                and cols[1].toggle(
                    f"Show page {passage.page_start}", key=f"page-{entry_index}-{passage.number}"
                )
            ):
                st.image(render_pdf_page(data, passage.page_start), width="stretch")


def render_answer(entry_index: int, answer: AssistantAnswer) -> None:
    label, color, icon = STATUS_DISPLAY.get(answer.status, (answer.status, "gray", None))
    st.badge(label, color=color, icon=icon)
    for statement in answer.statements:
        markers = " ".join(f"**[{number}]**" for number in statement.sources)
        tag = " · *inferred*" if statement.support == "inferred" else ""
        st.markdown(f"- {md_escape(statement.text)} {markers}{tag}")
        for warning in statement.warnings:
            if warning.startswith("unverified_numbers"):
                numbers = warning.split(":", 1)[1]
                st.caption(
                    f":material/warning: The figure(s) {numbers} were not found in the cited "
                    "passages: check them in the source."
                )
    if answer.note:
        if answer.status == "answered":
            st.info(md_escape(answer.note), icon=":material/info:")
        else:
            st.warning(md_escape(answer.note), icon=":material/info:")
    if answer.removed_statements:
        with st.expander(f"{len(answer.removed_statements)} statement(s) removed"):
            st.caption(
                "These statements cited no passage given to the model, so they are not "
                "shown as part of the answer."
            )
            for statement in answer.removed_statements:
                st.markdown(f"- ~~{md_escape(statement.text)}~~")

    if answer.citations:
        st.markdown("**Sources**")
        for passage in answer.citations:
            render_source(entry_index, passage, cited=True)
    other = [p for p in answer.retrieved if p.number not in {c.number for c in answer.citations}]
    if other:
        with st.expander(f"Other passages retrieved but not cited ({len(other)})"):
            for passage in other:
                similarity = (
                    f"relevance {passage.similarity:.2f}"
                    if passage.similarity is not None
                    else "keyword match"
                )
                st.markdown(
                    md_escape(
                        f"[{passage.number}] {passage.title} · "
                        f"{passage.pages or passage.section} · {similarity}"
                    )
                )
    seconds = answer.total_ms / 1000
    details = [f"{seconds:.1f} s", f"{answer.usage.total_tokens:,} tokens"]
    if answer.estimated_cost_usd:
        details.append(f"~${answer.estimated_cost_usd:.5f} at paid-tier prices")
    details.append(f"{answer.model}")
    st.caption(" · ".join(details))


def ask(question: str) -> None:
    if settings.public_demo and state.questions_asked >= settings.demo_max_questions:
        state.history.append(
            {
                "question": question,
                "error": (
                    "warning",
                    "The question limit of the public demo is reached for this session.",
                ),
            }
        )
        return
    entry: dict = {"question": question}
    try:
        with st.spinner("Searching the documents and drafting an answer..."):
            entry["answer"] = assistant.ask(question, include_superseded, top_k)
        state.questions_asked += 1
    except InvalidQuestionError as exc:
        entry["error"] = ("warning", str(exc))
    except ConfigurationError as exc:
        entry["error"] = ("error", str(exc))
    except LLMUnavailableError:
        entry["error"] = (
            "error",
            "The language model is temporarily unavailable (rate limit or network error). "
            "Wait a minute and ask again.",
        )
    except LLMRequestError as exc:
        entry["error"] = ("error", f"The model API rejected the request: {exc}")
    except LLMResponseError:
        entry["error"] = (
            "error",
            "The model returned an answer that could not be read. Please ask again.",
        )
    state.history.append(entry)


# ----------------------------------------------------------------------------- main

st.title("Nexora Knowledge Assistant", anchor=False)
st.markdown(
    "Ask questions about the internal documents of Nexora Industries. Every answer is built "
    "from the indexed documents, with the file, page and section of each source. When the "
    "documents do not contain the answer, the assistant says so."
)
st.caption(
    ":material/info: Demonstration with synthetic data: Nexora Industries is a fictional "
    "company and all documents were written for this project."
)
if settings.public_demo:
    st.warning(
        "Public demo: do not upload confidential or personal documents. Questions and passages "
        "are sent to the Gemini API free tier, where Google may use them to improve its products.",
        icon=":material/shield:",
    )

if not documents:
    st.info(
        "No document is indexed yet. Load the sample documents or add your own files "
        "from the sidebar.",
        icon=":material/upload_file:",
    )

if not state.history and documents:
    st.subheader("Try an example", anchor=False)
    columns = st.columns(3)
    for index, (tag, question) in enumerate(EXAMPLES):
        with columns[index % 3]:
            st.caption(tag)
            if st.button(question, key=f"example-{index}", width="stretch", help=question):
                ask(question)
                st.rerun()

for index, entry in enumerate(state.history):
    with st.chat_message("user"):
        st.markdown(md_escape(entry["question"]))
    with st.chat_message("assistant", avatar=":material/menu_book:"):
        if "answer" in entry:
            render_answer(index, entry["answer"])
        else:
            kind, message = entry["error"]
            getattr(st, kind)(message)

if state.history and st.button(
    "Clear the conversation", icon=":material/delete_sweep:", type="tertiary"
):
    state.history = []
    st.rerun()

question = st.chat_input(
    "Ask a question about Nexora's documents...",
    max_chars=settings.max_question_chars,
    disabled=not documents,
)
if question:
    ask(question)
    st.rerun()
