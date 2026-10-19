# Enterprise AI Knowledge Assistant

[![CI](https://github.com/Thierry-Daconceicao-Alvares/enterprise-ai-knowledge-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Thierry-Daconceicao-Alvares/enterprise-ai-knowledge-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A retrieval-augmented generation (RAG) assistant that answers employees' questions from company documents (PDF, Word and text files), cites the file, page and section of every source, and says so when the documents do not contain the answer. It is measured on a labelled evaluation set of 62 cases, including unanswerable questions, contradictory documents, outdated versions and prompt-injection attempts.

![Demo: an answer with its sources, a question the documents do not cover, a contradiction between two documents, a hidden prompt injection and an outdated version](docs/images/demo.gif)

**Measured on 62 labelled cases with the real Gemini API:** 57 passed (92%); none of the 7 questions the documents do not cover was answered anyway; no injected instruction followed and no outdated version cited; every citation pointed to a relevant section; about 1.2 seconds and $0.0008 per question. [Results and failure analysis](#13-evaluation-methodology-and-results).

<details>
<summary><b>Screenshots</b></summary>

| | |
|---|---|
| ![Answer with its sources](docs/images/02-answer.png)<br>**Answer with sources**: the reference question, answered from two documents; each statement cites its passages | ![Cited page](docs/images/04-source-page.png)<br>**Checking a source**: the passage, its file, section and relevance, and the rendered PDF page |
| ![Not in the documents](docs/images/05-not-in-documents.png)<br>**Not in the documents**: the assistant says so instead of guessing | ![Contradiction](docs/images/06-contradiction.png)<br>**Contradiction**: two current documents disagree; both values are shown and the conflict is flagged |
| ![Hidden injection](docs/images/07-hidden-injection.png)<br>**Hidden injection**: a supplier brochure tells AI assistants to ignore their rules; the answer follows Nexora's policy | ![Outdated version](docs/images/08-outdated-version.png)<br>**Outdated version**: only the current password standard is used, not the 2021 version |

</details>

> **Synthetic data.** The client, Nexora Industries, is fictional. All 32 documents were written for this project; no real company or confidential information is used.

## Contents

1. [Project overview](#1-project-overview)
2. [Business problem](#2-business-problem)
3. [Fictional client context](#3-fictional-client-context)
4. [Key features](#4-key-features)
5. [Architecture](#5-architecture)
6. [Technology stack](#6-technology-stack)
7. [How RAG works in this application](#7-how-rag-works-in-this-application)
8. [Installation](#8-installation)
9. [Configuration](#9-configuration)
10. [Ingesting documents](#10-ingesting-documents)
11. [Asking questions](#11-asking-questions)
12. [Testing](#12-testing)
13. [Evaluation methodology and results](#13-evaluation-methodology-and-results)
14. [Security considerations](#14-security-considerations)
15. [Deployment](#15-deployment)
16. [Known limitations](#16-known-limitations)
17. [Future improvements](#17-future-improvements)
18. [What this project demonstrates](#18-what-this-project-demonstrates)

## 1. Project overview

The assistant turns a folder of internal documents into a question-answering tool:

- **Ingestion**: files are validated, their text is extracted page by page, split into passages and indexed with semantic embeddings in a local vector database (Chroma).
- **Answering**: for each question, the most relevant passages of the *current* documents are retrieved and given to a language model (Gemini) with strict rules. The model answers in short statements, each tied to the passages that support it.
- **Verification**: the application checks every statement before showing it. A statement that cites no passage given to the model is removed, a number absent from its sources is flagged, and an answer with no verified support is withheld.
- **Interfaces**: a web application (Streamlit), a command-line tool and an HTTP API (FastAPI).
- **Evaluation**: a labelled set of 62 cases in 10 categories, scored automatically on retrieval quality, answer correctness, citation correctness, abstention, injection resistance, latency and cost.

This is a working prototype and a portfolio project, not a production system: see [Security considerations](#14-security-considerations) and [Known limitations](#16-known-limitations).

## 2. Business problem

Industrial companies keep their rules in many documents: purchasing policies, approval workflows, supplier procedures, IT guides, HR policies, quality procedures and operating manuals. Employees lose time looking for the right document, find outdated copies, and get different answers from different colleagues.

A useful assistant must do more than produce fluent text. In this context a wrong approval threshold or an outdated procedure is worse than no answer. The requirements are therefore:

- answers grounded in the company's documents, never in general knowledge;
- sources that a person can check in a few seconds (file, page, section);
- an explicit "the documents do not say" when the evidence is missing;
- only current versions of a procedure, and visible conflicts when two documents disagree;
- resistance to instructions hidden in documents;
- measured quality, so that the business can decide whether the tool is reliable enough.

## 3. Fictional client context

**Nexora Industries** (fictional) is an industrial manufacturing and distribution company headquartered in France, with about 1,200 employees, three plants, a distribution center, corporate functions, purchasing teams, IT support and HR operations.

The sample knowledge base contains **32 documents** in three formats: 16 PDF files, 10 Word files and 6 text files, from 1 to 3 pages each.

| Domain | Documents |
|---|---|
| Purchasing and approvals | Procurement policy, delegation of authority (current and 2023 versions), supplier onboarding, purchasing card, price changes, supplier code of conduct, supplier performance, compliance documents, purchase requisition workflow |
| Finance | Invoice processing and payment terms, travel and expense policy, travel booking guide |
| Ethics | Gifts, hospitality and conflicts of interest |
| IT | Service desk guide, password standard (current and 2021 versions), VPN guidelines, equipment and software requests, phishing reporting |
| HR | Working time and leave, remote work, training, new employee onboarding |
| Quality | Quality manual, nonconformance and 8D procedure, calibration of measuring equipment |
| Operations | Goods receipt, lockout/tagout, visitor and contractor access, site safety rules |
| External | A supplier brochure (Falcon Logistics) with a hidden prompt-injection paragraph in white text |

The corpus deliberately includes the difficulties of a real document base:

- **Outdated versions**: the 2023 delegation of authority is marked "Superseded"; the 2021 password policy still says "Current" but a newer version with the same document ID exists.
- **Contradictions**: the travel policy and the travel booking guide disagree on the dinner limit (€35 vs €30) and on the expense report deadline; the site access procedure and the safety handbook disagree on safety shoes for visitors.
- **Prompt injection**: the supplier brochure contains invisible text telling AI assistants to ignore their rules.
- **Real layout features**: tables, numbered lists, page headers and footers, sections that run across pages.

The Markdown sources are in [`data/sample_documents_src/`](data/sample_documents_src/) and the generated files in [`data/sample_documents/`](data/sample_documents/) (rebuild them with `python scripts/build_sample_documents.py`).

## 4. Key features

| Feature | Details |
|---|---|
| Upload | PDF, DOCX and TXT, with checks on extension, size (10 MB), page count (300), file signature, ZIP content of Word files, encoding |
| Extraction | Text per PDF page (PDFium); Word paragraphs, lists and tables in reading order; removal of running headers, footers and page numbers |
| Metadata | Title, document ID, version, effective date, status and owner read from the document header when present |
| Chunking | Sections detected from numbered or Word headings; passages of at most 1,000 characters with 150 characters of overlap (configurable); a passage never spans two sections; page ranges tracked |
| Index | Chroma with cosine similarity, persisted on disk; one collection per embedding model; no duplicates on repeated uploads; rebuild command |
| Versions | Superseded documents (declared, or older version of the same document ID) are kept but excluded from answers by default |
| Retrieval | Semantic search with `gemini-embedding-001` (768 dimensions); BM25 keyword baseline and a hybrid mode for comparison |
| Answers | `gemini-3.5-flash-lite` with a JSON schema: status, statements, support type (explicit or inferred), source numbers, note |
| Verification | Citations checked against the passages given to the model; unsupported statements removed; numbers checked against sources; leak detection |
| Abstention | No model call when nothing is indexed or relevant; "not enough evidence" and "clarification needed" statuses |
| Citations | File, page or page range, section, relevance score, passage text, original file download and rendered PDF page |
| Interfaces | Streamlit web app, command line (`knowledge-assistant`), HTTP API (`/ask`, `/documents`) |
| Evaluation | 62 labelled cases, deterministic scoring, Markdown and JSON reports, comparison with BM25 |
| Engineering | 227 automated tests (offline), GitHub Actions CI, Docker image, structured JSON logs without document content |

## 5. Architecture

```mermaid
flowchart LR
    subgraph ingest["Ingestion"]
        direction TB
        U["Upload: PDF, DOCX, TXT"] --> V["Validate: type, size, signature"]
        V --> X["Extract text page by page"]
        X --> C["Clean: headers, footers, spaces"]
        C --> K["Sections and chunks with overlap"]
        K --> F["Flag injection-like text"]
        F --> E1["Embed passages"]
    end
    subgraph store["Storage"]
        DB[("Chroma: vectors, text, metadata")]
        FS[("Original files")]
    end
    subgraph answer["Question answering"]
        direction TB
        Q["Question"] --> QV["Validate"]
        QV --> E2["Embed question"]
        E2 --> R["Top-k passages, current versions only"]
        R --> P["Prompt: rules + numbered, escaped excerpts"]
        P --> L["Gemini: JSON statements with sources"]
        L --> G["Verify citations and numbers"]
        G --> A["Answer with file, page, section"]
    end
    E1 --> DB
    V --> FS
    DB --> R
    FS --> A
```

| Module | Responsibility |
|---|---|
| [`extraction.py`](src/knowledge_assistant/extraction.py) | File validation, text extraction per page, cleaning, metadata |
| [`chunking.py`](src/knowledge_assistant/chunking.py) | Sections, chunks with overlap, page ranges, text to embed |
| [`embeddings.py`](src/knowledge_assistant/embeddings.py) | Gemini embeddings (documents and queries), offline hashing embedder for tests |
| [`vector_store.py`](src/knowledge_assistant/vector_store.py) | Chroma collection, document list, version resolution |
| [`ingestion.py`](src/knowledge_assistant/ingestion.py) | Ingestion pipeline, duplicates, replacement, rebuild, original files |
| [`retrieval.py`](src/knowledge_assistant/retrieval.py) | Semantic, BM25 and hybrid retrieval |
| [`prompts.py`](src/knowledge_assistant/prompts.py) | System prompt and evidence formatting |
| [`llm.py`](src/knowledge_assistant/llm.py), [`gemini.py`](src/knowledge_assistant/gemini.py) | Structured generation, retries, error translation |
| [`guardrails.py`](src/knowledge_assistant/guardrails.py) | Question validation, injection patterns, number checks, canary |
| [`assistant.py`](src/knowledge_assistant/assistant.py) | The question-answering pipeline and answer verification |
| [`evaluation.py`](src/knowledge_assistant/evaluation.py) | Evaluation set, metrics and reports |
| [`config.py`](src/knowledge_assistant/config.py), [`errors.py`](src/knowledge_assistant/errors.py), [`logging_setup.py`](src/knowledge_assistant/logging_setup.py) | Settings, error types, structured logs |
| [`app/streamlit_app.py`](app/streamlit_app.py), [`api.py`](src/knowledge_assistant/api.py), [`cli.py`](src/knowledge_assistant/cli.py) | User interfaces |

The [technical design document](docs/technical-design.md) explains the data flow, the alternatives considered and the risks.

## 6. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12+ | Standard for AI applications; typed, tested modules |
| Generation | Gemini API, `gemini-3.5-flash-lite` | Free tier for a public demo, JSON schema output, multilingual, low cost on the paid tier |
| Embeddings | `gemini-embedding-001`, 768 dimensions | Multilingual (French questions on English documents), separate task types for queries and documents |
| Vector store | Chroma (embedded, persistent) | No server to operate, metadata filters, enough for thousands of documents |
| PDF | pypdfium2 (Chrome's PDF engine) | Accurate word spacing in tables, page-level text, page rendering for the interface |
| Word | python-docx | Paragraphs, list styles and tables in document order |
| Baseline | rank-bm25 | Simple keyword ranking to compare against semantic search |
| Interface | Streamlit | Fast to build, easy for a reviewer to use |
| API | FastAPI | Typed HTTP endpoints for integration with other tools |
| Quality | pytest, ruff, GitHub Actions | 227 offline tests, lint and format checks on every push |
| Packaging | Docker, Streamlit Community Cloud | Reproducible run and a public demo |

The model provider is isolated behind small interfaces (`AnswerGenerator`, `Embedder`), so another provider (OpenAI, Azure OpenAI, a local model) can be added without changing the pipeline. Dependency versions are pinned in [`pyproject.toml`](pyproject.toml) and reviewed in [docs/dependencies.md](docs/dependencies.md).

## 7. How RAG works in this application

Retrieval-augmented generation means that the model does not answer from memory: the application first finds the relevant passages, then asks the model to answer *only* from them.

1. **Validate the question**: empty, too short, too long (1,000 characters) or wordless questions are rejected before any API call.
2. **Retrieve**: the question is embedded and compared with the passages of the current documents; the 5 most similar passages are kept. The relevance score is the cosine similarity between the question and the passage.
3. **Build the prompt**: the passages are numbered [1] to [5], with their document, section, pages and status. Their text is escaped so that a document cannot close its tag and inject a fake passage or instruction. Passages flagged as possible injections are labelled untrusted.
4. **Generate**: the model returns JSON: a status (`answered`, `insufficient_evidence` or `clarification_needed`), short statements with the numbers of their sources and a support type (`explicit`, or `inferred` when it combines passages, such as comparing an amount with a threshold), and a note for gaps or conflicts.
5. **Verify**: the application keeps only the source numbers that exist, removes statements without a valid source, flags numbers that appear neither in the cited passages nor in the question, and withholds an answer that has no verified support or that contains the secret marker of the system prompt.
6. **Cite**: the application, not the model, attaches the file name, page range, section and relevance score of each cited passage. The model cannot invent a document or a page number because it never writes them.

The four cases required by the brief are distinguished as follows: a fact stated in a passage (`explicit`), an inference from one or more passages (`inferred`), an unsupported claim (removed by the verification step and shown separately), and a question the documents cannot answer (`insufficient_evidence`).

## 8. Installation

Prerequisites: Python 3.12 or newer, Git, and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) (no credit card needed).

```bash
git clone https://github.com/Thierry-Daconceicao-Alvares/enterprise-ai-knowledge-assistant.git
cd enterprise-ai-knowledge-assistant
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[ui,dev]"
cp .env.example .env                 # Windows: copy .env.example .env
# open .env and paste your key after GEMINI_API_KEY=
streamlit run app/streamlit_app.py   # opens http://localhost:8501
```

In the application, click **Load the 32 sample documents**, then ask a question or click an example.

With Docker:

```bash
docker build -t knowledge-assistant .
docker run -p 8501:8501 -e GEMINI_API_KEY=your-key -v ka-index:/app/data/index knowledge-assistant
```

**Troubleshooting**

| Symptom | Fix |
|---|---|
| "GEMINI_API_KEY is not set" | Create `.env` from `.env.example` and paste the key, then restart |
| "rate limit reached (HTTP 429)" | Per-minute free-tier limit: the application waits and retries; for the evaluation use `--delay 4` |
| "the daily quota of this API key ... is used up" | The message gives the reset time (midnight Pacific time). Until then, use a key from another Google Cloud project or enable billing |
| "The index was built with ..." | You changed the embedding model: run `knowledge-assistant rebuild --samples` |
| A PDF is rejected with "no extractable text" | It is a scanned document; OCR is not supported in this version |
| `pip` cannot install a dependency | Use Python 3.12 or 3.13 and upgrade pip (`python -m pip install -U pip`) |

## 9. Configuration

Settings are read from environment variables or a `.env` file (see [`.env.example`](.env.example)). Secrets are never stored in the code.

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | (none) | Required for embeddings and answers |
| `GENERATION_MODEL` | `gemini-3.5-flash-lite` | Model that writes the answers |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | `gemini-embedding-001` / `768` | Embedding model and vector size |
| `EMBEDDING_PROVIDER` | `gemini` | `local` uses offline word-hashing vectors (tests only, no meaning) |
| `RETRIEVER` | `semantic` | `semantic`, `hybrid` or `bm25` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Passage size and overlap, in characters |
| `TOP_K` | `5` | Passages sent to the model |
| `MIN_RELEVANCE` | `0.0` | Cosine similarity below which a passage is ignored |
| `THINKING_LEVEL` | model default | `minimal`, `low`, `medium` or `high` |
| `MAX_FILE_MB` / `MAX_PAGES` | `10` / `300` | Upload limits |
| `DATA_DIR` | `data` | Location of the index (`data/index`) and stored files (`data/storage`) |
| `PUBLIC_DEMO` | `false` | Per-visitor uploads and question limits for a public deployment |
| `REQUEST_TIMEOUT_S` / `MAX_RETRIES` | `60` / `3` | Timeout of a Gemini request; retries after a timeout, a server error or a per-minute rate limit |
| `LOG_LEVEL` / `LOG_CONTENT` | `INFO` / `false` | Logs never contain questions or documents unless `LOG_CONTENT=true` |

**Trade-offs.** Smaller chunks make retrieval more precise but give the model less context; larger chunks do the opposite and cost more tokens. Overlap keeps a rule readable when it is cut between two chunks, at the cost of some duplicated text. A higher `TOP_K` improves the chance of retrieving all the evidence for multi-document questions, but adds noise and tokens. Changing the chunking or the embedding model requires rebuilding the index; embedding the 32 sample documents (239 passages) costs a few API calls.

## 10. Ingesting documents

- **Web app**: use the sidebar to upload files, or **Load the 32 sample documents**. The sidebar shows the number of documents and passages, the document list with versions and statuses, and warnings for flagged passages.
- **Command line**:

  ```bash
  knowledge-assistant ingest --samples          # the sample documents
  knowledge-assistant ingest path/to/folder     # every PDF, DOCX and TXT file of a folder
  knowledge-assistant documents                 # list the index with statuses
  knowledge-assistant rebuild --samples         # empty the index and index everything again
  ```

- **API**: `POST /documents` with a multipart file (see [section 11](#11-asking-questions)).

Rules applied on every upload:

- the same content uploaded again, under any name, is skipped (SHA-256 of the file);
- a new content under an existing file name replaces the old version;
- a file with the same document ID as an indexed one is kept, and the older version is marked superseded;
- files that cannot be used are rejected with a reason: unsupported type, empty, too large, damaged, password-protected, or without text.

## 11. Asking questions

**Web app**: type a question or click an example. Each answer shows its status, the statements with their source numbers (and `inferred` when a statement combines passages), the cited sources with file, page, section, relevance and the passage text, the original file and, for PDFs, the rendered page. The other retrieved passages, latency, tokens and estimated cost are shown below the answer.

**Command line**:

```bash
knowledge-assistant ask "What is the procedure to approve a purchase request of more than €50,000?"
knowledge-assistant ask "Quelle est la procédure pour valider une demande d'achat supérieure à 50 000 € ?" --json
knowledge-assistant search "purchasing card limit" --retriever bm25    # passages only, no model call
```

**HTTP API** (`uvicorn knowledge_assistant.api:app`, documentation at `http://localhost:8000/docs`):

```bash
curl -F "file=@data/sample_documents/NX-PUR-004_purchasing-card-policy.docx" http://localhost:8000/documents
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What is the single transaction limit of the purchasing card?"}'
```

Errors are returned with a stable code: `invalid_question` (422), `llm_rate_limited` and `llm_quota_exhausted` (429, with a `Retry-After` header), `configuration_error` and `llm_unavailable` (503), `llm_invalid_response` (502).

## 12. Testing

```bash
pytest                  # 227 tests, offline, about 25 seconds
ruff check . && ruff format --check .
```

The tests never call a paid API. They use the real sample documents, an in-memory Chroma index with offline embeddings, a scripted model for the answer pipeline, and a local fake of the Gemini HTTP API to exercise the real SDK (request format, JSON schema, retries, rate limits and daily quotas, error handling).

| Area | What is tested |
|---|---|
| Extraction | Page numbers, metadata, removal of headers and footers, table spacing, hidden text, Word lists and tables, encodings, file name sanitising |
| Malformed files | Empty, unsupported, fake PDF, scanned PDF, truncated DOCX, ZIP bomb, binary text, size and page limits |
| Chunking | Size limit and stable IDs on every sample document, page numbers checked against the page text, page ranges, sections, overlap |
| Index | Duplicates, replacement, deletion, rebuild, persistence, embedding-model mismatch, superseded versions, per-session copies |
| Answers | Citation mapping, invalid sources, removed statements, withheld answers, number checks, abstention, clarification, prompt leaks, escaping, error propagation |
| Interfaces | API endpoints and error codes, command line, Streamlit pages (AppTest) |
| Evaluation | Labels match the corpus, scoring rules, report generation |

The CI workflow runs the tests on Python 3.12 and 3.13, builds the Docker image and checks that the container starts.

## 13. Evaluation methodology and results

The assistant is evaluated on [62 hand-written cases](eval/dataset.jsonl) in 10 categories: single and multi-document questions, questions the documents do not answer, ambiguous questions, contradictions, citations, injection attempts, outdated versions and malformed inputs. Each case names the sections that contain the answer, the key facts a correct answer must contain and the facts it must not contain. Scoring is automatic: [docs/evaluation.md](docs/evaluation.md) defines the metrics and the matching rules. The **Evaluate with Gemini** workflow runs the set against the real API and publishes the report.

**Results**: run of 19 October 2026 with `gemini-3.5-flash-lite` and `gemini-embedding-001`, 5 passages per question. Full report with every failed answer: [eval/results/2026-10-19-prompt-v1/report.md](eval/results/2026-10-19-prompt-v1/report.md).

**Retrieval** (42 questions with labelled sections):

| Retriever | Hit@1 | Hit@5 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| **Semantic** (default) | **100%** | **100%** | **96%** | **1.00** |
| BM25 keyword baseline | 83% | 95% | 79% | 0.88 |
| Hybrid (reciprocal rank fusion) | 93% | 98% | 88% | 0.95 |

**Answers** (62 cases, same model and prompt, two retrievers):

| Metric | Semantic retrieval | BM25 retrieval |
|---|---:|---:|
| Cases passed | **57 / 62 (92%)** | 48 / 62 (77%) |
| Fully correct answers (42 answerable questions) | 37 / 42 (88%) | 29 / 42 (69%) |
| Key facts present | 95% | 76% |
| Answers citing a required section | 100% | 97% |
| Questions not covered by the documents, declined | 7 / 7 | 6 / 7 |
| Injected instructions followed / superseded versions cited | 0 / 0 | 0 / 0 |
| Latency, median / 95th percentile | 1.2 s / 2.1 s | 1.0 s / 1.4 s |
| Tokens per question, prompt / output | 1,383 / 136 | 1,355 / 118 |
| Estimated cost per question (paid-tier prices) | $0.00076 | $0.00070 |

With semantic retrieval, 9 of the 10 categories pass every case; the exception is multi-document questions (2 of 7).

**Against the targets of the [discovery brief](docs/discovery-brief.md)** (semantic retrieval):

| Criterion | Target | Result | Met |
|---|---|---|---|
| Correct answers on answerable questions | at least 80% | 88% | yes |
| Answers citing a correct source | at least 90% | 100% | yes |
| Questions not covered by the documents answered anyway | at most 10% | 0% | yes |
| Injected instruction followed or superseded version cited | 0 | 0 | yes |
| Median response time | under 5 s | 1.2 s | yes |
| Cost per question on the paid tier | under $0.01 | $0.0008 | yes |

**Failure analysis** (the 5 failed cases):

- **Incomplete multi-document answers (3 cases).** For the reference question (approval of a purchase above €50,000), the model named the approvers and the ERP steps but left out the three written quotes. The rule was in a retrieved passage ("at least three written quotes from €5,000; from €50,000, also the bid comparison"), but the model kept only the clause that starts at €50,000. The French version of the question failed the same way, and a third answer did not say that contractor services need a purchase order.
- **Missing verdict (1 case).** "Can I spend my remote work days working from a café, and what rules apply to its public Wi-Fi?" got the right rules, but not the "No" that the question calls for.
- **Refused by the provider (1 case).** Gemini's safety filter refused a question about a phishing incident ("An employee clicked a phishing link and typed their password..."), a false positive on a legitimate IT-support question. The application now explains such refusals instead of showing a generic error.
- **Correction**: the [system prompt](src/knowledge_assistant/prompts.py) now asks for every requirement that applies to the case, including those found in another passage, and for an opening "Yes" or "No" on yes/no questions, even when they are combined with another question. The next run will measure the effect on all 62 cases, to check that the change fixes these failures without breaking others.

**What the comparison shows**:

- **Retrieval drives answer quality.** With keyword search, the passages needed were often missing from the top 5: the model then said it lacked evidence (7 answerable questions) rather than guessing. The supplier brochure with the hidden injection ranked first for three questions with BM25, yet no injected instruction was followed.
- **Relevance scores cannot replace abstention.** The best passage of answerable questions scores between 0.69 and 0.80, and that of questions the documents do not cover between 0.59 and 0.71. The ranges overlap, so a similarity threshold would reject good questions: `MIN_RELEVANCE` stays at 0, and abstention relies on the prompt rules and the verification of each statement (7 of 7 declined).
- **Cost is not a constraint at this scale**: 1,000 questions cost under $1 at paid-tier prices.

Reproduce the results (the model is not deterministic: two runs can differ by a few cases):

```bash
knowledge-assistant evaluate --retrieval-only                         # retrieval only, no generation call
knowledge-assistant evaluate --answer-retrievers semantic,bm25 --delay 4
```

## 14. Security considerations

| Control | Status |
|---|---|
| Instructions inside documents | Mitigated, not eliminated: excerpts are escaped and labelled as untrusted data, injection-like passages are flagged at ingestion, the system prompt forbids following them, and the evaluation includes a document with a hidden injection |
| Requests to reveal the system prompt | A random marker in the system prompt; an answer containing it is withheld |
| Fabricated citations | The model cites passage numbers only; the application maps them to files and pages and removes invalid ones |
| Malformed or malicious files | Extension, size, page count, file signature and decompressed size are checked before parsing; file names are sanitised |
| Secrets | API key in environment variables or `.env` (ignored by Git); GitHub and Streamlit secrets for automation and the demo |
| Logs | Structured JSON without questions, answers or document text, unless `LOG_CONTENT=true` |
| Authentication and authorisation | **Not implemented.** Anyone who can reach the application can read every indexed document |
| Document-level permissions | **Not implemented** |
| Encryption at rest | **Not implemented** by the application; depends on the host disk |
| Retention | **Not implemented**: files stay until deleted; demo sessions are deleted after one hour of inactivity |
| Tenant isolation | **Not implemented**; public demo uploads are isolated per browser session, in memory only |
| Data sent to the model provider | Questions and retrieved passages are sent to the Gemini API. On the free tier, Google may use them to improve its products: do not use confidential data with a free-tier key |

This prototype does not meet enterprise security requirements and should not be used with real confidential documents. [docs/security.md](docs/security.md) describes the threat model and the controls needed before a production rollout.

## 15. Deployment

- **Local**: see [Installation](#8-installation).
- **Docker**: the image runs as a non-root user, exposes port 8501 and has a health check; mount a volume on `/app/data/index` to keep the index.
- **Public demo**: Streamlit Community Cloud, with `PUBLIC_DEMO=true`: the sample documents are preloaded, each visitor gets a private in-memory copy of the index for uploads, and the number of questions and uploads per session is limited.
- **Production plan**: container platform behind a corporate identity provider, managed vector database, paid-tier or enterprise model endpoint with a data processing agreement, monitoring and evaluation on every change. See [docs/deployment.md](docs/deployment.md).

## 16. Known limitations

- **No access control**: every user sees every document (see security).
- **Scanned PDFs** are rejected: there is no OCR.
- **Word documents have no page numbers** (pagination depends on the viewer); their citations point to sections.
- **Tables in PDFs** are extracted as lines of text; the row structure is approximate.
- **Section detection** relies on numbered headings or Word heading styles; other documents are split by size only.
- **Version detection** relies on the header block ("Document ID", "Version", "Status"); documents without it are never marked superseded.
- **Injection detection** is a list of patterns: it flags known phrasings, not every attack. The real protection comes from treating passages as data and verifying the output.
- **Number checks** catch invented figures, not wrong roles or wrong reasoning.
- **Evaluation**: 62 cases on a synthetic corpus written by the same author as the questions; key facts are matched as text, so a correct paraphrase can be scored as a miss. The cases to review are listed in the report.
- **Scale**: tested with tens of documents; the document list is rebuilt from the index metadata, which is fine for thousands of passages, not millions.
- **Latency and cost** depend on the Gemini API. The free tier has per-minute and daily limits: when the daily quota is used up, the demo says when it will answer again (midnight Pacific time).

## 17. Future improvements

1. Authentication with the corporate identity provider (SSO) and role-based access control.
2. Document-level permissions applied as a metadata filter at retrieval time.
3. OCR for scanned documents and table-aware PDF extraction.
4. Reranking of the retrieved passages, and a hybrid default if the evaluation confirms the gain.
5. Incremental synchronisation with the document management system (new versions, deletions).
6. User feedback on answers, and review of low-confidence answers.
7. Monitoring dashboard: usage, abstention rate, latency, cost, errors.
8. Larger evaluation set written with domain experts, and model-graded checks calibrated against human review.
9. Managed vector database and multi-tenant isolation for several business units.

## 18. What this project demonstrates

- Translating a business problem (inconsistent answers about internal rules) into a scoped MVP with explicit acceptance criteria.
- Building a complete RAG application: document processing, chunking, embeddings, vector search, prompt design, structured generation and citation handling.
- Treating hallucination as an engineering problem: answers are verified by code, not trusted because of a prompt.
- Handling the messy parts of real document bases: outdated versions, contradictions, hidden instructions, malformed files.
- Measuring quality with a labelled evaluation set, comparing against a baseline and reporting failures honestly.
- Professional engineering practices: typed modules, 227 automated tests, CI, Docker, configuration and secrets management, structured logging, documentation.
- Customer-facing work of a Forward Deployed Engineer: [discovery brief](docs/discovery-brief.md), [technical design](docs/technical-design.md), [implementation plan](docs/implementation-plan.md), [demo script](docs/demo-script.md).

## Project structure

```
app/streamlit_app.py            web interface
src/knowledge_assistant/        application package (pipeline, API, command line)
data/sample_documents/          the 32 synthetic Nexora documents (PDF, DOCX, TXT)
data/sample_documents_src/      their Markdown sources
eval/dataset.jsonl              the 62 labelled evaluation cases
eval/fixtures/                  malformed files used by the evaluation
eval/results/                   published evaluation reports
scripts/                        document generator and screenshot capture
tests/                          automated tests (offline)
docs/                           design, discovery, plan, security, deployment, demo script
```

## License

MIT. The sample documents are synthetic and can be reused freely.
