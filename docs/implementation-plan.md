# Implementation plan

The MVP was built in small tasks, each with acceptance criteria and tests. The order follows the dependencies: nothing downstream was started while an upstream step was broken.

| # | Task | Depends on | Acceptance criteria | Tests |
|---|---|---|---|---|
| 1 | Synthetic corpus | - | 32 documents in PDF, DOCX and TXT, with versions, contradictions and an injection; regenerated from Markdown sources | Metadata and page tests on every file |
| 2 | Settings and errors | - | All parameters from environment variables; invalid values explained; one error type per failure with a stable code | Configuration tests |
| 3 | File validation and extraction | 2 | Supported types only; size, page and decompression limits; text per PDF page; Word order kept; headers and footers removed | Extraction and malformed-file tests |
| 4 | Metadata | 3 | Title, document ID, version, effective date, status and owner read from the header when present | Metadata tests on PDF, DOCX and TXT |
| 5 | Chunking | 3 | Chunks within the size limit, no chunk across sections, overlap between consecutive chunks, correct page ranges | Chunking tests on all 32 documents, page check against the page text |
| 6 | Embeddings | 2 | Gemini document and query embeddings, batched, normalised; offline embedder for tests | SDK test against the fake API |
| 7 | Vector store | 5, 6 | Upsert, delete, rebuild, persistence; refuse an index built with another model; document list with statuses | Index tests |
| 8 | Ingestion service | 3-7 | Duplicates skipped, same file name replaced, rejected files reported, originals stored | Ingestion tests |
| 9 | Version resolution | 7 | Declared and implicit superseded versions excluded from retrieval by default | Version tests |
| 10 | Retrieval | 7, 9 | Semantic top-k with cosine similarity; BM25 and hybrid for comparison | Retrieval tests |
| 11 | Prompt and generation | 10 | Numbered, escaped excerpts; JSON schema; retries and error translation | Prompt and SDK tests |
| 12 | Verification | 11 | Invalid sources removed, unsupported statements removed, numbers checked, leaks withheld | Assistant tests with a scripted model |
| 13 | Interfaces | 8, 12 | Streamlit (upload, status, answers, citations, page view, errors), CLI and API | AppTest, API and CLI tests |
| 14 | Evaluation | 12 | 62 labelled cases in 10 categories; labels validated against the corpus; report with failures | Evaluation tests |
| 15 | Packaging | 13 | CI on every push; Docker image builds and starts; public demo mode | CI, Docker health check |
| 16 | Documentation | all | README with the 18 required sections, design, security, deployment, demo script | Review |

## Definition of done

A task is done when:

- the code is in a module with type hints and docstrings on its public functions;
- its tests pass offline, without an API key, and CI is green;
- errors are handled with a clear message and a stable error code;
- no secret or document content is logged or committed;
- the README or the design documents describe the behaviour and its limits;
- for answer-quality changes: the evaluation is run again and the report is updated with the actual results, including failures.

## What failed during development and how it was fixed

| Problem | How it was found | Fix |
|---|---|---|
| Words glued together in PDF table cells ("ChiefFinancial Officer") with pypdf | Reading the extracted text of the delegation of authority | Switched to PDFium (pypdfium2) and added a regression test |
| Running headers repeated in the middle of passages | Inspection of multi-page chunks | Conservative header and footer removal, with tests |
| The last item of a Word list merged with the next paragraph | Chunk inspection | Word paragraphs separated by blank lines at extraction |
| A numbered list mistaken for section headings | Test on the compliance documents procedure | Heading numbers must follow the previous heading |
| A forbidden fact ("90 days") present in a current document, which would have penalised correct answers | A test checking the labels against the corpus | Forbidden facts checked only against the labelled sections |
| numpy 2.5 not available for Python 3.11 | First CI run | Python 3.12+ required |
