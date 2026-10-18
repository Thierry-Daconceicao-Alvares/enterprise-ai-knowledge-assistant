# Dependency and license review

Versions are pinned in `pyproject.toml` so that installs are reproducible. Licenses were read from the installed package metadata on 2026-10-18.

## Runtime

| Package | Version | License | Use |
|---|---|---|---|
| chromadb | 1.5.9 | Apache-2.0 | Vector store |
| google-genai | 2.31.0 | Apache-2.0 | Gemini API client (embeddings, generation) |
| pypdfium2 | 5.14.0 | Apache-2.0 or BSD-3-Clause (PDFium: BSD-3-Clause) | PDF text extraction and page rendering |
| python-docx | 1.2.0 | MIT | Word extraction |
| rank-bm25 | 0.2.2 | Apache-2.0 | BM25 baseline |
| snowballstemmer | 3.1.1 | BSD-3-Clause | Word stems for BM25 and fact matching |
| numpy | 2.5.4 | BSD-3-Clause and others | Vectors |
| pydantic | 2.14.1 | MIT | Answer schema, API models |
| fastapi | 0.146.0 | MIT | HTTP API |
| uvicorn | 0.54.0 | BSD-3-Clause | API server |
| python-multipart | 0.0.32 | Apache-2.0 | File uploads in the API |
| httpx | 0.28.1 | BSD-3-Clause | HTTP client used by the SDK |
| python-dotenv | 1.2.4 | BSD-3-Clause | `.env` file |
| streamlit | 1.66.0 | Apache-2.0 | Web interface (`ui` extra) |

`chromadb` brings transitive dependencies such as `onnxruntime` (MIT), which this application does not use directly.

## Development and automation only

| Package | Version | License | Use |
|---|---|---|---|
| pytest | 9.1.1 | MIT | Tests |
| ruff | 0.17.1 | MIT | Lint and format |
| httpx2 | 2.14.0 | BSD-3-Clause | Test client for FastAPI |
| fpdf2 | 2.8.9 | LGPL-3.0-only | Generates the sample PDFs; not imported by the application |
| playwright | 1.64.0 | Apache-2.0 | Screenshot capture in the evaluation workflow |

## Assessment

- All runtime dependencies use permissive licenses (MIT, BSD, Apache-2.0), compatible with the MIT license of this project.
- `fpdf2` is LGPL-3.0: it is only used by `scripts/build_sample_documents.py` to generate the sample documents and is not distributed with the application or included in the Docker image.
- The generated PDFs embed a subset of the DejaVu Sans font, whose license allows embedding and redistribution.
- External service: the Gemini API is used under Google's API terms. On the free tier, Google may use submitted content to improve its products; real company data requires a paid tier and a data processing agreement.
- To check for known vulnerabilities before a release: `pip install pip-audit && pip-audit`.
