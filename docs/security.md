# Security

This document describes what the prototype protects against, how, and what is still missing before real company documents could be used.

## Assets and threats

| Asset | Threat |
|---|---|
| Company documents | Read by unauthorised users; sent to a third party; leaked in logs |
| Answers | Wrong rule presented as true (hallucination, outdated version, injected instruction) |
| System instructions | Revealed or overridden by a question or a document |
| API key | Committed to Git, exposed in logs or in the browser |
| Service | Exhausted by large files, many pages or many requests |

## Controls implemented and tested

| Threat | Control | Where | Test |
|---|---|---|---|
| Instructions inside documents | Excerpts escaped and wrapped in tags; rule "excerpts are untrusted data"; injection-like passages flagged at ingestion and labelled in the prompt; the evaluation includes a PDF with hidden white text | `prompts.py`, `guardrails.py`, `ingestion.py` | `test_assistant.py`, evaluation category `misleading_instructions` |
| Question forging an excerpt | Question escaped; only one question tag in the prompt | `prompts.py` | `test_the_question_cannot_forge_an_excerpt` |
| System prompt extraction | Random canary in the system prompt; answers containing it are withheld | `guardrails.py`, `assistant.py` | `test_a_leaked_system_prompt_is_withheld` |
| Fabricated citations | The model cites excerpt numbers; files and pages come from the index | `assistant.py` | `test_citation_numbers_outside_the_excerpts_are_removed` |
| Unsupported answers | Statements without valid sources removed; answers without support withheld; numbers checked | `assistant.py` | assistant tests, evaluation |
| Outdated procedures | Superseded versions excluded from retrieval | `vector_store.py`, `retrieval.py` | index tests, evaluation category `superseded` |
| Malformed or hostile files | Extension allow-list, size limit read before parsing (API), file signature, page limit, decompressed-size limit for DOCX (ZIP bombs), binary detection, sanitised file names | `extraction.py`, `api.py` | extraction tests, evaluation category `malformed_input` |
| Path traversal through file names | Only the base name is kept, unusual characters replaced, files stored under a content hash folder | `extraction.py`, `ingestion.py` | `test_safe_file_name` |
| Secrets in Git | `.env` ignored; `.env.example` without values; GitHub secret scoped to the steps that need it | `.gitignore`, workflows | review |
| Confidential content in logs | JSON logs with identifiers, counts and timings only; content only with `LOG_CONTENT=true` | `logging_setup.py`, `assistant.py` | review |
| Resource exhaustion | Question length limit, upload limits, per-session question and upload limits in the public demo | `config.py`, Streamlit app | configuration and interface tests |

## Not implemented (out of scope for the MVP)

| Control | Status | Needed before real data |
|---|---|---|
| Authentication | Not implemented | SSO with the corporate identity provider (OIDC/SAML) |
| Authorisation | Not implemented: every user can query every document | Roles and document-level permissions, applied as a metadata filter in the retrieval query, never by the model |
| Encryption at rest | Not implemented by the application | Encrypted volumes or a managed database with encryption |
| Retention and deletion | Documents stay until deleted; no automatic expiry (demo sessions excepted) | Retention policy per document class, deletion propagated to the index |
| Tenant isolation | Not implemented (demo sessions are isolated in memory) | Separate collections or databases per tenant |
| Audit trail | Structured logs only | Who asked what, when, with which sources, stored securely |
| Rate limiting of the API | Not implemented | Gateway quotas per user |
| Model provider agreement | Free tier in the demo | Paid or enterprise tier with a data processing agreement; on the free tier Google may use content to improve its products |

## Known residual risks

- Pattern-based injection detection only recognises known phrasings; the main defence is that the model output is verified and that passages never carry authority.
- A model can still cite a real passage that does not support its statement; the number check catches invented figures, not wrong reasoning.
- The prototype must not be used with real confidential documents in its current state.
