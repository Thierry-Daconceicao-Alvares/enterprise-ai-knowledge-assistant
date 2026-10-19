# Evaluation methodology

## Evaluation set

[`eval/dataset.jsonl`](../eval/dataset.jsonl) contains 62 cases written by hand against the 32 sample documents. Each case has a question (or a file to upload), a category, the expected behaviour, the relevant sections, the key facts a correct answer must contain, facts it must not contain, and a reference answer for human review.

| Category | Cases | What it tests | Expected behaviour |
|---|---:|---|---|
| `single_document` | 11 | A fact stated in one section (one case in French) | answer |
| `multi_passage` | 6 | Facts from several sections of one document | answer |
| `multi_document` | 7 | Facts from several documents, including the reference question in English and French | answer |
| `no_answer` | 7 | Questions the documents do not answer (one in French) | abstain |
| `ambiguous` | 4 | Vague questions ("What is the approval limit?") | clarify, or cover the main interpretations |
| `contradictory` | 4 | Two current documents that disagree | answer with both values and flag the conflict |
| `citation` | 5 | Questions about where a rule is written | answer citing the right section |
| `misleading_instructions` | 6 | Injection in the question, or a supplier PDF with hidden instructions | answer correctly without following the instruction, or refuse |
| `superseded` | 4 | Rules that changed between versions | answer with the current version only |
| `malformed_input` | 8 | Empty, wordless or too long questions; empty, unsupported, fake, scanned or truncated files | reject with the right error code |

Labels point to sections (`NX-PUR-002 §2`), not to chunk IDs, so they remain valid when the chunk size changes. Automated tests check that every label matches a section of a current document, that every English key fact appears in the labelled sections, and that no forbidden fact appears in them.

## Metrics

**Retrieval** (42 questions with labelled sections):

- **Hit@k**: share of questions with at least one relevant passage among the first k results (k = 1, 3, 5).
- **Recall@5**: share of the required sections found in the first 5 results, averaged over questions. It matters for multi-document questions.
- **MRR** (mean reciprocal rank): average of 1 / rank of the first relevant passage (0 when none is found).

**Answers** (all 62 cases):

- **Cases passed**: the case shows the expected behaviour and passes all its checks.
- **Fully correct answers**: answerable questions answered with every key fact, the right yes/no verdict, no forbidden content and no superseded citation.
- **Key-fact recall**: share of key facts present in the answer; an abstention scores 0.
- **Citation hit rate**: answers citing at least one required section.
- **Citation precision**: share of citations that point to a required or related section.
- **Correct abstentions / unsupported answers**: on questions the documents do not answer, the share declined and the share answered anyway.
- **Forbidden content**: answers that contain an injected marker or a superseded value.
- **Superseded citations**: citations of superseded documents (expected: 0).
- **Error rate**: cases that failed because of an API error.
- **Latency** (p50 and p95, end to end), **tokens** and **estimated cost** per question at paid-tier list prices.

## How facts are matched

A key fact is a short phrase, with alternatives separated by `|`. Text is normalised before matching: accents removed, numbers unified (`€50,000`, `50 000 €` and `50.000` all become `50000`), English and French number words converted (`three`, `trois` → `3`), plurals accepted, and a number never matches a longer number (`5000` does not match `50000`). A yes/no question also checks the first word of the answer.

## Limitations

- The corpus and the questions were written by the same person, which makes the questions closer to the wording of the documents than real employee questions would be. This favours keyword search (BM25).
- Matching phrases is stricter than a human reader: a correct paraphrase can fail a check. Every failed case is listed in the report with the answer, for manual review.
- 62 cases give a useful signal, not a precise estimate: one case is 1.6 percentage points.
- The model is not deterministic: two runs can differ by a few cases.
- The thresholds in the [discovery brief](discovery-brief.md) are project targets, not results.

## Running it

```bash
knowledge-assistant evaluate --retrieval-only               # retrieval metrics only
knowledge-assistant evaluate --delay 4                      # full run on the free tier
knowledge-assistant evaluate --answer-retrievers semantic,bm25 --delay 4
```

Each run indexes the sample documents in a fresh in-memory index, so the results do not depend on the local index. The report is written to `eval/results/latest/report.md` and `results.json`, with the models and settings of the run.

The **Evaluate with Gemini** workflow (GitHub Actions, started manually) runs the same command with the repository secret and records the evaluated commit in the report. It can also capture the README screenshots and commit the results to `eval/results/<date>-<label>/`. When a run fails, the reason appears in the run summary.

**API usage.** With `--answer-retrievers semantic,bm25`, a run sends about 110 generation requests and embeds about 300 texts: the 239 passages in 5 batch requests, then each question once (question vectors are cached, so the retrieval and answer evaluations share them). The screenshot step indexes the documents again. On the free tier, plan one full run per day: daily quotas reset at midnight Pacific time. When the daily quota is used up, the run stops at once and says so instead of recording 62 errors.
