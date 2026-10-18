# Discovery brief: Nexora Industries knowledge assistant

*Simulated customer engagement. Nexora Industries is a fictional company; the interviews, people and figures below are invented to frame the project and are not real research.*

## Customer problem

Nexora Industries (about 1,200 employees, headquarters in France, three plants and a distribution center) keeps its operating rules in a few hundred documents: purchasing policies, approval workflows, supplier procedures, IT guides, HR policies, quality procedures and operating manuals. The documents live in different shared folders, in PDF, Word and text formats, and several versions of the same procedure coexist.

Employees who need a rule (for example "who approves a purchase of €60,000?") search the folders, ask a colleague or open a ticket. Answers are slow and inconsistent, and sometimes based on an outdated version.

## User personas

| Persona | Typical questions | What matters to them |
|---|---|---|
| **Requester** (engineer, plant supervisor) | Who approves my purchase? Can I pay this with my P-card? How do I onboard a new supplier? | A fast, clear answer with the next step |
| **Buyer** (purchasing team) | Quote thresholds, single-source rules, price discrepancy tolerance | Exact thresholds and the section to quote to a requester |
| **New employee** | Leave, remote work, IT access, safety induction | Plain answers without knowing which document to open |
| **Process owner** (CPO office, HR, IT, Quality) | Are people applying the current version of my procedure? | Answers that use the current version and cite it |
| **Compliance / internal audit** | Can we trust what the assistant says? | Traceable sources, no invented rules, known limits |

## Current workflow

1. Search the shared folders by file name, or ask a colleague.
2. Open several documents to find the relevant section.
3. If unsure, open a ticket to the process owner (purchasing, HR, IT), who answers by email.
4. The answer is not recorded; the same question is asked again by someone else.

## Pain points

- Time lost searching, and questions sent to support teams that could be self-served.
- Outdated documents still in circulation (for example an old delegation of authority).
- Inconsistent answers from colleagues, and documents that contradict each other.
- No visibility on which rules are hard to find or unclear.

## Business requirements

| ID | Requirement | Priority |
|---|---|---|
| BR1 | Answer questions in natural language (English and French) from the company documents only | Must |
| BR2 | Show the source of every answer: document, page or section, and the passage | Must |
| BR3 | Say clearly when the documents do not contain the answer | Must |
| BR4 | Use the current version of a procedure; flag contradictions instead of choosing silently | Must |
| BR5 | Let authorised staff add PDF, Word and text documents | Must |
| BR6 | Measure answer quality before rollout, and after every change | Must |
| BR7 | Restrict documents by role (for example HR files) | Later |
| BR8 | Integrate with the corporate identity provider (SSO) | Later |
| BR9 | Collect user feedback on answers | Later |

## Technical constraints

- No dedicated platform team for the pilot: the solution must run as a single container, without a separate database server.
- Documents are heterogeneous: PDF exports with tables, Word files, text files; no guarantee of consistent templates.
- Budget for the pilot is limited: the model API cost must be estimated per question.
- The pilot uses synthetic or public documents only, until security and legal approve real data.

## Security concerns

- Confidential documents must not leak to unauthorised employees: access control is required before real HR or finance documents are indexed.
- Documents sent to an external model API: a data processing agreement and a paid or enterprise tier are required for real data (free tiers may use data for product improvement).
- Documents received from outside (supplier brochures) may contain instructions aimed at AI systems (prompt injection).
- No confidential content in application logs.

## Success criteria (pilot)

These are project targets, to be confirmed with the business after the first evaluation, not results:

| Criterion | Target |
|---|---|
| Correct answers on answerable questions of the evaluation set | at least 80% |
| Answers citing a correct source | at least 90% |
| Questions not covered by the documents answered anyway | at most 10% |
| Answers following an injected instruction or citing a superseded version | 0 |
| Median response time | under 5 seconds |
| Cost per question on the paid tier | under $0.01 |

The [evaluation report](../README.md#13-evaluation-methodology-and-results) compares the actual results with these targets.
