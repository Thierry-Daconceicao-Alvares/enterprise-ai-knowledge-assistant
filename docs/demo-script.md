# Two-minute demo script

Setting: the public demo (or a local run with the sample documents loaded). Speak while clicking; each block is about 20 to 30 seconds.

**0:00 - The problem (20 s)**

"Nexora Industries is a fictional manufacturer with 1,200 employees. Its rules live in dozens of documents: purchasing, approvals, IT, HR, quality. People waste time finding the right one, and sometimes use an outdated version. This assistant answers questions from those documents, shows exactly where each answer comes from, and says when the documents don't contain the answer. All the data is synthetic."

*Point at the sidebar: 32 documents, 239 passages, 2 superseded versions.*

**0:20 - A real question (35 s)**

*Click the example "What is the procedure to approve a purchase request of more than €50,000?"*

"This question needs two documents: the procurement policy for the quotes and the delegation of authority for the approvers. The answer is a list of short statements, and each one carries the number of the passage it comes from. When a statement combines passages, for example comparing an amount with a threshold, it is marked as inferred."

**0:55 - Citations you can check (25 s)**

*Open the first source.*

"The sources are not written by the model: it only cites passage numbers, and the application attaches the file, the page, the section and the relevance score from the index. Here is the exact passage, and for a PDF I can display the original page."

*Click "Show page".*

**1:20 - When the answer is not in the documents (25 s)**

*Click "What is the daily meal allowance for a business trip to Germany?"*

"The travel policy covers France and says that country rates are published elsewhere, so the assistant does not guess an amount: it says the documents don't contain it, and why. On the evaluation set, questions like this are scored separately, as abstentions."

**1:45 - How I know it works (15 s)**

"Every behaviour you saw is measured on 62 labelled cases: correct answers, citations, abstentions, contradictions, outdated versions, and a supplier PDF with hidden instructions. The report, including the failures, is in the repository, along with the limits: no authentication yet, and no OCR."

**Fallback if the API is rate-limited**: show the screenshots in the README and the evaluation report, and explain the message the user sees: after a per-minute limit, the application waits and retries; when the daily quota is used up, it says at what time the demo will answer again.
