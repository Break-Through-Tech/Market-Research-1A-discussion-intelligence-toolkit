# November team integration audit

Read-only GitHub/source audit on November 14, 2026 (America/Phoenix). Ownership and due dates below come from Rohan's supplied November plan; the fetched issues have no formal GitHub assignees. No ownership or issue status was changed.

## Published activity checked

`git fetch origin` left `origin/main` at `e734489` and PR #23's remote head at `f46ec2a`. GitHub's repository PR search for `updated:>=2026-11-08` returned no results; the open-PR search returned only #22 and #23. The specified issues/PRs and PR discussions showed no new published activity after November 7. This does not establish whether teammates have unpublished local work.

| Work | Owner / due date from supplied plan | GitHub state | Code actually available |
|---|---|---|---|
| [#24 analyze pipeline](https://github.com/Break-Through-Tech/Market-Research-1A-discussion-intelligence-toolkit/issues/24) | Rohan / Nov 10 | Open; updated Nov 7 | Draft PR #23 already has validation, enrichment adapters, retrieval, grounded extraction and reports. Production fitted artifacts are absent. |
| [#25 evidence retrieval](https://github.com/Break-Through-Tech/Market-Research-1A-discussion-intelligence-toolkit/issues/25) | Franceli / Nov 14 | Open; updated Nov 7 | Integration branch already has deterministic TF-IDF/cosine and injected embeddings. New work here supplies a reusable curated evaluator, not another algorithm. |
| [#26 report format](https://github.com/Break-Through-Tech/Market-Research-1A-discussion-intelligence-toolkit/issues/26) | Gia / Nov 18 | Open; updated Nov 7 | Strict `InsightReport` v1, JSON/Markdown and citation validators already work. Final design remains Gia's decision; a proposal/schema/sample is prepared for review. |
| [#27 LLM synthesis](https://github.com/Break-Through-Tech/Market-Research-1A-discussion-intelligence-toolkit/issues/27) | Aisha / Nov 25 | Open; updated Nov 7 | Existing quotation-provider contract/fallback works. A deterministic mock demo is added; no LLM or external service is executed. |
| [#28 package structure](https://github.com/Break-Through-Tech/Market-Research-1A-discussion-intelligence-toolkit/issues/28) | Rohan / Nov 30 | Open; updated Nov 7 | Installable package, CLI, locked dependencies and wheel/source builds already work. Trusted-bundle changes are checkpointed locally as `7a496c2`; the new evaluator is packaged with the toolkit. |
| [#17 topic clustering](https://github.com/Break-Through-Tech/Market-Research-1A-discussion-intelligence-toolkit/issues/17) | Franceli / October carryover | Open; no issue comments | Assignment protocol/grouping exists, but no learned clustering implementation or production assignments were found in the fetched relevant branches. Franceli's PR #22 comment says she is using MiniLM. |
| [PR #22 quality/embeddings](https://github.com/Break-Through-Tech/Market-Research-1A-discussion-intelligence-toolkit/pull/22) | Gia | Open, not draft, unmerged; head `c21747a` | `utils.quality` scorers/embedding cache and quality experiments exist on that branch. Inspected read-only with `git show`; nothing copied or merged. |
| [PR #23 integration](https://github.com/Break-Through-Tech/Market-Research-1A-discussion-intelligence-toolkit/pull/23) | Rohan | Open, draft, unmerged; head `f46ec2a` | Published version still describes 119 tests; this session's extra work remains local until Rohan approves pushing. |

PR #22's latest discussion is Rohan's Nov 7 comment supporting the work while requiring conversation-level civility and embedding coordination. GitHub records that review as `COMMENTED`, not a formal `APPROVED` review. Franceli's Oct 31 comment supports shared embedding utilities and clarifying civility semantics. PR #23 has no discussion comments at audit time.

## Experiments and actual integration gaps

Aisha's baseline notebook/results and Franceli's text/context experiments already implement discourse classification. Franceli's selected context configuration uses separate fitted target/parent TF-IDF vectorizers, concatenated in that order, then logistic regression. Her experiment documentation reports held-out macro F1 0.4214 on 9,555 examples. Training happens at module import in the experiment scripts, so the inference package must not import them. Those prior measurements were read, not rerun or attributed to this integration.

Gia's PR #22 implements constant, length, TF-IDF and embedding civility scorers, plus a lazily loaded sentence-transformer embedder/cache. Its documented best test ROC-AUC is 0.664; that is Gia's experiment result, not a current integration measurement. Its `predict([comment_texts])` and `embed(texts)` interfaces fit the existing adapters. Its model property can download a checkpoint on first use, so the eventual offline demo needs an explicit locally provisioned checkpoint/cache and compatible owner code. No checkpoint was downloaded in this session.

The current pipeline correctly keeps P(civil) at conversation scope. PR #22's proposed reuse of the per-utterance `quality_score` slot still needs team reconciliation: this integration keeps argument quality and civility separate. The starter loader's outcome-exclusion correction already exists on Rohan's branch; Gia's official-split loader and benchmark remain her separate implementation.

There is overlap between #24 integration and #25/#26 because the branch already needs retrieval and report fallbacks to be runnable. The bounded contribution here is evaluation plus reviewable contracts/examples. No existing retrieval or clustering algorithm, teammate training code, model selection, final report design, or LLM implementation is replaced.

## Owner handoffs and December work

- Franceli: review the proposed relevance judgments; provide selected retrieval/cluster implementation and ID/noise conventions; export the classifier plus both fitted vectorizers with versions/preprocessing notes.
- Gia: review the report contract/schema/sample and agree on final organization; supply reviewed civility code/artifact, runtime dependencies and quality semantics; coordinate locally provisioned embeddings.
- Aisha: supply the synthesis provider and runtime choice; coordinate a separate contract/evaluation for genuinely abstractive claims. The deterministic mock is wiring only.
- Rohan: integrate owner-approved artifacts, verify packaged offline inference, assemble the end-to-end walkthrough and track failures/coverage.

Before December: obtain independent relevance judgments on unseen threads, compare retrieval backends on frozen data, evaluate report coverage/citation relevance/readability with humans, test production artifact loading and failure modes, resolve civility/argument-quality semantics, validate remote CI across supported Python versions, and complete the final demo/notebook/results/limitations and license review. These remain open work, not completed by synthetic tests or the small curated reference evaluation.
