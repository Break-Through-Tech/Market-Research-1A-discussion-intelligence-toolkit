# Report format

This document settles issue #26: how the result of `analyze()` is organized and displayed. It builds on the `InsightReport` version `1.0` contract described in [report-contract-proposal.md](report-contract-proposal.md). No JSON field was added, renamed or removed, so `examples/insight-report.schema.json` and the exact-quotation rule are unchanged. What changed is the Markdown layout produced by `InsightReport.to_markdown()`.

## Two outputs, one report

| Output | Made by | Audience | Content |
|---|---|---|---|
| JSON | `report.to_json()` or `--format json` | Code, evaluation, storage | Everything: every analyzed comment, every assignment and score |
| Markdown | `report.to_markdown()` or `--format markdown` | A person reading the result | The same report, ordered for reading and limited to what the reader needs |

JSON is the complete record. Markdown is a view of it and never contains anything that is not in the JSON.

```bash
uv run --offline discussion-intelligence \
  --input data/coarse-discourse-reference-sample.jsonl \
  --query "RAID data drives" --top-k 3 --format markdown
```

## Markdown layout

Sections always appear in this order. A section marked "when present" is left out when it has nothing to show, and the At a glance table says why.

| # | Section | JSON fields | Shown |
|---|---|---|---|
| 1 | Title, source link, conversation ID, query | `title`, `source_url`, `conversation_id`, `query` | Always. The source link and query lines appear only when the value exists. |
| 2 | At a glance | `evidence`, `topic_assignments`, `conversation_quality`, `components` | Always, with the same seven rows in every report |
| 3 | Key points | `summary` | Always |
| 4 | Themes | `sections`, `topic_assignments` | When a clusterer assigned topics |
| 4 | More excerpts | `sections` | When there are no topics and more excerpts were selected than Key points shows |
| 5 | Agreement, disagreement and questions | `agreements`, `disagreements`, `questions` | When at least one comment has a discourse label |
| 6 | Open questions | `open_questions` | When present. The pipeline does not fill this yet. |
| 7 | Component availability | `components` | Always |
| 8 | Limitations | `limitations` | Always |
| 9 | Source evidence | `evidence` | Always |

The conclusions come first and the supporting detail comes last, so a reader can stop after Key points and still have the main result, or follow any citation down to the original comment.

## Decisions

### Missing and partial results

A report is often produced without every model. The rule is that a missing result is stated once, in a fixed place, and never silently dropped.

- At a glance always has the rows Comments analyzed, Comments quoted, Participants, Dates, Themes, Discourse labels and Civility estimate. A row whose result is missing reads `not available`, or `failed` when the component raised an error. The reason is in Component availability.
- Partial results show their coverage, for example `Discourse labels: 5 of 7 comments` or `Themes: 2 (6 of 7 comments assigned)`.
- Participants and Dates read `not recorded in the source` when the dataset has no author or timestamp. The shared placeholder speaker is not counted as a person, and its ID is not printed.
- Body sections with nothing to show are left out. An empty heading adds length without information, and the At a glance row already explains the absence.

### Key points

Key points shows the `summary` field, which holds the three highest-ranked excerpts. Each one is a blockquote followed by a `Source:` line that links to the comment under Source evidence.

Every point is an exact quotation of a comment. When the quotation starts or stops inside the comment, the Markdown adds `…` on that side so a cut sentence does not read as the whole comment. The `…` is display only. The JSON `text` stays an exact substring of the evidence.

### Themes

When a clusterer supplies topics, each topic becomes a `### Theme <cluster id> (n of N comments)` subsection that lists its excerpts. The count shows how much of the thread the theme covers. Themes appear in the order of their highest-ranked excerpt. Excerpts whose comment has no topic go under `Not assigned to a theme`.

The cluster ID is displayed as it is. Topic names are not generated yet, so a clusterer that wants readable names can return them as its cluster IDs. This is the open point to agree on with the clustering work in #17.

Key points repeats the top excerpts that also appear under their theme. That is intended, because the grouping is the new information. Without topics there is no grouping, so the report shows only the remaining excerpts under More excerpts and does not print the same quotation twice.

### Agreement, disagreement and questions

The three subsections always appear together when discourse labels exist. An empty one reads `None among the selected excerpts.` The section opens with one sentence explaining that a label describes the role of a single comment and does not show consensus or an unanswered question. This is the same limit the contract sets.

### Civility estimate

The estimate is one row in At a glance: the value with two decimals, the model name and the number of comments it was computed from. It is a single number for the whole conversation, so it does not get its own section, and it does not list every comment ID. Those remain in `conversation_quality.evidence_ids` in the JSON. Two decimals are enough for a score whose best test ROC-AUC so far is about 0.66 (#22).

### Citations and source evidence

- A citation is the canonical utterance ID, such as `reddit:t1_c6fwalq`, linked to its entry under Source evidence. The same ID finds the comment in the JSON and in the dataset.
- Source evidence lists each quoted comment and the comment it replies to, so every quotation can be read in context. The remaining comments stay in the JSON. A note gives the numbers, for example `Showing 5 of 7 analyzed comments`. For a long thread this keeps the report a digest and not a copy of the thread.
- Entries are in thread order, with each comment before its replies. Each entry has one line of details (parent, speaker, date, discourse label, theme, links to the source) and then the full text.
- Links to the original site appear only when the source provided an `http` or `https` URL.

### Component availability and limitations

Component availability is a table with one row per component: its state and the detail string from the pipeline. Limitations stay a list. Both sit after the findings because they qualify the result, and both are always shown so that a report produced by fallbacks cannot be mistaken for one produced by trained models.

## Examples

| File | What it shows |
|---|---|
| [examples/buildapc-report.md](../examples/buildapc-report.md) | A query report from the default pipeline, with no trained model attached |
| [examples/karma-report.md](../examples/karma-report.md) | The second reference thread, same settings as the contract proposal |
| [examples/full-report-example.md](../examples/full-report-example.md) | Every section filled in |

The full example comes from `examples/report_format_demo.py`. Its labels, themes and civility value are produced by simple placeholder rules, not by the team's models, so it illustrates the layout only.

## Not decided here

- Readable theme names, which depend on the clustering output in #17.
- Generated (non-quoted) summaries. They need the separately versioned contract described in the proposal, and the layout would then have to tell quotations and generated text apart.
- How readable the report is for people outside the team. The layout has been checked on the two reference threads only, and a review with real readers is still needed.
