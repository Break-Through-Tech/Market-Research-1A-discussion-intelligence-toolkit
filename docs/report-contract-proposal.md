# Report integration contract for Gia and Aisha to review

November 14, 2026. This proposal documents the **existing** `InsightReport` version `1.0`. Gia owns the final #26 organization and presentation; no field was renamed, report renderer redesigned, or semantic policy weakened. `examples/insight-report.schema.json` is a machine-readable snapshot of `InsightReport.model_json_schema()`. JSON Schema checks shape/types, but the Python validators must still check identity, quotations and citations.

## Current contract and review points

| Need | Existing JSON fields / Markdown behavior | Meaning and review boundary |
|---|---|---|
| Identity and provenance | `conversation_id`, `conversation_source_id`, `title`, `source_url`, `query` | Preserve canonical/source identities and provided URLs. No generated source links. |
| Main points and themes | `summary`, `sections[{heading, points}]`, `topic_assignments` | Summary contains up to 3 selected quotations. Default sections contain up to top_k quotations; an injected provider chooses its own number of valid quotations. Without supplied assignments the heading is "Discussion excerpts"; with them it is "Topic cluster <id>". No learned themes or semantic theme names are claimed. Gia and Franceli can agree on display names later. |
| Evidence-backed points | `Claim{text, evidence_ids}` | Nonempty exact quotation of each cited evidence's whitespace-normalized text. Default extraction uses at most the first 400 characters; a supplied quotation provider can select a longer exact excerpt. These are participant statements, not verified facts. |
| Traceable evidence | `evidence[canonical_id]` | Full raw/cleaned text, original source ID, conversation/parent/speaker IDs, available time and URLs, provenance metadata. Markdown anchors use actual canonical IDs and citations encode the matching URL fragments. |
| Optional discourse | `Evidence.discourse_act`; `agreements`, `disagreements`, `questions` | Null when unavailable. Gold annotations remain metadata. Labeled excerpts do not establish interpersonal agreement, disagreement targets, or unresolved questions. |
| Optional civility | `conversation_quality{scope,target,value,evidence_ids,model_name}` | Scope `conversation`, target `civility_probability`, finite value in [0,1], actual context IDs. Null without a successful scorer. Never copied into per-comment `argument_quality_score`. |
| Availability / failure | `components[name]{state,detail}` | `available`, `unavailable`, or `failed`. Prediction/assignment counts show coverage; synthesis/retrieval details name the actual fallback. A successful report does not mean production models ran. |
| Limits and missing data | `limitations`, nullable evidence dates/URLs, metadata flags | Missing timestamps/authors are disclosed. The shared unknown-speaker placeholder must not be counted as distinct people. Report evidence retains provenance, including reference annotations. |
| Unresolved questions | `open_questions` | Empty in the current implementation. No inference of consensus, resolution or unanswered status is supported. |

The sample [karma report](../examples/karma-report.md) is actual current CLI output from the second reference thread. It retains "THIS IS BASED ON SPECULATION", shows unavailable models, and includes all 4 usable evidence utterances with thread links. It does not claim these explanations are correct or consensual. The original [build report](../examples/buildapc-report.md) remains byte-identical.

```bash
uv run --offline discussion-intelligence \
  --input data/coarse-discourse-reference-sample.jsonl \
  --conversation-index 1 --query "comment karma easier" --top-k 2 --format markdown
```

For Gia's review: agree on theme labels and hierarchy when clusters exist; determine which optional fields to display; decide how to show partial predictions and missing information; evaluate redundancy, truncation and readability with humans. Keep IDs/evidence/status semantics compatible, or explicitly version any breaking contract. This document does not approve a final design on her behalf.

## Aisha's synthesis integration demo

The existing interface is `SynthesisProvider.synthesize(evidence, query) -> SynthesisDraft` (or its matching dictionary). Only copies of **retrieved** evidence are passed, in retrieval order. Return `points: [{text, evidence_ids}]`. A successful injected provider replaces the selected extractive points; unknown/unretrieved/repeated citations, unsupported text, malformed outputs or exceptions cause a visible `failed` synthesis status and retain the deterministic extractive points.

The isolated `examples/synthesis_provider_demo.py` implements a deterministic mock quotation provider. It does not run an LLM, call an API, load credentials, or download a model. The report adds an explicit mock/no-LLM limitation. Run:

```bash
uv run --offline python examples/synthesis_provider_demo.py --mode quote
uv run --offline python examples/synthesis_provider_demo.py --mode fabricated
uv run --offline python examples/synthesis_provider_demo.py --mode error
uv run --offline pytest -q tests/integration/test_review_examples.py
```

`quote` returns an actual retrieved quotation with its unchanged canonical ID. `fabricated` deliberately returns an unsupported consensus claim citing a real ID; validation rejects it. `error` raises deliberately; extraction survives. The tests check that the provider only sees the one selected RAID reply, both rejected modes retain the original extractive summary, and reports never expose the fabricated assertion. Existing pipeline tests also reject unknown, unretrieved and duplicate IDs, mutation of evidence, and empty drafts.

## What genuinely abstractive synthesis would need

Aisha can first plug an owner-supplied provider into this quotation-only interface without changing it. An LLM used this way would select quotations; it would not make free-form abstraction valid.

For paraphrases or newly composed claims, agree on a separately versioned contract with Gia: distinguish quotes from generated claims, retain exact supporting spans/IDs and context, define how each citation supports the entire claim, add independently tested semantic support/contradiction checks and human review, and expose uncertainty and fallback reasons. Existence of an ID or passing JSON Schema is insufficient. Consensus, disagreement targets and question resolution require their own evidence and evaluation. Decide local model/runtime/resource policy with Aisha before adding a client. No such policy, semantic validator or actual LLM is implemented here, and the exact-quotation policy remains unchanged.
