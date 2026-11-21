# Curated reference retrieval evaluation

Prepared on November 14, 2026 for Franceli's #25 work. This adds an evaluator, not a competing retrieval algorithm. `EvidenceIndex`, TF-IDF configuration, cosine ranking and citation behavior are unchanged. The evaluator reuses `artifacts.evaluation.retrieval_recall_at_k` and accepts an injected index factory for future comparisons.

## Data and proposed relevance judgments

The tracked `data/coarse-discourse-reference-sample.jsonl` has two already inspected Reddit threads, containing 7 build-help and 4 karma utterances. All 11 have usable text; none is deleted. They are the same examples used during development, **not a held-out or statistically representative benchmark**.

`examples/retrieval-judgments.json` contains 10 source-inspected cases: 8 with relevant replies and 2 with no answer. Judgments are agent-authored proposals pending Franceli/team review, not independent human annotations or corpus-supplied relevance labels. Each positive ID has a literal supporting quotation and an intent-specific rationale. The question/root is excluded when it merely asks for the requested explanation. Relevance denotes a participant's answer, not agreement with its truth. No discourse gold annotation is used as a prediction or relevance label.

The queries/labels were written before measuring this set, with knowledge of the existing TF-IDF implementation and prior demos. They were not tuned after the reported run. The set includes keyword matches, two multi-reply questions, a semantic paraphrase, an irrelevant query without shared vocabulary, and an irrelevant query with shared vocabulary. Query rows within a thread are correlated; confidence intervals or generalization claims would be misleading here.

The runner checks the exact input SHA-256 (`53be6730d7043963bb78647c69a84d377f371081a0e0b82df9e188f57fc943a0`). Changed data requires reviewed judgments and an updated hash. The evaluator verifies every relevant ID and supporting quote against usable source text, rejects repeated judgments, and verifies every returned ID, original source ID, parent, speaker, raw/cleaned text and supplied links. The source only supplies thread URLs; per-comment `source_url` stays null. No permalink is invented.

## Metrics and actual measurements

For each positive query, Recall@k is the number of distinct judged relevant IDs among the first k returned results divided by the number of relevant IDs. The reported macro mean weights each of the 8 positive queries equally. No-answer cases have undefined recall (`null`), and are excluded from that mean. Their separate abstention rate is the fraction with **zero returned evidence**, not a calibrated answerability score. Missing denominators are `null`/`n/a`, never a made-up perfect score.

The unmodified TF-IDF retriever measured:

| k | Macro Recall@k, 8 positive queries | No-answer abstention, 2 queries |
|---|---:|---:|
| 1 | 0.7291666667 | 0.5 (1/2) |
| 2 | 0.8333333333 | 0.5 (1/2) |
| 3 | 0.8750000000 | 0.5 (1/2) |

| Case | Relevant replies | Recall@1 | Recall@2 | Recall@3 |
|---|---:|---:|---:|---:|
| build-raid | 1 | 1 | 1 | 1 |
| build-memory-multiple | 2 | 1/2 | 1 | 1 |
| build-clock | 1 | 1 | 1 | 1 |
| build-semantic-paraphrase | 1 | 0 | 0 | 0 |
| karma-effort | 1 | 1 | 1 | 1 |
| karma-links | 1 | 1 | 1 | 1 |
| karma-abundance | 1 | 1 | 1 | 1 |
| karma-explanations-multiple | 3 | 1/3 | 2/3 | 1 |

`graphics upgrade worth extra cost` returns no evidence despite a relevant pricing/clock-speed reply (`reddit:t1_c6fwmq5`). This demonstrates lexical mismatch. `karma password reset` returns two irrelevant replies because they share "karma"; the corpus contains no password-reset answer. `quantum entanglement` correctly returns nothing. These misses are retained. There is no threshold tuning, algorithm change, or claim that Recall@3 of 0.875 transfers to new threads.

## Reproduce and reuse

From the repository root:

```bash
uv sync --locked
uv run --offline python examples/evaluate_retrieval.py
uv run --offline python examples/evaluate_retrieval.py \
  --format json --output /tmp/retrieval-evaluation.json
cmp examples/retrieval-evaluation.json /tmp/retrieval-evaluation.json
uv run --offline pytest -q tests/integration/test_retrieval_evaluation.py
```

The committed JSON includes per-query rankings, scores, original source IDs and links. It has no run timestamp, allowing deterministic regeneration with the locked environment.

Franceli can keep the same reviewed judgments and swap the index factory in Python:

```python
from discussion_intelligence import EvidenceIndex
from discussion_intelligence.retrieval_evaluation import evaluate_retrieval

# conversations and judgments are loaded as in examples/evaluate_retrieval.py.
# owner_embedder is already loaded by its owner; no model is selected here.
result = evaluate_retrieval(
    conversations, judgments,
    index_factory=lambda conv: EvidenceIndex(conv, embedder=owner_embedder),
    retriever_name="owner-supplied-embedder/version",
)
```

Other retrievers can return the existing `SearchHit` objects from `search(query, top_k=...)`. Returned order is respected. Provider errors raise; evaluation never silently switches to a fallback retriever. This API does not independently hash in-memory conversations: the example runner checks the source file hash, and callers evaluating other data must provide reviewed judgments/provenance for that data.

Deleted-content exclusion is tested using a **controlled deletion flag on a copy** of a real reference reply. That mutation is a robustness check, excluded from all curated metrics. The evaluator rejects stale judgments that cite the now-unusable reply. Existing synthetic tests cover removed/empty content and invalid trees separately.

Next steps belong to the retrieval evaluation/review process: have Franceli review judgments, add independent human relevance annotations and unseen threads, include synonyms/negation/partial answers, compare her selected retriever on frozen data, and assess citation usefulness and coverage. These curated measurements establish a reproducible diagnostic, not a production quality gate.
