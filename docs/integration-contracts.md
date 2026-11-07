# Inference contracts

This integration reuses `artifacts.schema` without modifying it. `analyze(conversation, ...)` and `EvidenceIndex(conversation, ...)` validate a copy. Models are dependencies supplied by the caller, not downloaded or trained by the pipeline.

## Data passed to models

Each `InferenceUtterance` is an immutable record with `id`, cleaned `text`, `parent_id`, usable `parent_text`, `speaker_id`, and nullable `created_at`. Deleted/empty source content is excluded. Annotation metadata and gold discourse labels are absent. Inputs use parent-before-child source traversal; epoch sentinels become `None`. IDs always identify the original canonical utterance.

| Argument | Contract | Validation/failure behavior |
|---|---|---|
| `classifier` | `predict(utterances) -> Mapping[canonical_id, discourse_label]` | Only input IDs and the documented Coarse Discourse labels; partial mapping allowed |
| `quality_scorer` | `score(utterances) -> ConversationQuality` | Finite `[0,1]` civility probability, scope `conversation`, nonempty actual context IDs |
| `clusterer` | `assign(utterances) -> Mapping[canonical_id, str_cluster_id]` | Partial mapping allowed; unknown IDs/blank/non-string labels rejected |
| `embedder` | `embed(texts) -> numpy.ndarray` of shape `(N,D)` | Finite values, stable nonzero dimension; vectors need not be normalized |
| `synthesis_provider` | `synthesize(retrieved_evidence, query) -> SynthesisDraft` | Nonempty exact quotations from every attached citation; only retrieved IDs accepted |

Invalid input/arguments raise. Optional components catch ordinary exceptions, expose `failed` status and exception type, and retain the working pipeline. Exception messages are not copied into reports. Partial predictions are counted explicitly; unassigned utterances remain unlabeled/ungrouped. An empty conversation skips optional model calls. An embedding failure during indexing or query encoding uses TF-IDF. A synthesis failure uses the original deterministic extractive points.

## Franceli's context classifier

The existing experiments fit two `TfidfVectorizer` objects and concatenate target features followed by parent features before a `LogisticRegression` classifier. They are not safe to import for inference: training executes at module import and requires prepared corpora. No trained artifact is currently tracked.

After Franceli exports or supplies the **three fitted objects** from her selected configuration:

```python
from discussion_intelligence import analyze
from discussion_intelligence.adapters import ContextClassifierAdapter

# model, target_vectorizer, parent_vectorizer are fitted objects from the owner.
classifier = ContextClassifierAdapter(model, target_vectorizer, parent_vectorizer)
report = analyze(conversation, classifier=classifier)
```

The adapter only calls `transform` and `predict`, with target features first. Unavailable/deleted parents contribute empty text. Text-only sklearn pipelines or other existing `TextClassifier` implementations can use `TextClassifierAdapter(model)` instead. These bridges were verified with small synthetic fitted sklearn models, not Franceli's unavailable production artifact. Export version, class order, feature order, cleaning policy, and package versions with the artifact.

## Gia's civility scorer and embeddings

PR #22 supplies scorers with `predict([list_of_comment_texts]) -> [P(civil)]`. Once its code is reviewed/merged and a scorer has been fitted elsewhere:

```python
from discussion_intelligence.adapters import ConversationCivilityAdapter

quality = ConversationCivilityAdapter(fitted_scorer, model_name="owner/version")
report = analyze(conversation, quality_scorer=quality)
```

The adapter orders comments chronologically when **all** real timestamps are available; otherwise it keeps parent-before-child traversal and the report discloses missing chronology. It cites the IDs actually passed to the scorer. It never writes civility into `Utterance.quality_score`: that existing schema field describes per-comment argument quality, a different target.

All input is **observed context**. The adapter does not infer and drop a future outcome from arbitrary new threads. For CGA-CMV evaluation, use Gia's outcome-excluding, official-split dataset loader. Feeding a complete labeled CGA-CMV conversation with its terminal removed outcome into `analyze` is not a valid prediction evaluation.

Gia's embedder implements the existing `artifacts.evaluation.TextEmbedder` structural contract. After PR #22 merges:

```python
from utils.quality.embeddings import SentenceTransformerEmbedder

# Construct/load once outside analyze; initial use may download an open model.
embedder = SentenceTransformerEmbedder()
report = analyze(conversation, embedder=embedder)
```

This import is intentionally absent from the current pipeline: `utils.quality` is not on the audited main. The integration defines a lightweight structural protocol instead of importing the full evaluation metrics module. `uv sync --extra embeddings` installs optional ML dependencies; an already loaded sentence-transformer can also be wrapped by an object whose `embed(texts)` calls its `encode(texts, convert_to_numpy=True)`. No second embedding implementation/cache is copied here.

## Franceli's future clusters

Supply an object implementing `assign(utterances)` with canonical IDs. The pipeline groups selected excerpts under `Topic cluster <id>`; it does not invent a semantic theme title. Unassigned/mixed-cluster evidence goes under `Discussion excerpts`. Persisted assignments can be exposed by a small mapping adapter; online clustering belongs to the cluster owner. No clustering algorithm is duplicated by this branch.

## Trusted artifact example

`examples/fitted_model_integration.py` loads a trusted joblib mapping with any of:

- `discourse_model`: fitted text classifier;
- `context_model`, `target_vectorizer`, `parent_vectorizer`: Franceli-compatible fitted context objects;
- `civility_model`, optional `civility_model_name`: Gia-compatible fitted scorer;
- `topic_clusterer`: object implementing `assign`;
- `embedder`: object implementing `embed`.

```bash
uv run --offline python examples/fitted_model_integration.py \
  --bundle /path/to/trusted-fitted-models.joblib \
  --input data/coarse-discourse-reference-sample.jsonl
```

Joblib artifacts can execute code on load. Load only model-owner artifacts you trust; no checkpoint is included in this repository. A production artifact still requires the original owner's review and domain evaluation. The normal CLI does not load artifacts and has no paid-service dependency.

## Synthesis and grounding

An injected provider receives copies of **retrieved** evidence, not the full conversation. It may return a `SynthesisDraft` model or matching dictionary containing `points: [{text, evidence_ids}]`. Every point must be a nonempty exact substring of each cited evidence's whitespace-normalized source text. Duplicate, unknown, or unretrieved IDs and invented paraphrases trigger extractive fallback. An LLM could select quotations through this interface, but no LLM client or prompt is included.

This restrictive contract is intentional: checking ID existence alone cannot establish entailment. Free-form abstractive synthesis needs a separate validation/evaluation design before it can be accepted. Model labels do not establish agreement between participants or whether a question is unresolved; `open_questions` remains empty. `agreements`, `disagreements`, and `questions` contain selected quotations whose supplied/predicted discourse labels match those roles.

The report's `summary` is up to three cited excerpts. `sections` contains up to `top_k` selected excerpts, each capped at 400 characters; full usable source text remains in `evidence`. Gold annotations remain provenance metadata, never predictions. Model-produced fields retain explicit availability status. Serialization checks ID consistency and exact quotations again, including after nested mutable data is changed.
