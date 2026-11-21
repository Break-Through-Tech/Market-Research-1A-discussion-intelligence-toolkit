# Quality Scoring and Embeddings (Prototype)

Issue #16 ("9. Prototype quality scoring and embeddings"): choose a supported quality target from labeled discussion data, test a simple scorer and sentence-transformer embeddings with appropriate metrics, and document the results.

**Result in one line.** On ConvoKit's CGA-CMV corpus, logistic regression on `all-mpnet-base-v2` embeddings predicts whether a ChangeMyView discussion will have a comment removed by moderators with a test ROC-AUC of **0.664** (95% CI 0.637 to 0.689). TF-IDF reaches 0.611 and a length-only baseline 0.559. This is a useful but weak signal; see [Limitations](#limitations).

## Quality target

| Candidate | Labels | Why not / why |
|---|---|---|
| Coarse Discourse Reddit score (`ups`) | per comment | Not a quality label. Score depends on post visibility, timing and comment position more than content, and the corpus has no timestamps to control for that. |
| Coarse Discourse discourse acts | per comment | Already the classifier's target (#15). Acts such as `negative_reaction` are a behavior, not a quality judgment. |
| **CGA-CMV moderator removal** | **per conversation, human decision** | **Chosen.** A moderator decided a comment broke CMV's civility rules. It comes with matched pairs that control for topic, balanced labels and an official split. It is the corpus the project brief recommends for quality modeling. |

**Definition.** For each conversation, the quality score is `P(civil | comments so far)` in [0, 1], the same range as `Utterance.quality_score` in the schema. Civil means no comment was later removed by moderators. Higher means the discussion is more likely to stay civil.

This is a *conversation-level civility forecast*, not an argument-strength score. It answers "is this thread heading somewhere bad?", which is useful for ranking and flagging threads in a report. It does not say whether an individual argument is convincing.

## Data

| | Train | Dev (official `val`) | Test | Total |
|---|---:|---:|---:|---:|
| Conversations | 4,089 | 1,367 | 1,364 | 6,820 |
| Civil share | 49.8% | 50.0% | 49.9% | |
| Context comments per conversation (mean) | 5.21 | 5.15 | 5.03 | |

Source: `conversations-gone-awry-cmv-corpus` (6,842 conversations, 42,964 utterances, all linear reply chains). The module `utils/quality/cga_cmv.py` reads the downloaded files directly, the same way `utils/modeling_data/` does for Coarse Discourse.

### What the model is allowed to see

| Rule | Count | Reason |
|---|---:|---|
| Drop the **final comment** of every conversation | 6,842 | Derailed conversations end at the removed comment, so the last utterance *is* the label. It is dropped from civil conversations too, so input length does not reveal the label. |
| Drop moderator removal notices ("your comment has been removed", "Comment Rule 2") | 5 | They state the label. The pattern is kept narrow, because broader words such as "removed", "mods" and "reported" appear about equally in civil (106 / 68 / 12) and derailed (117 / 73 / 26) threads, mostly in ordinary arguments. |
| Drop `[deleted]` / `[removed]` texts inside the context | 926 | No content. |
| Skip conversations with no context left | 22 | Nothing to score. |

**Leakage demo.** Leaving the final comment in raises TF-IDF's test AUC from 0.611 to 0.648, which is about as much as switching from TF-IDF to embeddings. Without that rule, a model would look better while partly reading the moderator's target.

The starter `artifacts.corpora.CgaCmvLoader` uses the first 4 utterances and labels them with `has_removed_comment`. For conversations with 4 or fewer utterances (1,111 of the 3,421 derailed ones), that window includes the removed comment. This module does not change the starter loader. Use `utils.quality.cga_cmv.load_quality_dataset` for quality work instead.

### Splits and leakage checks

We keep the corpus's official split. Each derailed conversation is paired with a civil one that replies to the same original post, and the official split never separates a pair, so the model never trains on the civil twin of a test thread. The bootstrap confidence intervals resample **pairs**, not conversations, for the same reason.

| Check | Result |
|---|---|
| `conversation_disjoint` | pass: 6,820 conversations, each in one split |
| `pair_disjoint` | pass: 3,421 pairs, each in one split |
| `example_disjoint` | pass |
| `label_coverage` | pass |
| `label_distribution` | pass: max gap vs train is 0.0017 (dev) and 0.0006 (test) |
| `text_overlap` | pass: 0 dev or test conversations appear verbatim in train |

The checks reuse `utils/modeling_data/leakage.py`. `pair_disjoint` is the one new check.

## Models

All scorers output `P(civil)` and share one interface: `fit(conversations, civil)` and `predict(conversations)`, where a conversation is its list of comment texts.

| Scorer | Features | Role |
|---|---|---|
| `ConstantScorer` | none (train civil rate) | Floor |
| `LengthScorer` | log of comment count, mean comment length and last comment length | Shows how much is solvable without reading |
| `TfidfScorer` | word 1-2 grams, `min_df=2`, sublinear TF, 50k features | Simple text scorer |
| `EmbeddingScorer` | mean of per-comment embeddings concatenated with the latest comment's embedding, standardized | Sentence-transformer scorer |

Every model uses logistic regression. C is picked on dev ROC-AUC from a grid of 3e-5 to 10, the model is refit on train, and the test split is scored once. Each comment is embedded separately, because joining a thread would hit the encoder's 256 to 384 token limit (on average 5 comments of about 100 words each, with 12% over 200 words). A single very long comment is still truncated at that limit. Embeddings are L2-normalized and cached by text hash under `data/processed/embeddings/` (gitignored).

Embedding models compared:

| Model | Dimensions | Notes |
|---|---:|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 6 layers, small and fast, good for Colab |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 12 layers, a larger general-purpose model, slower than MiniLM |

## Metrics

| Metric | Type | Why it is reported |
|---|---|---|
| **ROC-AUC** (primary) | ranking | Chance a civil thread outscores a derailed one. Threshold-free, and 0.5 is chance on a balanced set. |
| Spearman ρ | ranking | Listed for the quality scorer in the project brief. With a binary target it measures the same ranking as AUC. |
| MAE | level | Listed in the brief. On a binary target it rewards extreme scores, so read it next to Brier. |
| Brier | level | Proper scoring rule for probabilities; 0.25 is the constant baseline. |
| Accuracy at 0.5 | threshold | Easy to read. |
| Calibration table | level | Whether a score of 0.7 means about 70% civil. |

## Results

Full output: `experiments/quality_scoring/results/quality_results.md` and `.json`.

### Main comparison (test, n = 1,364)

| Scorer | C | Dev AUC | Test AUC [95% CI] | Spearman | MAE | Brier | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constant | | 0.500 | 0.500 | 0.000 | 0.500 | 0.250 | 0.501 |
| Length + LR | 10 | 0.574 | 0.559 [0.538, 0.579] | 0.102 | 0.494 | 0.248 | 0.539 |
| TF-IDF + LR | 1 | 0.626 | 0.611 [0.584, 0.638] | 0.192 | 0.483 | 0.241 | 0.579 |
| MiniLM + LR | 3e-4 | 0.652 | 0.647 [0.621, 0.673] | 0.255 | 0.474 | 0.234 | 0.622 |
| **MPNet + LR** | 3e-4 | **0.678** | **0.664 [0.637, 0.689]** | **0.283** | 0.463 | **0.230** | 0.620 |

- Both embedding models beat TF-IDF, and their confidence intervals do not overlap TF-IDF's point estimate. MPNet is ahead of MiniLM by 0.017 AUC, but the two intervals overlap heavily.
- The embedding models need very strong regularization (C = 3e-4 on 768 to 1,536 standardized features with 4k training threads). Dev AUC falls steadily as C grows, from 0.678 down to 0.597 for MPNet at C = 10. Without that search the embeddings would look no better than TF-IDF.
- The length baseline is above chance (0.559). Derailed threads are a little longer (6.6 vs 6.0 utterances on average), so some of every model's signal is thread length, not language.

### How early can we tell? (test ROC-AUC, first k context comments only)

| Scorer | k = 1 | k = 2 | k = 3 | All |
|---|---:|---:|---:|---:|
| TF-IDF + LR | 0.528 | 0.545 | 0.563 | 0.611 |
| MiniLM + LR | 0.516 | 0.534 | 0.564 | 0.647 |
| MPNet + LR | 0.527 | 0.534 | 0.577 | 0.664 |

The opening comment alone is close to chance, which is what the matched-pair design intends, since both threads in a pair reply to the same post. The signal comes from how the back-and-forth develops, and the most recent comments carry the most. This is why `EmbeddingScorer` includes the latest comment's embedding as its own feature.

### Calibration (MPNet + LR, test)

| Score bin | n | Mean score | Civil rate |
|---|---:|---:|---:|
| 0.0–0.2 | 14 | 0.172 | 0.214 |
| 0.2–0.4 | 298 | 0.327 | 0.302 |
| 0.4–0.6 | 749 | 0.504 | 0.510 |
| 0.6–0.8 | 296 | 0.665 | 0.679 |
| 0.8–1.0 | 7 | 0.820 | 0.571 |

The scores are well calibrated where there is data, but cautious: 55% of threads score between 0.4 and 0.6. Ranking threads works better than reading a single score as a verdict.

### What TF-IDF picks up

The strongest derailed-side features are second-person address and profanity (`you re`, `you`, `your`, `yourself`, `fucking`, `shit`) plus dismissive words (`literally`, `nothing`, `no`). The civil side has hedges and questions (`could`, `could you`, `would you`, `seems`, `personally`, `don think`). This matches the politeness-strategy findings that motivated CGA, and suggests the models learn tone rather than topic.

### Embeddings beyond scoring

As a check for later retrieval and clustering work: from a test conversation's mean embedding, how often is its matched pair (a different thread replying to the same post) the nearest neighbor among the other 1,363 test threads?

| Model | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| MiniLM | 0.265 | 0.463 | 0.559 |
| MPNet | 0.304 | 0.511 | 0.605 |

Random retrieval would give about 0.0007 at k=1. Both models clearly capture topic across different threads, and MPNet is better here too. This uses the harness's `retrieval_recall_at_k`.

## Recommendation

1. **Prototype quality scorer:** `EmbeddingScorer` with `all-mpnet-base-v2` and C = 3e-4, writing `P(civil)` to the conversation's utterances as `quality_score` (conversation-level for now).
2. **If compute is tight** (Colab CPU, large threads), use MiniLM. It is about 0.017 AUC behind on test, within the confidence interval, and half the dimensions.
3. **Shared embeddings.** The same cached per-comment embeddings can feed topic clustering and retrieval, so the pipeline only encodes each comment once.

## Limitations

- **Weak signal.** An AUC of 0.66 is useful for ranking a set of threads, not for judging a single one. Forecasting derailment is a hard task. Published models built for it, such as CRAFT (Chang & Danescu-Niculescu-Mizil, 2019), are the natural next comparison, but they use a different setup, so their numbers are not reproduced here.
- **Civility is not argument quality.** A polite thread can still be uninformative. If the team wants argument strength, a labeled corpus such as IBM-ArgQ-Rank-30k would be needed. It is not set up here.
- **Domain shift.** CMV has strict rules and active moderators. On other subreddits, "removed by a moderator" can mean spam or off-topic, so scores will transfer imperfectly.
- **Conversation level only.** Labels are per thread. Per-comment scores would need a different target.
- **Length confound.** Part of the signal is thread length. The length baseline is reported so this stays visible.
- **No fine-tuning yet.** A fine-tuned compact transformer (next step on the modeling ladder) was out of scope for this prototype.

## Reproducing

```bash
# 1. Download the corpus once (about 50 MB zipped, 390 MB unzipped)
uv run --with convokit python -c "from convokit import download; print(download('conversations-gone-awry-cmv-corpus'))"

# 2. Run every experiment (about 6 minutes the first time on an M-series laptop, under 1 minute with cached embeddings)
uv run --no-project --with pydantic --with scikit-learn --with scipy --with sentence-transformers \
    python -m experiments.quality_scoring.run_quality_experiments

# 3. Use in code
from utils.quality.cga_cmv import load_quality_dataset
from utils.quality.embeddings import SentenceTransformerEmbedder
from utils.quality.scorers import EmbeddingScorer

data = load_quality_dataset()
scorer = EmbeddingScorer(SentenceTransformerEmbedder("sentence-transformers/all-mpnet-base-v2"), C=3e-4)
scorer.fit(data.comments("train"), data.civil("train"))
scores = scorer.predict(data.comments("test"))   # P(civil) per conversation
```

Pass `--models sentence-transformers/all-MiniLM-L6-v2` to skip MPNet. Logistic regression is deterministic and the split is fixed by the corpus, so reruns give the same numbers on the same library versions. The JSON results record Python, platform and package versions.

## Tests

```bash
uv run --no-project --with pydantic --with scikit-learn --with scipy --with pytest pytest tests -q
```

`tests/test_quality.py` uses a small synthetic corpus and a fake embedder, so it needs no download or model. It covers:

- the final comment never reaching the model's input
- reply-chain ordering regardless of file order
- context filtering and exclusion counts
- the official split, with `val` mapped to dev
- prefix truncation
- moderator-notice removal
- pair grouping and the `pair_disjoint` check catching a split pair
- each scorer's output range and separability
- metric values on known inputs
- pair-level bootstrap
- the embedding cache round trip
