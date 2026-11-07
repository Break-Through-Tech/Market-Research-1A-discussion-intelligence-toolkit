## November integration milestone: runnable evidence-linked analysis

The repository had schema/data/model experiments but no installable inference pipeline, evidence retrieval, or report generation. This change adds `analyze(Conversation)` and an offline CLI that turn the tracked sample threads into JSON/Markdown reports with source-linked quotations, without training a model or requiring an API key.

Implementation was prepared by the coding agent on Rohan Verma's existing branch for Rohan and the team to review. Teammates retain authorship of their existing datasets, models and experiments.

### Included

- `src/discussion_intelligence`: canonical tree/identity validation, conservative preprocessing, deterministic TF-IDF/cosine retrieval, structured report/evidence models and extractive synthesis.
- Source IDs, parent links, speaker references, supplied URLs, raw text and provenance retained. Empty/deleted/removed text excluded from retrieval; missing reference timestamps/author identities explicitly flagged.
- Every discussion point must exactly quote its cited source evidence. Invalid, fabricated, duplicate or unretrieved synthesis citations trigger extractive fallback. JSON/Markdown serialization rechecks grounding.
- Working adapters for fitted text classifiers, Franceli-compatible target+parent feature inference, conversation-level civility scoring, supplied topic assignments and `TextEmbedder` retrieval. No training-experiment import, model duplication or implicit checkpoint download.
- P(civil) stays at conversation scope, separate from `Utterance.quality_score`'s argument-quality semantics. Unavailable/failed components are explicit. Labeled excerpts do not assert consensus or unresolved questions.
- Safe adapter for both real reference conversations, CLI entry points, reproducible example reports, trusted-artifact example, `pyproject.toml`, `uv.lock`, optional ML/ingestion extras and package import corrections.
- 83 additional offline tests, alongside all 36 original tests. Lightweight CI for Python 3.11–3.13, lint, demo and builds.
- Starter `CgaCmvLoader` correction: follow the linear chain, exclude the final outcome before applying the context limit, keep reciprocal matched pairs in one hash split, and fail closed on malformed chains. The historical hash policy remains; this is not an official-split benchmark reproduction.
- Detailed learning/review handoff, integration contracts, validation record and an updated README with the correct clone path and actual feature status.

### Team boundaries

Related: #16 / PR #22 (Gia's civility/embeddings) and #17 (Franceli's clustering). These remain separate work. PR #22 is not merged or copied, and no alternative clustering implementation is introduced. The loader correction addresses the specific starter issue identified in PR #22, in a file its implementation leaves unchanged. No issue is automatically closed by this draft.

This advances the **November integration milestone** in `Challenge-Project-Overview.md`: packaging, inference integration, evidence retrieval and grounded report assembly. There was no November GitHub milestone object at audit time.

### Validation actually run

- Before changes: `python3 -m pytest -q` → **36 passed**.
- `uv sync` → locked/installable default environment; no heavy model downloads.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline pytest -q` → **119 passed**, 0 failed.
- `uv run --offline ruff check src/discussion_intelligence tests/integration examples/fitted_model_integration.py` → passed.
- Corresponding `ruff format --check` → passed.
- Both real sample threads analyzed; RAID query's first evidence is `reddit:t1_c6fwalq`. Committed JSON regeneration is byte-identical.
- `uv build` → source archive and wheel built; isolated wheel analysis from `/tmp` succeeds. Package inventories exclude exploration files/raw corpora.
- Fitted-model adapters exercised with tiny real sklearn models; executable joblib example exercised with an uncommitted synthetic fitted bundle.
- `git diff --check` → passed. Original four untracked files have unchanged SHA-256 values and remain outside commits/packages.

### Reproduce

```bash
uv sync --locked
uv run --offline discussion-intelligence \
  --input data/coarse-discourse-reference-sample.jsonl \
  --query "RAID data drives" --top-k 3 --format markdown
uv run --offline pytest -q
```

### Review focus and limitations

- Review the source adapter's sentinel/unknown-speaker handling and the target/parent/civility contracts with the model owners.
- The loader correction intentionally changes short-thread examples and pair-aware hash split assignments. Full-corpus counts were not recomputed; use Gia's official-split loader for her benchmark after its approval.
- The default report is extractive. No free-form LLM synthesis, bundled production classifier, civility checkpoint, embedding model, learned clusters, or new per-comment argument-quality model is claimed.
- Integration tests are not model-accuracy or held-out retrieval/summary-quality evaluation. Real artifact integration and human-reviewed coverage/citation evaluation remain before December.
- CI is configured separately; verify remote job status on this draft.

See `docs/ROHAN_INTEGRATION_HANDOFF.md`, `docs/integration-contracts.md`, and `docs/integration-validation.md` for file-by-file explanations and actual results. No license is added; advisor approval remains required. This PR is a **draft** and must not be automatically merged.
