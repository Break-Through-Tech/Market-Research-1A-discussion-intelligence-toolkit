# November integration validation

Local validation on 2026-11-07 (America/Phoenix), branch `rohan/november-toolkit-integration`, audit base `e734489`. Implementation was prepared by the coding agent for Rohan's review. No teammate model was retrained for these checks.

## Actual executed commands and results

| Command/check | Observed result |
|---|---|
| `git status --short` and `git branch --show-current` before edits | Requested branch; only the four protected untracked exploration files |
| `git fetch origin` and `git rev-list --left-right --count HEAD...origin/main` | Main unchanged at `e734489`; `0 0` before commits; no merge/rebase |
| `python3 -m pytest -q` before edits | **36 passed in 0.17s** in the initial audit; no existing failures |
| `uv sync` | Lock created; 136 packages resolved including optional extras; **19 packages installed** for core + dev; no optional models installed/downloaded |
| `uv run pytest -q` after installing the package, before new tests | **36 passed in 0.73s** |
| `uv run --offline ruff check src/discussion_intelligence tests/integration examples/fitted_model_integration.py` | **All checks passed** |
| `uv run --offline ruff format --check src/discussion_intelligence tests/integration examples/fitted_model_integration.py` | **15 files already formatted** |
| `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline pytest -q` | **119 passed in 1.12s**, 0 failed; includes all original tests |
| `uv run --offline pytest --collect-only -q` | **119 tests collected** |
| `uv build` and subsequent `uv build --quiet` | Source archive and universal wheel built successfully |
| Archive inventory using `zipfile`/`tarfile` | Protected files and raw corpora absent; source archive includes test fixtures and small reference JSONL |
| Isolated `uv run --offline --no-project --with <absolute-wheel-path> python -c ...` from `/tmp` | Package loaded from uv site-packages outside checkout; `artifacts.connector` imported; RAID query returned `['reddit:t1_c6fwalq']` |
| Unpacked source archive tested in an isolated offline wheel + pytest environment | **119 passed in 29.38s** outside the checkout |
| `cmp examples/buildapc-report.json /tmp/rohan-report-repeat.json` | Exit 0: byte-identical repeated JSON demo output |
| `git diff --check` | Exit 0 |
| Repeated SHA-256 checks of the four exploration files | Identical to pre-edit hashes listed below |

The first preimplementation run reported **36 passed in 0.17s**. Timing is environment-dependent; test counts and exit status are the evidence. The baseline was measured using the pre-existing Python 3.13 environment; final package validation uses uv's Python 3.12.15.

## Reproduced real-data demos

```bash
uv run --offline discussion-intelligence \
  --input data/coarse-discourse-reference-sample.jsonl \
  --query 'RAID data drives' --top-k 3 --format json \
  --output examples/buildapc-report.json
uv run --offline discussion-intelligence \
  --input data/coarse-discourse-reference-sample.jsonl \
  --query 'RAID data drives' --top-k 3 --format markdown \
  --output examples/buildapc-report.md
uv run --offline python -m discussion_intelligence \
  --input data/coarse-discourse-reference-sample.jsonl \
  --conversation-index 1 --format json --output /tmp/rohan-karma-report.json
```

Results: `reddit:t3_10r2f8` has 7 usable evidence utterances; its RAID query ranks the real reply `reddit:t1_c6fwalq` first. `reddit:t3_169c79` has 4 usable evidence utterances and produces representative excerpts. Both reports explicitly show discourse/quality/clustering/embeddings unavailable in the default mode. All substantive points cite real utterances. No annotated gold label is used as a model prediction. Evidence dates are null and missing-author limitations are visible.

The artifact example was executed with a tiny fitted text classifier created once in `/tmp/rohan-synthetic-test-bundle.joblib`:

```bash
uv run --offline python examples/fitted_model_integration.py \
  --bundle /tmp/rohan-synthetic-test-bundle.joblib \
  --input data/coarse-discourse-reference-sample.jsonl \
  > /tmp/rohan-fitted-adapter-report.json
```

It produced a report with discourse `available`. This is a synthetic wiring check, not the teammates' production artifact or a quality metric. The bundle is not committed.

To repeat the wheel smoke test after `uv build`, run from `/tmp` (substitute your checkout path):

```bash
uv run --offline --no-project \
  --with /absolute/checkout/dist/discussion_intelligence_toolkit-0.1.0-py3-none-any.whl \
  discussion-intelligence \
  --input /absolute/checkout/data/coarse-discourse-reference-sample.jsonl \
  --query 'RAID data drives' --top-k 1 --format json
```

Initial dependency installation may need network; `--offline` requires cached compatible dependencies. The normal demo does not download any model.

## Dependencies actually used

| Package | Local validated version | Role |
|---|---|---|
| Python | 3.12.15 | uv environment; package supports >=3.11 |
| pydantic | 2.14.3 | Existing schema and strict report validation |
| numpy | 2.5.4 | Embedding arrays and numeric checks |
| scipy | 1.18.1 | Sparse target/parent features; sklearn dependency |
| scikit-learn | 1.9.1 | TF-IDF, cosine similarity, fitted-model adapter tests |
| pytest | 9.2.0 | Development tests |
| ruff | 0.17.4 | Development lint and formatting |

Core dependencies are pydantic, numpy, scipy, scikit-learn. `setuptools>=77` is the build requirement. New optional extras are `embeddings` (sentence-transformers), `reddit` (pandas, pyarrow, requests), and `corpora` (ConvoKit). `uv.lock` resolves optional dependencies without installing them in the default environment. No API-client/key dependency is required. Original `requirements.txt` remains an empty compatibility scaffold.

## Corrected interim failures

- A new nested `conftest.py` shadowed the existing tests' direct `from conftest import raw_utt`. The shared fixture was moved to the original `tests/conftest.py`; collection then succeeded.
- One new Markdown assertion expected raw underscores even though link labels escape Markdown. It now verifies actual encoded citation targets. Report IDs render clearly in code elements.
- The first archive-inventory script attempted to open uv's `dist/.gitignore` as a tar archive. The corrected script filters `*.whl` and `*.tar.gz` before checking inventory. Both archives pass.
- Final citation review found that encoded HTML anchor IDs could disagree with browser-decoded link fragments. Anchor IDs now preserve the actual canonical ID, and an HTML-parser test checks every decoded link target.
- Staged diff checks found trailing whitespace at an excerpt cutoff and an extra Markdown EOF newline. Extraction/rendering now strip those edges; generated examples were regenerated and checked.
- Ruff identified import ordering during implementation; fixes were applied before the final passing lint/format run.

No baseline failure was hidden or marked as passed. All final local tests pass. No full corpus, model checkpoint, network-dependent benchmark, GPU evaluation, or paid API was used.

## Preserved local files

These files remain untracked and were neither moved, modified, staged, committed, nor packaged:

```text
33559f7ec8eacf0ec8c72f1fd34e13616cf313f1a9b51f2ad342f0ce0d040720  .DS_Store
7b235b1158aedcff13cf75e2aedf8670ddb05e6d286f21f70f451439d4e33172  leagueoflegends_data.csv
e8baf77670b605706f813fdae2ac885ad7a4df5f11c8f2ee9d6fa4b2c53f74a9  subreddit_conversation_counts.txt
26c6f82151a90c9afa2c43c551b9b64621423eb79b99cda1e88a3b9103d6a49a  subreddits_small_sample.txt
```

## Boundaries and remaining validation

The 119 tests establish integration behavior, not model accuracy, retrieval recall on a held-out benchmark, or human-rated summary quality. Franceli's recorded test macro F1 and Gia's PR #22 results belong to their experiments and were not reproduced here. Full-corpus CGA-CMV counts/splits were not rebuilt. The starter-loader fix has synthetic regression coverage; Gia's official-split loader remains a separate reviewed benchmark path.

CI is configured for Python 3.11–3.13 on Linux. Local success does not imply those remote jobs passed; inspect the draft PR's checks. Installing optional extras and loading real model-owner artifacts remain integration follow-ups.
