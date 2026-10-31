"""
Quality scoring experiments on CGA-CMV (issue #16).

    uv run --no-project --with pydantic --with scikit-learn --with scipy \
        --with sentence-transformers \
        python -m experiments.quality_scoring.run_quality_experiments

Steps:

  1. Build the dataset and run the leakage checks (exits 1 on a failed check).
  2. For each scorer, pick C on dev (ROC-AUC), refit on train, score dev and test.
  3. Context ablation: forecast from only the first k comments.
  4. Leakage demo: the same TF-IDF model when the final comment (the removed
     one, for derailed threads) is left in the input.
  5. Embedding check: can a conversation's embedding find its matched pair
     (same opening post) among all test conversations?

Test scores are computed once, after model selection on dev. Results go to
experiments/quality_scoring/results/.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np

from artifacts.evaluation import retrieval_recall_at_k
from utils.quality.cga_cmv import (
    QualityDataset,
    default_corpus_dir,
    load_quality_dataset,
    read_corpus,
    reply_chain,
    run_quality_leakage_checks,
)
from utils.quality.embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder
from utils.quality.metrics import quality_metrics, with_ci
from utils.quality.scorers import ConstantScorer, EmbeddingScorer, LengthScorer, TfidfScorer

log = logging.getLogger("quality")

RESULTS_DIR = Path(__file__).parent / "results"
C_GRID = (0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0)
ABLATION_K = (1, 2, 3)


def select_and_evaluate(make, data: QualityDataset, grid=C_GRID) -> dict:
    """Pick C on dev, then report dev and test metrics for that C."""
    train_x, train_y = data.comments("train"), data.civil("train")
    dev_x, dev_y = data.comments("dev"), data.civil("dev")
    search = {}
    best_c, best_auc, best_model = None, -1.0, None
    for c in grid:
        model = make(c).fit(train_x, train_y)
        auc = quality_metrics(dev_y, model.predict(dev_x)).roc_auc
        search[str(c)] = round(auc, 4)
        if auc > best_auc:
            best_c, best_auc, best_model = c, auc, model
    out = {"C": best_c, "dev_auc_by_C": search if len(grid) > 1 else None}
    for split in ("dev", "test"):
        y, scores = data.civil(split), best_model.predict(data.comments(split))
        out[split] = with_ci(quality_metrics(y, scores), y, scores, data.groups(split)).model_dump()
    out["model"] = best_model
    return out


def with_final_comment(corpus_dir: Path) -> QualityDataset:
    """Every non-empty comment, including the last one. For derailed threads
    the last comment is the one moderators removed, so this leaks the label."""
    from artifacts.corpora import ClassificationExample, ClassificationSplits
    from utils.quality.cga_cmv import SPLIT_NAMES

    conversations, grouped = read_corpus(corpus_dir)
    buckets = {"train": [], "dev": [], "test": []}
    for cid in sorted(grouped):
        meta = conversations[cid]
        comments = [t for t in ((u.get("text") or "").strip() for u in reply_chain(grouped[cid])) if t]
        buckets[SPLIT_NAMES[meta["split"]]].append(ClassificationExample(
            id=cid, text="\n\n".join(comments),
            label="derailed" if meta["has_removed_comment"] else "civil",
            metadata={"comments": comments, "pair_id": meta["pair_id"], "conversation_id": cid},
        ))
    return QualityDataset(splits=ClassificationSplits(**buckets))


def pair_retrieval(embedder, data: QualityDataset, split: str = "test") -> dict:
    """Recall@k for finding a conversation's matched pair by cosine similarity."""
    examples = getattr(data.splits, split)
    vectors = []
    for comments in data.comments(split):
        v = embedder.embed(comments).mean(axis=0)
        vectors.append(v / (np.linalg.norm(v) or 1.0))
    sims = np.vstack(vectors) @ np.vstack(vectors).T
    np.fill_diagonal(sims, -np.inf)
    ids = [ex.id for ex in examples]
    present = set(ids)
    relevant, retrieved = [], []
    for i, ex in enumerate(examples):
        if ex.metadata["pair_id"] not in present:
            continue
        relevant.append({ex.metadata["pair_id"]})
        retrieved.append([ids[j] for j in np.argsort(-sims[i])[:10]])
    return {
        "queries": len(relevant),
        "candidates": len(ids) - 1,
        **{f"recall@{k}": round(retrieval_recall_at_k(relevant, retrieved, k), 4) for k in (1, 5, 10)},
    }


def calibration_table(civil, scores, bins: int = 5) -> list[dict]:
    y, s = np.asarray(civil), np.asarray(scores)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (s >= lo) & ((s < hi) if hi < 1 else (s <= hi))
        if mask.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                         "mean_score": round(float(s[mask].mean()), 3),
                         "civil_rate": round(float(y[mask].mean()), 3)})
    return rows


def fmt_row(name: str, r: dict) -> str:
    t, d = r["test"], r["dev"]
    lo, hi = t["roc_auc_ci95"]
    c = "" if r["C"] is None else f"{r['C']:g}"
    return (f"| {name} | {c} | {d['roc_auc']:.3f} | {t['roc_auc']:.3f} [{lo:.3f}, {hi:.3f}] | "
            f"{t['spearman_r']:.3f} | {t['mae']:.3f} | {t['brier']:.3f} | {t['accuracy']:.3f} |")


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Quality scoring results (CGA-CMV)",
        "",
        "Generated by `experiments/quality_scoring/run_quality_experiments.py`. "
        "Score = P(civil). C picked on dev ROC-AUC; test evaluated once. "
        "95% CI from 1,000 bootstrap resamples of matched pairs.",
        "",
        "## Main results",
        "",
        "| Scorer | C | Dev AUC | Test AUC [95% CI] | Test Spearman | Test MAE | Test Brier | Test acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [fmt_row(name, r) for name, r in report["models"].items()]
    lines += ["", "## Context ablation (first k comments only, test ROC-AUC)", "",
              "| Scorer | " + " | ".join(f"k={k}" for k in ABLATION_K) + " | all |",
              "|---|" + "---:|" * (len(ABLATION_K) + 1)]
    for name, by_k in report["ablation"].items():
        full = report["models"][name]["test"]["roc_auc"]
        lines.append(f"| {name} | " + " | ".join(f"{by_k[str(k)]:.3f}" for k in ABLATION_K) + f" | {full:.3f} |")
    leak = report["leaky_with_final_comment"]
    lines += ["", "## Leakage demo", "",
              f"Same TF-IDF model with the final comment left in (for derailed threads, the removed comment): "
              f"test AUC {leak['test']['roc_auc']:.3f}, versus "
              f"{report['models']['tfidf_lr']['test']['roc_auc']:.3f} with it held out.",
              "", "## Embedding pair retrieval (test)", "",
              "| Model | Queries | Recall@1 | Recall@5 | Recall@10 |", "|---|---:|---:|---:|---:|"]
    for name, r in report["pair_retrieval"].items():
        lines.append(f"| {name} | {r['queries']} | {r['recall@1']:.3f} | {r['recall@5']:.3f} | {r['recall@10']:.3f} |")
    lines += ["", f"## Calibration of best scorer on test ({report['best_model']})", "",
              "| Score bin | n | Mean score | Civil rate |", "|---|---:|---:|---:|"]
    lines += [f"| {r['bin']} | {r['n']} | {r['mean_score']:.3f} | {r['civil_rate']:.3f} |"
              for r in report["calibration"]]
    terms = report["tfidf_top_terms"]
    lines += ["", "## TF-IDF terms with the largest weights", "",
              "| Toward civil | Toward derailed |", "|---|---|"]
    lines += [f"| {c[0]} ({c[1]:+.2f}) | {d[0]} ({d[1]:+.2f}) |" for c, d in zip(terms["civil"], terms["derailed"])]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", type=Path, default=None)
    p.add_argument("--models", default=f"{DEFAULT_MODEL},sentence-transformers/all-mpnet-base-v2",
                   help="Comma-separated sentence-transformer models")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    a = p.parse_args(argv)
    corpus_dir = a.corpus_dir or default_corpus_dir()
    started = time.time()

    data = load_quality_dataset(corpus_dir)
    leakage = run_quality_leakage_checks(data.splits)
    log.info("%s\n%s", data.splits.summary(), leakage.format())
    if not leakage.ok:
        return 1

    embedders = {
        m.split("/")[-1]: SentenceTransformerEmbedder(m) for m in a.models.split(",") if m
    }
    makers = {
        "constant": lambda c: ConstantScorer(),
        "length_lr": lambda c: LengthScorer(C=c),
        "tfidf_lr": lambda c: TfidfScorer(C=c),
        **{f"emb_{n}_lr": (lambda e: lambda c: EmbeddingScorer(e, C=c))(e) for n, e in embedders.items()},
    }
    models = {}
    for name, make in makers.items():
        log.info("Fitting %s", name)
        models[name] = select_and_evaluate(make, data, grid=(None,) if name == "constant" else C_GRID)

    text_models = [n for n in makers if n.startswith(("tfidf", "emb_"))]
    ablation = {}
    for k in ABLATION_K:
        short = load_quality_dataset(corpus_dir, max_comments=k)
        for name in text_models:
            c = models[name]["C"]
            scorer = makers[name](c).fit(short.comments("train"), short.civil("train"))
            auc = quality_metrics(short.civil("test"), scorer.predict(short.comments("test"))).roc_auc
            ablation.setdefault(name, {})[str(k)] = round(auc, 4)

    leaky = select_and_evaluate(makers["tfidf_lr"], with_final_comment(corpus_dir),
                                grid=(models["tfidf_lr"]["C"],))

    best = max(text_models, key=lambda n: models[n]["dev"]["roc_auc"])
    best_scores = models[best]["model"].predict(data.comments("test"))

    report = {
        "dataset": {
            "corpus": "conversations-gone-awry-cmv-corpus",
            "counts": {s: len(getattr(data.splits, s)) for s in ("train", "dev", "test")},
            "civil_rate": {s: round(float(np.mean(data.civil(s))), 4) for s in ("train", "dev", "test")},
            "exclusions": dict(data.exclusions),
            "leakage": leakage.as_dict(),
        },
        "models": {n: {k: v for k, v in r.items() if k != "model"} for n, r in models.items()},
        "ablation": ablation,
        "leaky_with_final_comment": {k: v for k, v in leaky.items() if k != "model"},
        "pair_retrieval": {n: pair_retrieval(e, data) for n, e in embedders.items()},
        "best_model": best,
        "calibration": calibration_table(data.civil("test"), best_scores),
        "tfidf_top_terms": models["tfidf_lr"]["model"].top_terms(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "runtime_seconds": round(time.time() - started, 1),
            **_versions(),
        },
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "quality_results.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, a.out_dir / "quality_results.md")
    log.info("Wrote %s", a.out_dir)
    return 0


def _versions() -> dict[str, str]:
    out = {}
    for pkg in ("numpy", "scikit-learn", "scipy", "sentence-transformers", "torch"):
        try:
            from importlib.metadata import version
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = "not installed"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
