"""
Metrics for quality scores against binary civil / derailed labels.

    roc_auc      ranking: chance a civil conversation outscores a derailed one
    spearman_r   ranking, as listed for the quality scorer in the project brief
    mae          level: mean |score - label|, also from the brief
    brier        level: mean (score - label)^2, a proper scoring rule
    accuracy     at a 0.5 threshold, for readers who want a single rate

ROC-AUC is the primary metric. With a binary target, Spearman is a rescaled
point-biserial correlation of the same ranking, and MAE rewards pushing scores
to 0 or 1 whether or not they are right, so Brier is the better level check.

`bootstrap_ci` resamples matched pairs rather than single conversations,
because the two members of a pair share an opening post and are not
independent.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from artifacts.evaluation import regression_metrics


class QualityResult(BaseModel):
    n_examples: int
    roc_auc: float
    spearman_r: float
    mae: float
    brier: float
    accuracy: float
    roc_auc_ci95: tuple[float, float] | None = None


def quality_metrics(civil: Sequence[int], scores: Sequence[float]) -> QualityResult:
    y = np.asarray(civil, dtype=float)
    s = np.asarray(scores, dtype=float)
    if len(y) != len(s):
        raise ValueError(f"Length mismatch: {len(y)} labels vs {len(s)} scores")
    reg = regression_metrics(y.tolist(), s.tolist())
    auc = float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")
    return QualityResult(
        n_examples=len(y),
        roc_auc=auc,
        spearman_r=reg.spearman_r,
        mae=reg.mae,
        brier=float(brier_score_loss(y, s)),
        accuracy=float(accuracy_score(y, s >= 0.5)),
    )


def bootstrap_ci(
    civil: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile interval for ROC-AUC, resampling whole groups.

    `groups` must give both members of a pair the same key; see
    `QualityDataset.groups`.
    """
    y = np.asarray(civil)
    s = np.asarray(scores, dtype=float)
    members: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    blocks = list(members.values())
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_resamples):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        if len(set(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def with_ci(result: QualityResult, civil, scores, groups, **kwargs) -> QualityResult:
    return result.model_copy(update={"roc_auc_ci95": bootstrap_ci(civil, scores, groups, **kwargs)})
