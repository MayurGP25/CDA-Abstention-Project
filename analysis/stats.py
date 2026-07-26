"""Statistics: paired tests, bootstrap CIs, AUROC. Pure numpy/scipy."""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_paired(a, b):
    """Paired Wilcoxon signed-rank; returns (statistic, p, median_diff)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.allclose(a, b):
        return float("nan"), float("nan"), float(np.median(a - b) if len(a) else np.nan)
    w, p = stats.wilcoxon(a, b)
    return float(w), float(p), float(np.median(a - b))


def bootstrap_ci(x, *, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def auroc(pos_scores, neg_scores):
    """AUROC that `pos_scores` rank above `neg_scores` (Mann-Whitney U)."""
    pos = np.asarray(pos_scores, float)
    neg = np.asarray(neg_scores, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return float(u / (len(pos) * len(neg)))


def cohens_d_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d = d[~np.isnan(d)]
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")
