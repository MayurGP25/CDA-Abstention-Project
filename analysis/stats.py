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


# ---------------------------------------------------------------------------
# Selective prediction + calibration (A2 / A3).
# `scores` are the confidence in the POSITIVE class (attack success); `labels`
# are 0/1. NaNs are dropped pairwise. All pure numpy.
# ---------------------------------------------------------------------------

def _clean(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels, float)
    m = ~(np.isnan(s) | np.isnan(y))
    return s[m], y[m]


def reliability_curve(scores, labels, *, n_bins=10):
    """Equal-width reliability curve on probability-like scores in [0, 1].

    Returns (bin_conf, bin_acc, bin_count) with NaN in empty bins. Feed RAW
    (pre-recalibration) scores here to VISUALISE miscalibration -- that curve is
    the A3 finding; recalibrate only for the reported post-hoc number.
    """
    s, y = _clean(scores, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins)
    for b in range(n_bins):
        sel = idx == b
        cnt[b] = sel.sum()
        if cnt[b]:
            conf[b] = s[sel].mean()
            acc[b] = y[sel].mean()
    return conf, acc, cnt


def ece(scores, labels, *, n_bins=10):
    """Expected Calibration Error (equal-width bins). Scores must be in [0, 1]
    (probabilities). Report ECE on the RAW score to evidence miscalibration,
    and again on the recalibrated score to show the fix."""
    conf, acc, cnt = reliability_curve(scores, labels, n_bins=n_bins)
    tot = cnt.sum()
    if tot == 0:
        return float("nan")
    ok = ~np.isnan(conf)
    return float((cnt[ok] * np.abs(acc[ok] - conf[ok])).sum() / tot)


def risk_coverage(scores, labels):
    """Selective-prediction risk--coverage curve.

    Higher `score` = more confident the generation is a successful attack, so we
    abstain on the LOWEST scores first. At each coverage c (fraction answered,
    most-confident first) risk = error rate on the answered set, where "error" =
    (1 - label) i.e. a non-successful generation we failed to abstain on.

    Returns (coverage, risk, aurc): arrays over the sorted thresholds plus the
    scalar area under the risk--coverage curve (lower AURC = better signal).
    """
    s, y = _clean(scores, labels)
    n = len(s)
    if n == 0:
        return np.array([]), np.array([]), float("nan")
    order = np.argsort(-s)                      # most-confident first
    y = y[order]
    err = 1.0 - y                               # answered a non-success = error
    cum_err = np.cumsum(err)
    k = np.arange(1, n + 1)
    risk = cum_err / k
    coverage = k / n
    aurc = float(np.trapz(risk, coverage))
    return coverage, risk, aurc


def isotonic_recalibrate(scores_train, labels_train, scores_eval):
    """Fit isotonic regression on a held-out split and map eval scores through it.
    Use for the POST-recalibration ECE only; never fit and evaluate on the same
    split. Returns calibrated eval scores in [0, 1]."""
    from sklearn.isotonic import IsotonicRegression

    st, yt = _clean(scores_train, labels_train)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(st, yt)
    return iso.predict(np.asarray(scores_eval, float))
