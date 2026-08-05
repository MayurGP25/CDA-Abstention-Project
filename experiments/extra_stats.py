"""The three additions a reviewer asked for, none of which needs new data.

1. QUANTIFIED NULL. "Retained mass is close to prompt-independent" is
   load-bearing, since it is what licenses "shape, not quantity". It is currently
   supported by prose about overlapping quartiles. An AUROC of log R_t with an
   interval containing 0.5 is a much harder claim to argue with than a
   factor-of-two remark.

2. RANK CORRELATION, both pooled and within class. The title claims a rank
   inversion between two measurements, so a rank correlation states it directly.
   A POOLED rho over both arms is not, however, label-free evidence: with two
   clusters it measures the between-cluster separation and so re-reports the
   AUROC in another form. The label-free version is the WITHIN-class rho. If
   that is near zero, the effect is entirely between-class, which is still the
   result but must be said plainly. We report both.

3. AN OUTCOME NUMBER. Every other quantity in the paper is intrinsic to the
   distributions. A cascade escalates when observed uncertainty is high, so at a
   fixed escalation rate we can ask what fraction of prompts receive a different
   decision under the served entropy than under the model's own. That converts
   "this could matter" into a measurement.

No GPU.

    python3 experiments/extra_stats.py --only fmt2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from claim1_inversion import auroc_ci  # noqa: E402
from dump_all import load_all  # noqa: E402

RNG = np.random.default_rng(20260805)
B = 10000
REGIME = ("none", "neutral")
SHORT = {"Qwen2.5-7B-Instruct": "Qwen", "Llama-3.1-8B-Instruct": "Llama"}


def spearman(x, y):
    """Rank correlation as Pearson on ranks, so no scipy dependency is needed.
    Average ranks for ties, which is the standard Spearman convention."""
    x = pd.Series(np.asarray(x, float)).rank().to_numpy()
    y = pd.Series(np.asarray(y, float)).rank().to_numpy()
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_ci(x, y, b=B, alpha=0.05):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return (float("nan"),) * 3
    point = spearman(x, y)
    draws = np.empty(b)
    n = len(x)
    for i in range(b):
        j = RNG.integers(0, n, n)          # pairs resampled together
        draws[i] = spearman(x[j], y[j])
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def disagreement(h_pre, h_post, rate):
    """Fraction of prompts escalated differently at a fixed escalation rate.

    A cascade escalates the top `rate` fraction by observed uncertainty. We take
    the top `rate` by the served entropy and the top `rate` by the model's own,
    and count prompts whose decision differs. Using a matched rate rather than a
    matched threshold is what makes the two comparable, since the two entropies
    live on different supports.
    """
    n = len(h_pre)
    k = max(1, int(round(rate * n)))
    esc_pre = set(np.argsort(-np.asarray(h_pre, float))[:k])
    esc_post = set(np.argsort(-np.asarray(h_post, float))[:k])
    return len(esc_pre ^ esc_post) / n, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="fmt2")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cache", default=str(ROOT / "results" / "_dump_cache.parquet"))
    args = ap.parse_args()

    df = load_all(Path(args.cache), args.refresh,
                  set(args.only.split(",")) if args.only else None)
    d = df[(df["pos"] == 0)
           & (df["fmt"].isin(REGIME))
           & (df["grammar"] == "forced_steps")
           & (df["condition"].isin(["harmful_forced", "benign_forced"]))
           & (df["prompt_source"].isin(["harmbench", "alpaca"]))].copy()
    d = d.drop_duplicates(subset=["model", "fmt", "grammar", "arm", "prompt_id"])
    d["logR"] = -d["R_bits"]                  # log2 R_t, negative
    cells = [(m, f) for m in ("Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct")
             for f in REGIME]

    print("=" * 74)
    print("1. QUANTIFIED NULL: does retained mass separate the arms?")
    print("=" * 74)
    print("An interval CONTAINING 0.5 is the result we want here. It is what")
    print("licenses the claim that the amount removed is not the discriminator.\n")
    print("| model | fmt | AUROC(log2 R_t) [95% CI] | contains 0.5 |")
    print("|" + "---|" * 4)
    nulls = []
    for m, f in cells:
        g = d[(d["model"] == m) & (d["fmt"] == f)]
        hf = g[g["condition"] == "harmful_forced"]["logR"]
        bf = g[g["condition"] == "benign_forced"]["logR"]
        if hf.empty or bf.empty:
            continue
        a, lo, hi = auroc_ci(hf, bf)
        ok = lo <= 0.5 <= hi
        nulls.append(ok)
        print("| %s | %s | %.3f [%.3f, %.3f] | %s |"
              % (SHORT[m], f, a, lo, hi, "yes" if ok else "NO"))
    print("\n%d of %d cells contain 0.5." % (sum(nulls), len(nulls)))
    if not all(nulls):
        print("Where it does not, say so: retained mass carries some signal there")
        print("and the 'shape not quantity' wording has to be softened.")

    print("\n" + "=" * 74)
    print("2. RANK CORRELATION between the two entropy readings")
    print("=" * 74)
    print("Pooled rho mixes two clusters, so a strong negative value there is")
    print("the AUROC restated. WITHIN-class rho is the label-free evidence.\n")
    print("| model | fmt | pooled rho [CI] | harmful only [CI] | benign only [CI] |")
    print("|" + "---|" * 5)
    for m, f in cells:
        g = d[(d["model"] == m) & (d["fmt"] == f)]
        if g.empty:
            continue
        row = ["%s | %s" % (SHORT[m], f)]
        for sel in (g,
                    g[g["condition"] == "harmful_forced"],
                    g[g["condition"] == "benign_forced"]):
            p, lo, hi = spearman_ci(sel["H_pre"], sel["H_post"])
            row.append("%.3f [%.2f, %.2f]" % (p, lo, hi))
        print("| " + " | ".join(row) + " |")
    print("\nIf the within-class values straddle zero while pooled is strongly")
    print("negative, the reversal is a between-class effect. That is still the")
    print("paper's result, but it must be stated that way and not as a")
    print("prompt-by-prompt rank reversal.")

    print("\n" + "=" * 74)
    print("3. OUTCOME: how often would a cascade route differently?")
    print("=" * 74)
    print("Top-k by served entropy against top-k by the model's own, at a matched")
    print("escalation rate over all 100 prompts in the cell.\n")
    print("| model | fmt | rate | k | prompts routed differently |")
    print("|" + "---|" * 5)
    for m, f in cells:
        g = d[(d["model"] == m) & (d["fmt"] == f)]
        if g.empty:
            continue
        for rate in (0.10, 0.20, 0.30, 0.50):
            frac, k = disagreement(g["H_pre"].to_numpy(), g["H_post"].to_numpy(), rate)
            print("| %s | %s | %.0f%% | %d | %.0f%% |"
                  % (SHORT[m], f, 100 * rate, k, 100 * frac))
    print("\nQuote the 50%% row in the conclusion: at that rate the two readings")
    print("disagree about a majority of prompts, which is the concrete form of")
    print("the claim that the ordering is reversed.")


if __name__ == "__main__":
    main()
