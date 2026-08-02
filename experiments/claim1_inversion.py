"""Claim 1, and nothing else: does constrained decoding invert the sign of
token-level uncertainty?

THE CLAIM. At the first generated token under an unannounced grammar, the
model's own entropy H_pre is LOWER on harmful prompts (it is confident, it wants
to refuse) while the served entropy H_post is HIGHER on those same prompts. If
that holds, an uncertainty-gated abstention rule reading H_post routes exactly
the wrong queries.

Operationally the claim is a sign statement about two AUROCs:

    AUROC(H_pre)  < 0.5 < AUROC(H_post)

with neither confidence interval crossing 0.5. AUROC is used rather than a mean
difference because it is scale-free, so the two entropies are compared on equal
terms without either being rescaled.

Restricted to the unannounced regime (fmt in {none, neutral}) and to
forced_steps, because that is the regime the claim is about. The aligned levels
are the scope boundary and belong in their own analysis.

No GPU. Reads the dump cache if present.

    python3 experiments/claim1_inversion.py
    python3 experiments/claim1_inversion.py --refresh
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
from dump_all import auroc, load_all  # noqa: E402

RNG = np.random.default_rng(20260802)
B = 10000
REGIME = ("none", "neutral")


def auroc_ci(pos, neg, b=B, alpha=0.05):
    """Percentile bootstrap, resampling both groups independently.

    Stratified rather than pooled: the two arms are fixed by design (50 harmful,
    50 benign), so the sampling distribution to imitate is one that holds those
    sizes, not one that lets the split wander.
    """
    pos = np.asarray(pos, float); pos = pos[~np.isnan(pos)]
    neg = np.asarray(neg, float); neg = neg[~np.isnan(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), float("nan"), float("nan")
    point = auroc(pos, neg)
    draws = np.empty(b)
    for i in range(b):
        draws[i] = auroc(RNG.choice(pos, len(pos), replace=True),
                         RNG.choice(neg, len(neg), replace=True))
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cache", default=str(ROOT / "results" / "_dump_cache.parquet"))
    ap.add_argument("--only", default=None,
                    help="comma-separated results/ subdirs. AFTER the log-space "
                         "recollection both fmt/ and fmt2/ exist and hold the same "
                         "conditions, so without this the dedupe keeps whichever "
                         "loaded first -- possibly the censored one. Pass --only fmt2.")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    if only:
        args.refresh = True      # the cache holds whatever the last run loaded
    df = load_all(Path(args.cache), args.refresh, only)
    d = df[(df["pos"] == 0)
           & (df["fmt"].isin(REGIME))
           & (df["grammar"] == "forced_steps")
           & (df["condition"].isin(["harmful_forced", "benign_forced"]))].copy()

    # The depth and fmt runs measure the SAME t0 on the SAME prompts, so pooling
    # them would double n and shrink every interval by ~sqrt(2) for free.
    before = len(d)
    d = d.drop_duplicates(subset=["model", "fmt", "grammar", "arm", "prompt_id"])
    if before != len(d):
        print(f"[dedupe] dropped {before - len(d)} duplicate prompt-level rows "
              f"(depth and fmt measure the same t0)\n")

    print("=" * 78)
    print("CLAIM 1: constrained decoding inverts the sign of token uncertainty")
    print("=" * 78)
    print("Regime: unannounced grammar (fmt in {none, neutral}), forced_steps, t0.")
    print("Benign control: Alpaca. 95% percentile bootstrap, 10k resamples.\n")

    print("--- the two entropies, in bits ---")
    tab = (d.groupby(["model", "fmt", "arm"])
             .agg(n=("H_pre", "size"),
                  H_pre=("H_pre", "mean"), H_post=("H_post", "mean"),
                  A=("n_allowed", "median"))
             .round(3))
    print(tab.to_string())

    print("\n--- the sign test ---")
    print("| model | fmt | n_h/n_b | AUROC(H_pre) [95% CI] | AUROC(H_post) [95% CI] "
          "| inverted? |")
    print("|" + "---|" * 6)
    verdicts = []
    for (m, f), g in d.groupby(["model", "fmt"]):
        hf = g[g["condition"] == "harmful_forced"]
        bf = g[(g["condition"] == "benign_forced") & (g["prompt_source"] == "alpaca")]
        if hf.empty or bf.empty:
            continue
        pre = auroc_ci(hf["H_pre"], bf["H_pre"])
        post = auroc_ci(hf["H_post"], bf["H_post"])
        # The claim is not "the two differ" but "they sit on OPPOSITE sides of
        # 0.5, and neither interval touches it".
        ok = pre[2] < 0.5 < post[1]
        verdicts.append(ok)
        print(f"| {m} | {f} | {len(hf)}/{len(bf)} | "
              f"{pre[0]:.3f} [{pre[1]:.3f}, {pre[2]:.3f}] | "
              f"{post[0]:.3f} [{post[1]:.3f}, {post[2]:.3f}] | "
              f"{'YES' if ok else 'no'} |")

    print("\n--- per-prompt view: is the gap H_post - H_pre wider on harmful? ---")
    print("(the same claim without AUROC, so it does not rest on one statistic)")
    d = d.assign(gap=d["H_post"] - d["H_pre"])
    print("| model | fmt | gap harmful | gap benign | AUROC(gap) [95% CI] |")
    print("|" + "---|" * 5)
    for (m, f), g in d.groupby(["model", "fmt"]):
        hf = g[g["condition"] == "harmful_forced"]
        bf = g[(g["condition"] == "benign_forced") & (g["prompt_source"] == "alpaca")]
        if hf.empty or bf.empty:
            continue
        a = auroc_ci(hf["gap"], bf["gap"])
        print(f"| {m} | {f} | {hf['gap'].mean():+.3f} | {bf['gap'].mean():+.3f} | "
              f"{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] |")

    print("\n--- the mechanism, one number ---")
    print("H_post is bounded by log2|A_t|, which the GRAMMAR AUTHOR sets. If the")
    print("bound is what is being measured, H_post should sit near it and R_t")
    print("should be tiny. Both, per cell:")
    mech = d.groupby(["model", "fmt", "arm"]).agg(
        R_median=("R", "median"),
        H_post=("H_post", "mean"),
        ceiling=("n_allowed", lambda s: float(np.log2(max(s.median(), 1)))))
    mech["frac_of_ceiling"] = (mech["H_post"] / mech["ceiling"]).round(3)
    print(mech.round(4).to_string())

    print("\n" + "=" * 78)
    if verdicts and all(verdicts):
        print("VERDICT: inversion holds in every cell, no interval touching 0.5.")
        print("Claim 1 stands. This is the paper.")
    elif any(verdicts):
        print("VERDICT: inversion holds in SOME cells. Report only those, and")
        print("say plainly where it does not hold.")
    else:
        print("VERDICT: inversion does not survive the intervals. Do not write")
        print("this claim.")
    print("=" * 78)


if __name__ == "__main__":
    main()
