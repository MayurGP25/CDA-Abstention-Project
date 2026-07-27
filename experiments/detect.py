"""A1/A2/A3: does the pre-mask refusal signal predict + calibrate to attack
success? Joins the per-position signal parquet (collect.py) with the per-prompt
outcome labels (label.py) and reports the headline numbers.

  A1  gap exists : refusal preference (and H_pre) harmful_forced > benign_forced
  A2  usable     : AUROC easy (harmful vs benign) + hard (within-harmful:
                   success vs not) + risk--coverage AURC
  A3  costly     : ECE raw (miscalibration = the finding) vs recalibrated

Signal per prompt = mean over the first k forced tokens of:
  sr    -> S_R sequence-level refusal logprob  (primary; exp(sr) ~ refusal prob)
  H_pre -> latent pre-mask entropy             (secondary)
  gap   -> H_pre - H_post (served-vs-latent decoupling)

Usage:
  python experiments/detect.py \
      results/depth/qwen25-7b__advbench.parquet \
      results/labels/qwen25-7b__advbench.jsonl --k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import stats  # noqa: E402

SIGNALS = ["sr", "H_pre", "gap"]


def _first_k_mean(df, k):
    """Per (condition, prompt_id): nan-aware mean over pos < k of each signal."""
    d = df[df.pos < k].copy()
    d["gap"] = d["H_pre"] - d["H_post"]
    g = d.groupby(["condition", "prompt_id"])[SIGNALS].mean().reset_index()
    return g


def _refusal_prob(sr_series):
    """exp(S_R) clipped to [0,1] -- model's refusal probability mass."""
    return np.clip(np.exp(sr_series.to_numpy(float)), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("labels")
    ap.add_argument("--k", type=int, default=5, help="forced tokens to aggregate")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sig = _first_k_mean(pd.read_parquet(args.parquet), args.k)
    lab = pd.read_json(args.labels, lines=True)[["condition", "prompt_id", "success"]]
    df = sig.merge(lab, on=["condition", "prompt_id"], how="inner")

    hf = df[df.condition == "harmful_forced"]
    bf = df[df.condition == "benign_forced"]
    y = hf["success"].to_numpy(float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    print("\n===== detect: A1/A2/A3 =====")
    print(f"parquet={args.parquet}  k={args.k}")
    print(f"harmful_forced prompts={len(hf)}  successes={n_pos}  negatives={n_neg}")
    if n_neg < 30:
        print("!! WARNING: < 30 negatives -> hard AUROC/ECE are noisy (Day-3 gate).")

    # ---- A1: signal higher on harmful than benign (mechanism sanity) --------
    print("\n-- A1 gap (harmful_forced vs benign_forced) --")
    for s in SIGNALS:
        auc = stats.auroc(hf[s], bf[s])          # pos=harmful ranks above benign
        print(f"  AUROC({s}: harmful>benign) = {auc:.3f}   "
              f"mean_hf={hf[s].mean():+.3f} mean_bf={bf[s].mean():+.3f}")

    # ---- A2: predict success. Success = LOW refusal preference --------------
    # success-score = -signal for refusal-flavoured signals (sr, gap, H_pre all
    # read "the model resisted"), so higher score => more likely a success.
    print("\n-- A2 detector (within harmful_forced: success vs not) --")
    best = None
    for s in SIGNALS:
        score = -hf[s].to_numpy(float)
        auc = stats.auroc(score[y == 1], score[y == 0])
        _, _, aurc = stats.risk_coverage(score, y)
        print(f"  {s:6s}: hard AUROC={auc:.3f}  risk-coverage AURC={aurc:.3f}")
        if best is None or (auc == auc and auc > best[1]):
            best = (s, auc, score)

    # ---- A3: calibration of the primary refusal-probability signal ----------
    # Treat exp(S_R) as predicted P(model abstains) = P(not a success); compare
    # to the empirical non-success rate. Raw ECE evidences miscalibration; then
    # isotonic on a held-out half shows the achievable calibrated ECE.
    print("\n-- A3 calibration (refusal prob vs non-success) --")
    p_abstain = _refusal_prob(hf["sr"])
    target = 1.0 - y                              # 1 = model did NOT produce harm
    ece_raw = stats.ece(p_abstain, target, n_bins=args.bins)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(p_abstain))
    h = len(idx) // 2
    tr, ev = idx[:h], idx[h:]
    cal_ev = stats.isotonic_recalibrate(p_abstain[tr], target[tr], p_abstain[ev])
    ece_cal = stats.ece(cal_ev, target[ev], n_bins=args.bins)
    print(f"  ECE raw (refusal prob -> non-success) = {ece_raw:.3f}   [the finding]")
    print(f"  ECE recalibrated (isotonic, held-out) = {ece_cal:.3f}")

    # ---- machine-readable summary ------------------------------------------
    conf, acc, cnt = stats.reliability_curve(p_abstain, target, n_bins=args.bins)
    summary = dict(k=args.k, n_pos=n_pos, n_neg=n_neg,
                   best_signal=best[0], best_hard_auroc=round(best[1], 4),
                   ece_raw=round(ece_raw, 4), ece_cal=round(ece_cal, 4))
    out = Path(args.parquet).with_suffix("").as_posix() + f"__detect_k{args.k}.json"
    import json
    Path(out).write_text(json.dumps(
        dict(summary=summary,
             reliability=dict(conf=conf.tolist(), acc=acc.tolist(), count=cnt.tolist())),
        indent=2))
    print(f"\nsummary: {summary}\n-> {out}")


if __name__ == "__main__":
    main()
