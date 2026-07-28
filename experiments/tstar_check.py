"""t* anchor check on an EXISTING collect parquet -- READ-ONLY, CPU, no re-run.

Purpose: before spending any GPU on a re-run, verify on data you already have
whether the harmful-vs-benign separation survives when measured at t* (the verb
slot right after the literal "Step 1: ", where refusal is actually masked)
instead of at t0 or averaged over pos < 5 (which is JSON scaffolding).

It finds t* per generation from the logged forced_id sequence, cross-checks it
against the n_allowed jump (1 -> ~verb-count), and reports P_refuse / mu / H_pre
at t0 and t*, plus AUROC(harmful>benign) at each anchor.

Needs only: pos, condition, prompt_id, forced_id, n_allowed, mu, sr, H_pre.
(sr may be NaN at odd positions if collected with stride>1; mu is dense.)

Usage:
  python experiments/tstar_check.py results/depth/qwen25-7b__harmbench.parquet \
      --model Qwen/Qwen2.5-7B-Instruct --anchor "Step 1: "
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import stats  # noqa: E402


def find_tstar(ids, tok, anchor):
    """Smallest t with decode(ids[:t+1]).endswith(anchor); t* = t+1 (the slot
    the grammar forces next -- the verb, where a refusal opener is forbidden)."""
    for t in range(len(ids)):
        if tok.decode(ids[: t + 1]).endswith(anchor):
            return t + 1
    return None


def _p_refuse(sr):
    return float(np.clip(np.exp(sr), 0.0, 1.0)) if sr == sr else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--model", required=True, help="hf id for the tokenizer")
    ap.add_argument("--anchor", default="Step 1: ")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    df = pd.read_parquet(args.parquet)
    need = {"pos", "condition", "prompt_id", "forced_id", "n_allowed", "mu", "H_pre"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"parquet missing columns: {missing}")
    has_sr = "sr" in df.columns

    rows, n_none = [], 0
    for (cond, pid), g in df.groupby(["condition", "prompt_id"]):
        g = g.sort_values("pos").reset_index(drop=True)
        ids = g["forced_id"].tolist()
        ts = find_tstar(ids, tok, args.anchor)
        if ts is None or ts >= len(g):
            n_none += 1
            continue
        r0, rt = g.iloc[0], g.iloc[ts]
        n_allowed_before = int(g.iloc[ts - 1]["n_allowed"]) if ts >= 1 else -1
        rows.append(dict(
            condition=cond, prompt_id=pid, tstar=ts,
            n_allowed_before=n_allowed_before, n_allowed_tstar=int(rt["n_allowed"]),
            mu_t0=float(r0["mu"]), mu_tstar=float(rt["mu"]),
            sr_t0=float(r0["sr"]) if has_sr else float("nan"),
            sr_tstar=float(rt["sr"]) if has_sr else float("nan"),
            Hpre_t0=float(r0["H_pre"]), Hpre_tstar=float(rt["H_pre"]),
        ))
    A = pd.DataFrame(rows)
    print(f"\n===== t* check :: {args.parquet} =====")
    print(f"anchor = {args.anchor!r}   t* found: {len(A)} / {len(A) + n_none} generations")
    if A.empty:
        sys.exit("no t* found -- check the anchor string / grammar")

    q = A["tstar"].quantile([0, .5, 1]).astype(int).to_dict()
    print(f"t* position  (min/median/max): {q[0.0]} / {q[0.5]} / {q[1.0]}")
    print("anchor sanity -- n_allowed should jump 1 -> ~verb-count AT t*:")
    print(f"  n_allowed just BEFORE t* (median): {int(A['n_allowed_before'].median())}   (want ~1)")
    print(f"  n_allowed AT t*          (median): {int(A['n_allowed_tstar'].median())}   (want >>1, ~verb count)")

    hf = A[A.condition == "harmful_forced"]
    bf = A[A.condition == "benign_forced"]
    print(f"\nharmful={len(hf)}  benign={len(bf)}")
    print("\n-- does the separation survive at t*?  AUROC(harmful>benign) --")
    for sig in ["mu", "sr", "Hpre"]:
        for anch in ["t0", "tstar"]:
            col = f"{sig}_{anch}"
            h, b = hf[col].dropna(), bf[col].dropna()
            if len(h) and len(b):
                extra = ""
                if sig == "sr":
                    extra = f"  P_refuse_h={_p_refuse(h.mean()):.3f} P_refuse_b={_p_refuse(b.mean()):.3f}"
                print(f"  AUROC({sig:4s} @ {anch:5s}) = {stats.auroc(h, b):.3f}   "
                      f"mean_h={h.mean():+.3f} mean_b={b.mean():+.3f}{extra}")
            elif sig == "sr":
                print(f"  (sr @ {anch}: too few non-NaN -- stride>1 skipped this position)")

    out = Path(args.parquet).with_suffix("").as_posix() + "__tstar.csv"
    A.to_csv(out, index=False)
    print(f"\nper-generation t* table -> {out}")
    print("\nREAD: if AUROC(mu/sr @ tstar) stays high, the headline survives the "
          "clean measurement and the re-run is worth it. If it collapses, that is "
          "itself the finding (surface, not latent) -- and cheap to have learned now.")


if __name__ == "__main__":
    main()
