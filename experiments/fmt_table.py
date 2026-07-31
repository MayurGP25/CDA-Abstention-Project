"""Format-instruction sweep: the t0 table.

Reads every parquet under results/fmt/ and reports, per (model, format level,
condition), the quantities the paper needs at t0:

    R_t          retained mass = exp(-D), median [IQR] and bootstrap CI
    -log2 R_t    the same number in bits -- this is KL(P_mask || P_free), and it
                 is the numerically stable form. Report this as primary: a
                 reviewer cannot object to bf16 precision at 24.9 bits the way
                 they can at 3.6e-11.
    H_pre        entropy of the model's own distribution
    H_post       entropy of the served distribution (bounded by log2|A_t|)
    P_refuse     exp(sr), the refusal mass under P_free

No GPU. Run after scripts/run_fmt_sweep.sh.

    python experiments/fmt_table.py
    python experiments/fmt_table.py --root results/fmt --csv results/fmt/t0.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from analysis.stats import auroc, bootstrap_ci_median  # noqa: E402

LEVEL_ORDER = ["none", "neutral", "terse", "schema"]


def load(root: Path) -> pd.DataFrame:
    """All shards under `root`, restricted to t0.

    Reads shards/ rather than the merged parquet so a sweep that is still
    running is still readable -- the merged file only appears at the end of a
    level, and waiting for all four to finish before looking is how you discover
    a broken arm an hour late.
    """
    files = sorted(root.rglob("shards/*.parquet")) or sorted(root.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {root} -- has the sweep run?")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    if "fmt" not in df.columns:
        raise SystemExit(
            "no `fmt` column: these shards predate the format-instruction axis. "
            "They are the unprompted runs; collect the sweep into results/fmt/.")

    df = df[df["pos"] == 0].copy()
    df["R_mass"] = np.exp(-df["D"].astype(float))
    # Bits, i.e. KL(P_mask || P_free). Guarded so a hypothetical R=0 row prints
    # as inf rather than crashing the table.
    with np.errstate(divide="ignore"):
        df["KL_bits"] = -np.log2(df["R_mass"].where(df["R_mass"] > 0))
    df["P_refuse"] = np.exp(df["sr"].astype(float))
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, fmt, cond), g in df.groupby(["model", "fmt", "condition"]):
        r = g["R_mass"].to_numpy(float)
        med, lo, hi = bootstrap_ci_median(r)
        rows.append(dict(
            model=model, fmt=fmt, condition=cond, n=len(g),
            n_allowed=float(g["n_allowed"].median()),
            R_median=med, R_lo=lo, R_hi=hi,
            R_q1=float(np.nanpercentile(r, 25)), R_q3=float(np.nanpercentile(r, 75)),
            KL_bits_median=float(g["KL_bits"].median()),
            H_pre=float(g["H_pre"].mean()),
            H_post=float(g["H_post"].mean()),
            P_refuse=float(g["P_refuse"].mean()),
        ))
    out = pd.DataFrame(rows)
    out["_o"] = out["fmt"].map({k: i for i, k in enumerate(LEVEL_ORDER)}).fillna(99)
    return out.sort_values(["model", "condition", "_o"]).drop(columns="_o")


def render(t: pd.DataFrame) -> str:
    lines = ["| model | fmt | cond | n | |A_t| | R median [IQR] | KL bits | "
             "H_pre | H_post | P_refuse |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        lines.append(
            f"| {r.model} | {r.fmt} | {r.condition} | {r.n} | {r.n_allowed:.0f} | "
            f"{r.R_median:.2e} [{r.R_q1:.1e}, {r.R_q3:.1e}] | "
            f"{r.KL_bits_median:.1f} | {r.H_pre:.2f} | {r.H_post:.2f} | "
            f"{r.P_refuse:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / "fmt"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    df = load(Path(args.root))
    t = summarise(df)
    print("\n===== retained mass at t0, by format instruction =====")
    print(render(t))

    # The claim RQ1 rests on: R_t describes the constraint, not the input. It
    # survives only if harmful and benign stay indistinguishable at EVERY level.
    print("\n===== does R_t separate harmful from benign at t0? =====")
    print("(0.5 = no separation, which is the result we are claiming)")
    keys = sorted(df.groupby(["model", "fmt"]).groups,
                  key=lambda k: (k[0], LEVEL_ORDER.index(k[1])
                                 if k[1] in LEVEL_ORDER else 99))
    for model, fmt in keys:
        g = df[(df["model"] == model) & (df["fmt"] == fmt)]
        h = g[g["condition"] == "harmful_forced"]["R_mass"].to_numpy(float)
        b = g[g["condition"] == "benign_forced"]["R_mass"].to_numpy(float)
        if len(h) and len(b):
            print(f"  {model:<14} fmt={fmt:<8} AUROC(R_t) = {auroc(h, b):.3f}   "
                  f"AUROC(H_pre) = "
                  f"{auroc(g[g.condition == 'harmful_forced']['H_pre'].to_numpy(float), g[g.condition == 'benign_forced']['H_pre'].to_numpy(float)):.3f}")

    # Cross-check against the depth runs: the `none` arm here is the same
    # measurement Table 1 makes at t0, so a disagreement means the new prompting
    # path perturbed something it should not have.
    print("\nCross-check: fmt=none R_median above must match the t0 row of the "
          "depth runs.\n(If it does not, stop -- the prompting change altered "
          "the unprompted condition.)")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
