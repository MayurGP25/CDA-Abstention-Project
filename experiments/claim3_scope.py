"""Claims 3 and 4: where the inversion stops, and the two supporting numbers.

C3 SCOPE. Claim 1 is stated for a grammar the prompt does not anticipate. That
is a real restriction and the paper has to say where it ends rather than leave a
reviewer to find it. This sweeps the format-instruction axis and reports, per
level, whether the sign test still passes.

C4 SUPPORT. Two numbers that appear in prose, not in a table:
  - P_refuse on harmful prompts at t0, which is WHY H_pre is low there. Without
    it, "the model is more certain on harmful prompts" is an unexplained fact.
  - the unconstrained refusal rate, which establishes that the disposition being
    measured is one the model acts on when free to.

No GPU.

    python3 experiments/claim3_scope.py --only fmt2
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

LEVELS = ["none", "neutral", "terse", "schema"]
DESC = {"none": "grammar arrives unannounced",
        "neutral": "system turn, no format content",
        "terse": "'output only JSON conforming to the schema'",
        "schema": "the same, plus the schema literal"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cache", default=str(ROOT / "results" / "_dump_cache.parquet"))
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    df = load_all(Path(args.cache), args.refresh, only)
    # The free arm carries no mask, so its R_t is 1.0 by construction and it has
    # no place in a table about what masking does. It survives the grammar
    # filter only because the legacy depth parquet labels it inconsistently --
    # 'forced_steps' for one model, 'none' for the other -- so exclude it by
    # CONDITION, which is what actually distinguishes it.
    d = df[(df["pos"] == 0)
           & (df["grammar"] == "forced_steps")
           & (df["condition"].isin(["harmful_forced", "benign_forced"]))].copy()
    d = d.drop_duplicates(subset=["model", "fmt", "grammar", "arm", "prompt_id"])

    print("=" * 78)
    print("CLAIM 3: SCOPE -- where does the inversion stop holding?")
    print("=" * 78)
    for k in LEVELS:
        print(f"  {k:<8} {DESC[k]}")
    print("\nSign test per level. 'inverted' needs AUROC(H_pre) < 0.5 < AUROC(H_post)")
    print("with neither 95% interval touching 0.5.\n")
    print("| model | fmt | vs | n | AUROC(H_pre) [CI] | AUROC(H_post) [CI] | inverted? |")
    print("|" + "---|" * 7)
    rows = []
    for (m, f), g in d.groupby(["model", "fmt"]):
        hf = g[g["condition"] == "harmful_forced"]
        if hf.empty:
            continue
        for src in sorted(g[g["condition"] == "benign_forced"]["prompt_source"].unique()):
            bf = g[(g["condition"] == "benign_forced") & (g["prompt_source"] == src)]
            if bf.empty:
                continue
            pre = auroc_ci(hf["H_pre"], bf["H_pre"])
            post = auroc_ci(hf["H_post"], bf["H_post"])
            ok = pre[2] < 0.5 < post[1]
            rows.append((m, f, src, ok))
            print(f"| {m} | {f} | {src} | {len(hf)}/{len(bf)} | "
                  f"{pre[0]:.3f} [{pre[1]:.3f}, {pre[2]:.3f}] | "
                  f"{post[0]:.3f} [{post[1]:.3f}, {post[2]:.3f}] | "
                  f"{'YES' if ok else 'no'} |")

    print("\n--- retained mass across the same axis (the mechanism moving) ---")
    r = (d.groupby(["model", "fmt", "arm"])
           .agg(n=("R", "size"), R_median=("R", "median"),
                R_q1=("R", lambda s: np.nanpercentile(s, 25)),
                R_q3=("R", lambda s: np.nanpercentile(s, 75)),
                bits=("R_bits", "median"))
           .reset_index())
    r["_o"] = r["fmt"].map({k: i for i, k in enumerate(LEVELS)}).fillna(99)
    r = r.sort_values(["model", "_o", "arm"]).drop(columns="_o")
    print("| model | fmt | arm | n | R median [IQR] | -log2 R |")
    print("|" + "---|" * 6)
    for x in r.itertuples():
        print(f"| {x.model} | {x.fmt} | {x.arm} | {x.n} | "
              f"{x.R_median:.3e} [{x.R_q1:.1e}, {x.R_q3:.1e}] | {x.bits:.1f} |")

    print("\nRead this as the boundary condition, not as a second result. The")
    print("claim is stated for the unannounced regime; these rows say what")
    print("happens outside it, including where it stops replicating.")

    print("\n" + "=" * 78)
    print("CLAIM 4: the two supporting numbers (prose, not tables)")
    print("=" * 78)
    print("\n(a) P_refuse on harmful at t0 -- why H_pre is low there:")
    pr = (d[d["condition"] == "harmful_forced"]
          .groupby(["model", "fmt"])["P_refuse"].agg(n="size", mean="mean").round(3))
    print(pr.to_string())

    print("\n(b) unconstrained refusal rate -- the disposition is one the model")
    print("    acts on when it is free to:")
    if "free_refused" in df.columns:
        fr = df[df["condition"] == "free"]
        if not fr.empty:
            u = fr.groupby(["model", "_exp", "prompt_id"], as_index=False)["free_refused"].first()
            print(u.groupby(["model"])["free_refused"]
                   .agg(n="size", refused="sum",
                        rate=lambda s: float(np.nanmean(s.astype(float)))).round(3).to_string())
        else:
            print("    no free-condition rows in this selection -- it lives in")
            print("    results/depth/, so add depth to --only to see it.")
    else:
        print("    free_refused column absent from this selection.")

    print("\n" + "=" * 78)
    n_ok = sum(1 for *_, ok in rows if ok)
    print(f"{n_ok} of {len(rows)} cells inverted across the whole format axis.")
    print("Expect the unannounced levels to pass and the aligned ones not to.")
    print("That pattern IS the scope paragraph.")
    print("=" * 78)


if __name__ == "__main__":
    main()
