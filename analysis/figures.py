"""Figures from the tidy per-step parquet files. Reads results/, writes PNGs.

  fig_depth_profile : mu_t vs position, harmful_forced vs benign_forced (centerpiece).
  fig_decision_bar  : mean mu_0 per model (E1).
  fig_coercion_area : mean D_t split into abstention-attributable (alpha*D) and residual (E3).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless server (no display)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _mean_by_pos(df, col):
    g = df.groupby("pos")[col]
    return g.mean(), g.sem()


def fig_depth_profile(parquet, out, *, model=None):
    df = pd.read_parquet(parquet)
    if model:
        df = df[df.model == model]
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, style in [("harmful_forced", "-o"), ("benign_forced", "--s"), ("free", ":^")]:
        sub = df[df.condition == cond]
        if sub.empty:
            continue
        m, e = _mean_by_pos(sub, "mu")
        ax.plot(m.index, m.values, style, label=cond, markersize=3)
        ax.fill_between(m.index, m - e, m + e, alpha=0.15)
    ax.set_xlabel("generation position t")
    ax.set_ylabel(r"latent refusal mass $\mu_t$")
    ax.set_title(f"Depth profile{f' — {model}' if model else ''}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_decision_bar(parquet, out):
    df = pd.read_parquet(parquet)
    d0 = df[df.pos == 0]
    piv = d0.groupby(["model", "condition"])["mu"].mean().unstack("condition")
    ax = piv.plot(kind="bar", figsize=(7, 4))
    ax.set_ylabel(r"$\mu_0$ (decision-point refusal mass)")
    ax.set_title("E1: latent refusal mass at the decision point")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=150)
    plt.close(ax.figure)


def fig_coercion_area(parquet, out, *, model=None):
    df = pd.read_parquet(parquet)
    df = df[df.condition == "harmful_forced"]
    if model:
        df = df[df.model == model]
    df = df.assign(attrib=df.alpha * df.D, resid=(1 - df.alpha) * df.D)
    a = df.groupby("pos")[["attrib", "resid"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.stackplot(a.index, a["attrib"], a["resid"],
                 labels=["abstention-attributable (αD)", "residual ((1-α)D)"], alpha=0.8)
    ax.set_xlabel("generation position t")
    ax.set_ylabel(r"coercion force $D_t$")
    ax.set_title(f"E3: coercion decomposition{f' — {model}' if model else ''}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fig_decision_bar(args.parquet, f"{args.outdir}/e1_decision_bar.png")
    fig_depth_profile(args.parquet, f"{args.outdir}/e2_depth_profile.png", model=args.model)
    fig_coercion_area(args.parquet, f"{args.outdir}/e3_coercion_area.png", model=args.model)
    print(f"figures -> {args.outdir}")
