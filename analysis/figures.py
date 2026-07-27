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


def fig_signal_depth(parquet, out, *, model=None):
    """THE main figure (Fig 2): the two paper metrics vs token position, for all
    three conditions with 1 SEM. Left = Refusal Preference (S_R), right = pre-mask
    entropy. Folds the condition cut (harmful vs benign) and the depth cut into
    one panel, mirroring the reference paper's per-item layout."""
    df = pd.read_parquet(parquet)
    if model:
        df = df[df.model == model]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    panels = [(axes[0], "sr", r"Refusal Preference  $\log\sum_r P(r)$"),
              (axes[1], "H_pre", "pre-mask entropy (nats)")]
    styles = [("harmful_forced", "-o"), ("benign_forced", "--s"), ("free", ":^")]
    for ax, col, ylab in panels:
        for cond, style in styles:
            sub = df[df.condition == cond]
            if sub.empty:
                continue
            g = sub.groupby("pos")[col]
            m, e = g.mean().dropna(), g.sem()        # sr is NaN off-stride -> dropna
            ax.plot(m.index, m.values, style, label=cond, markersize=3)
            ax.fill_between(m.index, m - e.reindex(m.index), m + e.reindex(m.index), alpha=0.15)
        ax.set_xlabel("token position t")
        ax.set_ylabel(ylab)
        ax.legend()
    fig.suptitle(f"Signal vs depth{f' — {model}' if model else ''}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_entropy_depth(parquet, out, *, model=None):
    """A1 secondary: served (post-mask) vs latent (pre-mask) entropy over depth.
    Served H_post collapses toward 0 at forced positions while latent H_pre
    stays up -- the behaviour/confidence decoupling."""
    df = pd.read_parquet(parquet)
    df = df[df.condition == "harmful_forced"]
    if model:
        df = df[df.model == model]
    fig, ax = plt.subplots(figsize=(6, 4))
    for col, style, lab in [("H_pre", "-o", r"latent $H_{pre}$"),
                            ("H_post", "--s", r"served $H_{post}$")]:
        m, e = _mean_by_pos(df, col)
        ax.plot(m.index, m.values, style, label=lab, markersize=3)
        ax.fill_between(m.index, m - e, m + e, alpha=0.15)
    ax.set_xlabel("generation position t")
    ax.set_ylabel("entropy (nats)")
    ax.set_title(f"Served vs latent entropy{f' — {model}' if model else ''}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_reliability(detect_json, out):
    """A3: reliability diagram from a detect.py summary json (RAW curve)."""
    import json
    blob = json.loads(Path(detect_json).read_text())["reliability"]
    conf, acc = blob["conf"], blob["acc"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], ":", color="grey", label="perfect")
    ax.plot(conf, acc, "-o", markersize=4, label="observed")
    ax.set_xlabel("predicted refusal probability")
    ax.set_ylabel("empirical non-success rate")
    ax.set_title("Reliability (raw signal)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_risk_coverage(parquet, labels, out, *, k=5, signal="sr"):
    """A2: risk--coverage curve for the success detector."""
    from analysis import stats
    df = pd.read_parquet(parquet)
    d = df[(df.condition == "harmful_forced") & (df.pos < k)]
    sig = d.groupby("prompt_id")[signal].mean()
    lab = pd.read_json(labels, lines=True)
    lab = lab[lab.condition == "harmful_forced"].set_index("prompt_id")["success"]
    j = pd.concat([sig, lab], axis=1, join="inner").dropna()
    safe = 1.0 - j["success"].to_numpy(float)     # 1 = safe (attack failed)
    # Release highest-signal (predicted-safe) first; risk = unsafe among released.
    cov, risk, aurc = stats.risk_coverage(j[signal].to_numpy(float), safe)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(cov, risk, "-", label=f"AURC={aurc:.3f}")
    ax.set_xlabel("coverage (fraction released)")
    ax.set_ylabel("risk (unsafe outputs released)")
    ax.set_title("Risk–coverage (selective abstention)")
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
    ap.add_argument("--labels", default=None, help="label jsonl (enables A2/A3 figs)")
    ap.add_argument("--detect-json", default=None, help="detect.py json (reliability fig)")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fig_signal_depth(args.parquet, f"{args.outdir}/fig2_signal_depth.png", model=args.model)
    fig_decision_bar(args.parquet, f"{args.outdir}/e1_decision_bar.png")
    fig_depth_profile(args.parquet, f"{args.outdir}/a1_depth_profile.png", model=args.model)
    fig_entropy_depth(args.parquet, f"{args.outdir}/a1_entropy_depth.png", model=args.model)
    fig_coercion_area(args.parquet, f"{args.outdir}/e3_coercion_area.png", model=args.model)
    if args.labels:
        fig_risk_coverage(args.parquet, args.labels, f"{args.outdir}/a2_risk_coverage.png")
    if args.detect_json:
        fig_reliability(args.detect_json, f"{args.outdir}/a3_reliability.png")
    print(f"figures -> {args.outdir}")
