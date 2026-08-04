"""Figure 2: per-prompt entropy before and after masking, as a slope plot.

One line per prompt, from H_pre on the left axis to H_post on the right. Harmful
prompts start low and rise, benign prompts start high and stay or fall, so the
two populations cross between the two columns. That crossing IS the paper's
claim, which is why this figure is worth a column: a reader sees in one glance
what the AUROC column of Table 1 states numerically.

No GPU. Reads the same cache the claim scripts use.

    python3 experiments/fig2_slope.py --only fmt2

Writes UncertainNLP-paper/fig2_slope.pdf, which acl_latex.tex picks up
automatically via \\IfFileExists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

from dump_all import load_all                            # noqa: E402

REGIME = ("none", "neutral")
SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Llama-3.1-8B-Instruct": "Llama-3.1-8B"}
# Colour-blind safe, and distinguishable in greyscale by line style as well.
C_HARM, C_BEN = "#B2182B", "#2166AC"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="fmt2")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cache", default=str(ROOT / "results" / "_dump_cache.parquet"))
    ap.add_argument("--out", default=str(ROOT / "UncertainNLP-paper" / "fig2_slope.pdf"))
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    df = load_all(Path(args.cache), args.refresh, only)
    d = df[(df["pos"] == 0)
           & (df["fmt"].isin(REGIME))
           & (df["grammar"] == "forced_steps")
           & (df["condition"].isin(["harmful_forced", "benign_forced"]))
           & (df["prompt_source"].isin(["harmbench", "alpaca"]))].copy()
    d = d.drop_duplicates(subset=["model", "fmt", "grammar", "arm", "prompt_id"])

    cells = [(m, f) for m in ("Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct")
             for f in REGIME]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5), sharey=True)

    for ax, (model, fmt) in zip(axes, cells):
        g = d[(d["model"] == model) & (d["fmt"] == fmt)]
        for cond, colour, style in (("harmful_forced", C_HARM, "-"),
                                    ("benign_forced", C_BEN, "--")):
            sub = g[g["condition"] == cond]
            for _, r in sub.iterrows():
                ax.plot([0, 1], [r["H_pre"], r["H_post"]], color=colour,
                        linestyle=style, alpha=0.28, linewidth=0.7, zorder=1)
            # The mean line carries the message; the per-prompt lines show spread.
            ax.plot([0, 1], [sub["H_pre"].mean(), sub["H_post"].mean()],
                    color=colour, linestyle=style, linewidth=2.4, zorder=3,
                    solid_capstyle="round")
        ax.set_xlim(-0.18, 1.18)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pre", "post"], fontsize=8)
        ax.set_title("%s\n%s" % (SHORT[model], fmt), fontsize=8)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # The ceiling is a property of the schema, not the model, so it is worth
    # drawing: it is what H_post is bounded by and what a reader will otherwise
    # assume is being measured.
    ceiling = float(np.log2(5))
    for ax in axes:
        ax.axhline(ceiling, color="0.6", linewidth=0.7, linestyle=":", zorder=0)
    axes[0].set_ylabel("entropy (bits)", fontsize=8)
    axes[-1].text(1.20, ceiling, r"$\log_2|A_t|$", fontsize=7, color="0.4",
                  va="center")

    h = [plt.Line2D([], [], color=C_HARM, linestyle="-", linewidth=2),
         plt.Line2D([], [], color=C_BEN, linestyle="--", linewidth=2)]
    fig.legend(h, ["harmful", "benign"], loc="lower center", ncol=2,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out = Path(args.out)
    fig.savefig(out, bbox_inches="tight")
    print("wrote %s" % out)
    for model, fmt in cells:
        g = d[(d["model"] == model) & (d["fmt"] == fmt)]
        for cond in ("harmful_forced", "benign_forced"):
            s = g[g["condition"] == cond]
            print("  %-14s %-8s %-15s n=%d  pre %.3f -> post %.3f"
                  % (SHORT[model], fmt, cond, len(s),
                     s["H_pre"].mean(), s["H_post"].mean()))


if __name__ == "__main__":
    main()
