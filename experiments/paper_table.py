"""Emit Table 1 of the paper as LaTeX, straight from the parquet.

Every number in the paper's main table comes from this script, so the table
cannot drift from the artefacts through a transcription slip. The output is
written to UncertainNLP-paper/table1.tex, which acl_latex.tex \\input's.

    python3 experiments/paper_table.py --only fmt2

Reports mean and standard deviation for the entropies and the paired shift,
median for retained mass (it spans orders of magnitude, and its mean runs eight
to nine times higher), and one AUROC per cell with a bootstrap interval.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from claim1_inversion import auroc_ci  # noqa: E402
from dump_all import load_all  # noqa: E402

REGIME = ("none", "neutral")
SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B",
         "Llama-3.1-8B-Instruct": "Llama-3.1-8B"}
LEVEL = {"none": r"\textsc{none}", "neutral": r"\textsc{neutral}"}


def sci(x: float) -> str:
    """LaTeX scientific notation. A plain round() renders 1e-11 as 0.00, which
    would silently delete the column this table exists to show."""
    if not np.isfinite(x) or x == 0:
        return "--"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return "$%.1f{\\times}10^{%d}$" % (m, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="fmt2")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cache", default=str(ROOT / "results" / "_dump_cache.parquet"))
    ap.add_argument("--out", default=str(ROOT / "UncertainNLP-paper" / "table1.tex"))
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    df = load_all(Path(args.cache), args.refresh, only)
    d = df[(df["pos"] == 0)
           & (df["fmt"].isin(REGIME))
           & (df["grammar"] == "forced_steps")
           & (df["condition"].isin(["harmful_forced", "benign_forced"]))
           & (df["prompt_source"].isin(["harmbench", "alpaca"]))].copy()
    d = d.drop_duplicates(subset=["model", "fmt", "grammar", "arm", "prompt_id"])
    d["dH"] = d["H_post"] - d["H_pre"]

    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Level & Arm & $H_{\text{pre}}$ & $H_{\text{post}}$ & $\Delta H$ & $R_t$ \\",
         r"\midrule"]

    aurocs = []
    for model in ("Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"):
        L.append(r"\multicolumn{6}{l}{\emph{%s}} \\" % SHORT[model])
        for fmt in REGIME:
            g = d[(d["model"] == model) & (d["fmt"] == fmt)]
            hf = g[g["condition"] == "harmful_forced"]
            bf = g[g["condition"] == "benign_forced"]
            if hf.empty or bf.empty:
                continue
            a = auroc_ci(hf["dH"], bf["dH"])
            aurocs.append((SHORT[model], fmt, a))
            for label, sub, bold in ((LEVEL[fmt], hf, True), ("", bf, False)):
                arm = "harmful" if bold else "benign"
                f = (lambda s: r"\textbf{%s}" % s) if bold else (lambda s: s)
                L.append("%s & %s & %s & %s & %s & %s \\\\" % (
                    label, arm,
                    f("%.2f\\,\\tiny{$\\pm$%.2f}" % (sub["H_pre"].mean(), sub["H_pre"].std())),
                    f("%.2f\\,\\tiny{$\\pm$%.2f}" % (sub["H_post"].mean(), sub["H_post"].std())),
                    f("%+.2f" % sub["dH"].mean()),
                    sci(float(np.median(sub["R"])))))
        if model == "Qwen2.5-7B-Instruct":
            L.append(r"\midrule")

    cap = ("AUROC of $\\Delta H$ separating the two arms is "
           + ", ".join("%.3f $[%.2f, %.2f]$" % (a[0], a[1], a[2]) for _, _, a in aurocs)
           + " in row order.")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Entropy in bits at the first generated token, mean and "
          r"standard deviation over $n=50$ prompts per arm. $\Delta H$ is the "
          r"per-prompt shift $H_{\text{post}} - H_{\text{pre}}$. $R_t$ is the "
          r"median retained mass. $|A_t| = 5$ throughout, so "
          r"$H_{\text{post}} \le \log_2 5 = 2.32$. " + cap +
          r" Note that $R_t$ barely differs between arms while the two "
          r"entropies order them oppositely.}",
          r"\label{tab:main}", r"\end{table}"]

    out = Path(args.out)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("\nwrote %s" % out)
    print("\nAUROC(dH) per cell, for the Results prose:")
    for m, f, a in aurocs:
        print("  %-14s %-8s %.3f [%.3f, %.3f]" % (m, f, a[0], a[1], a[2]))


if __name__ == "__main__":
    main()
