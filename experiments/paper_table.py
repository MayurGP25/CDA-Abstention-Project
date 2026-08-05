"""Emit Table 1 of the paper as LaTeX, straight from the parquet.

Every number in the paper's main table comes from this script, so the table
cannot drift from the artefacts through a transcription slip. The output is
written to UncertainNLP-paper/table1.tex, which acl_latex.tex \\input's.

    python3 experiments/paper_table.py --only fmt2

Two panels in one table, so there is one caption rather than two. The upper
panel gives the per-arm values, the lower the AUROCs. Retained mass is reported
as -log2 R_t rather than as a raw mass: nobody can compare 3.6e-11 against
1.6e-11 by eye, whereas 34.7 against 35.9 bits is immediate, and in that form the
column is directly the divergence of the served distribution from the model's own.
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
SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Llama-3.1-8B-Instruct": "Llama-3.1-8B"}
LEVEL = {"none": r"\textsc{none}", "neutral": r"\textsc{neutral}"}
BS = "\\" * 2                    # a LaTeX row terminator, two backslashes


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
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r" & & Pre-mask & Post-mask & & $-\log_2$ " + BS,
         r"Level & Arm & entropy & entropy & $\Delta H$ & $R_t$ " + BS,
         r"\midrule"]

    aurocs = []
    for model in ("Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"):
        L.append(r"\multicolumn{6}{l}{\emph{" + SHORT[model] + "}} " + BS)
        for fmt in REGIME:
            g = d[(d["model"] == model) & (d["fmt"] == fmt)]
            hf = g[g["condition"] == "harmful_forced"]
            bf = g[g["condition"] == "benign_forced"]
            if hf.empty or bf.empty:
                continue
            aurocs.append((SHORT[model], fmt,
                           auroc_ci(hf["dH"], bf["dH"]),
                           auroc_ci(hf["H_pre"], bf["H_pre"]),
                           auroc_ci(hf["H_post"], bf["H_post"])))
            for label, sub, bold in ((LEVEL[fmt], hf, True), ("", bf, False)):
                arm = "harmful" if bold else "benign"
                em = (lambda x: r"\textbf{" + x + "}") if bold else (lambda x: x)
                L.append(" & ".join([
                    label, arm,
                    em("%.2f" % sub["H_pre"].mean()),
                    em("%.2f" % sub["H_post"].mean()),
                    em("$%+.2f$" % sub["dH"].mean()),
                    em("%.1f" % float(np.median(sub["R_bits"]))),
                ]) + " " + BS)
        if model == "Qwen2.5-7B-Instruct":
            L.append(r"\midrule")

    # Lower panel. The reversal is stated far more crisply as one column below
    # 0.5 and another above it than as any single shift statistic, which is why
    # the marginal AUROCs are here and not only in the text.
    #
    # The model goes in a header row, not in column one. Putting "Llama-3.1-8B"
    # in column one widened the whole tabular past the text column, because every
    # other entry there is just a level name.
    L += [r"\midrule",
          r"\multicolumn{6}{l}{\emph{AUROC, harmful against benign}} " + BS]
    last = None
    for short, fmt, a_dh, a_pre, a_post in aurocs:
        if short != last:
            L.append(r"\multicolumn{6}{l}{\emph{" + short + "}} " + BS)
            last = short
        L.append(" & ".join([
            LEVEL[fmt], "",
            "%.3f" % a_pre[0], "%.3f" % a_post[0],
            r"\textbf{%.3f}" % a_dh[0], "",
        ]) + " " + BS)

    cap = (r"\caption{Entropy in bits at the first generated token, means over "
           r"$n\!=\!50$ prompts per arm, with $-\log_2 R_t$ as the median. "
           r"$\Delta H = H_{\text{post}} - H_{\text{pre}}$ is the per-prompt "
           r"shift, paired within a prompt. It is a summary rather than a "
           r"physical quantity, since the two entropies live on different "
           r"supports. $-\log_2 R_t$ is the divergence of the served "
           r"distribution from the model's own in bits, and larger means less of "
           r"the model's probability survives the mask. $|A_t| = 5$ throughout, "
           r"so the post-mask entropy cannot exceed $\log_2 5 = 2.32$. In the "
           r"lower panel the pre-mask column lies below $0.5$ and the post-mask "
           r"column above it in every cell, which is the reversal. Intervals are "
           r"given in the text.}")
    L += [r"\bottomrule", r"\end{tabular}", cap, r"\label{tab:main}",
          r"\end{table}"]

    out = Path(args.out)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("\nwrote %s" % out)
    print("\nAUROCs with intervals, for the Results prose:")
    for short, fmt, a_dh, a_pre, a_post in aurocs:
        print("  %-13s %-8s dH %.3f [%.3f, %.3f]  pre %.3f [%.3f, %.3f]  "
              "post %.3f [%.3f, %.3f]"
              % (short, fmt, a_dh[0], a_dh[1], a_dh[2],
                 a_pre[0], a_pre[1], a_pre[2], a_post[0], a_post[1], a_post[2]))


if __name__ == "__main__":
    main()
