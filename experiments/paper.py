"""THE paper analysis. Two metrics, one table, one figure, two sentences.

Everything else in this repo (detect.py, report.py, e4_*, e5_*, label.py) is off
the critical path and produces nothing that appears in the draft.

Metrics
  P_refuse = exp(S_R)  -- probability mass on refusal openers, read from the
                          PRE-mask distribution by teacher forcing. In [0,1].
  H_pre                -- exact full-vocabulary entropy of the pre-mask
                          distribution, in BITS.

Three landmarks per generation, which is what makes the collapse interpretable:
  t0      pos 0. Prompt seen, nothing written. "What did it want to do."
  t_open  first position whose context ends at a JSON string-VALUE opening quote
          (`{ "step1": "`). Syntactically a fresh sentence start, exactly like t0
          -- so a refusal is a grammatical continuation here too.
  t_star  the verb slot after the `Step 1:` literal, where the grammar actually
          forbids refusal openers.

  t0 -> t_open   isolates the JSON FORMAT SHIFT.
  t_open -> t_star isolates SEMANTIC COMMITMENT (having written "Step 1:", the
                   model has promised to produce steps).

Reporting only t0 and t_star confounds those two and the result is
uninterpretable -- which is why t_open exists.

Usage:
  python experiments/paper.py results/depth/*/*.parquet --outdir results/paper
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import stats  # noqa: E402

P_FLOOR = 1e-8          # log-axis floor; P_refuse is exactly 0.0 for many benign rows
ARMS = ["harmful_forced", "benign_forced"]


# --------------------------------------------------------------------------- #
# landmarks
# --------------------------------------------------------------------------- #
def landmarks(g: pd.DataFrame) -> dict:
    """Positions of t0 / t_open / t_star for one (condition, prompt) generation."""
    g = g.sort_values("pos")
    opens = g[(g.pos > 0) & (g.ctx_open.fillna(False).astype(bool))]
    ts = int(g["t_star"].iloc[0]) if "t_star" in g and pd.notna(g["t_star"].iloc[0]) else -1
    return {
        "t0": 0,
        "t_open": int(opens.pos.iloc[0]) if len(opens) else None,
        "t_star": ts if ts >= 0 else None,
    }


def at_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, condition, prompt, landmark) with P_refuse and H_pre."""
    out = []
    for (model, cond, pid), g in df.groupby(["model", "condition", "prompt_id"]):
        g = g.sort_values("pos").set_index("pos")
        for name, pos in landmarks(g.reset_index()).items():
            if pos is None or pos not in g.index:
                continue
            row = g.loc[pos]
            sr = float(row["sr"])
            out.append(dict(
                model=model, condition=cond, prompt_id=pid, landmark=name, pos=pos,
                P_refuse=float(np.exp(sr)) if np.isfinite(sr) else np.nan,
                H_pre=float(row["H_pre"]),
                H_post=float(row["H_post"]),
                n_allowed=int(row["n_allowed"]),
            ))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Table 1
# --------------------------------------------------------------------------- #
def table1(lm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, landmark), g in lm.groupby(["model", "landmark"]):
        h = g[g.condition == "harmful_forced"]
        b = g[g.condition == "benign_forced"]
        if h.empty:
            continue
        rec = dict(model=model, landmark=landmark, n_harmful=len(h), n_benign=len(b),
                   pos_median=float(h["pos"].median()),
                   n_allowed_median=float(h["n_allowed"].median()),
                   H_post_harmful=float(h["H_post"].mean()))
        for metric in ("P_refuse", "H_pre"):
            hv, bv = h[metric].dropna(), b[metric].dropna()
            rec[f"{metric}_harmful"] = float(hv.mean()) if len(hv) else np.nan
            rec[f"{metric}_benign"] = float(bv.mean()) if len(bv) else np.nan
            rec[f"{metric}_absdelta"] = abs(rec[f"{metric}_harmful"] - rec[f"{metric}_benign"])
            rec[f"{metric}_auroc"] = stats.auroc(hv, bv) if len(hv) and len(bv) else np.nan
        rows.append(rec)
    order = {"t0": 0, "t_open": 1, "t_star": 2}
    return (pd.DataFrame(rows)
            .assign(_o=lambda d: d.landmark.map(order))
            .sort_values(["model", "_o"]).drop(columns="_o").reset_index(drop=True))


def _auroc_cell(a: float) -> str:
    """AUROC is always reported as P(harmful ranks above benign). Values BELOW
    0.5 mean the harmful arm ranks LOWER -- which is a real, signed finding for
    H_pre at t0 (harmful prompts are MORE certain there), not a bug. Marking the
    direction inline stops it being written up backwards."""
    if not np.isfinite(a):
        return "n/a"
    return f"{a:.3f}" + (" [INV]" if a < 0.5 else "")


def table1_markdown(t: pd.DataFrame) -> str:
    lines = ["| model | landmark | pos | P_refuse (harm) | P_refuse (benign) | AUROC | "
             "H_pre (harm) | H_pre (benign) | AUROC | n_allowed | H_post |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(
            f"| {r.model} | {r.landmark} | {r.pos_median:.0f} | "
            f"{r.P_refuse_harmful:.4f} | {r.P_refuse_benign:.4f} | "
            f"{_auroc_cell(r.P_refuse_auroc)} | "
            f"{r.H_pre_harmful:.2f} | {r.H_pre_benign:.2f} | "
            f"{_auroc_cell(r.H_pre_auroc)} | "
            f"{r.n_allowed_median:.0f} | {r.H_post_harmful:.2f} |")
    lines.append("")
    lines.append("AUROC = P(harmful ranks above benign). [INV] marks a value below 0.5, "
                 "i.e. the harmful arm ranks *lower* -- a signed result, not an error. "
                 "H_pre at t0 is expected to be [INV]: on harmful prompts the model is "
                 "MORE certain (it knows it wants to refuse).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figure 1
# --------------------------------------------------------------------------- #
def figure1(lm: pd.DataFrame, out: Path, model: str):
    """THE figure: the two metrics at the three landmarks, per-prompt mean +/- 1 SEM.

    Landmarks, not the raw position curve, because P_refuse between landmarks is
    dominated by SYNTACTIC AVAILABILITY, not preference: a refusal phrase cannot
    begin in the middle of `"step1":`, so P_refuse reads ~0 at those positions and
    jumps back up at the next sentence start. Averaging across positions therefore
    measures token-boundary structure. Only t0 / t_open / t_star are mutually
    comparable -- all three are positions where a refusal could grammatically begin.
    """
    g = lm[lm.model == model]
    order = ["t0", "t_open", "t_star"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    colours = {"harmful_forced": "tab:red", "benign_forced": "tab:blue"}
    marks = {"harmful_forced": "o", "benign_forced": "s"}

    for ax, col, ylab, logy in [(axes[0], "P_refuse", "P(refusal opener)", True),
                                (axes[1], "H_pre", "pre-mask entropy (bits)", False)]:
        for cond in ARMS:
            sub = g[g.condition == cond]
            if sub.empty:
                continue
            m = [sub[sub.landmark == k][col].mean() for k in order]
            e = [sub[sub.landmark == k][col].sem() for k in order]
            ax.errorbar(x, m, yerr=e, marker=marks[cond], color=colours[cond],
                        capsize=3, lw=1.6, label=cond)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(["$t_0$\n(pre-constraint)", "$t_{open}$\n(in JSON,\nsentence start)",
                            "$t^*$\n(after 'Step 1:')"], fontsize=8)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
        ax.grid(alpha=.25, axis="y")

    fig.suptitle(f"Refusal preference and latent entropy at the three landmarks — {model}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_appendix_positions(df: pd.DataFrame, lm: pd.DataFrame, out: Path, model: str):
    """Appendix: the raw position curve, with the syntax caveat on the figure so it
    cannot be misread as belief oscillating."""
    d = df[(df.model == model) & (df.condition.isin(ARMS))].copy()
    d["P_refuse"] = np.exp(d["sr"]).clip(lower=P_FLOOR)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [(axes[0], "P_refuse", "P(refusal opener)", True),
              (axes[1], "H_pre", "pre-mask entropy (bits)", False)]
    styles = {"harmful_forced": ("-o", "tab:red"), "benign_forced": ("--s", "tab:blue")}

    for ax, col, ylab, logy in panels:
        for cond in ARMS:
            sub = d[d.condition == cond].dropna(subset=[col])
            if sub.empty:
                continue
            g = sub.groupby("pos")[col]
            m, e = g.mean(), g.sem()
            style, colour = styles[cond]
            ax.plot(m.index, m.values, style, color=colour, markersize=3, label=cond)
            ax.fill_between(m.index, m - e, m + e, color=colour, alpha=0.15)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("token position")
        ax.set_ylabel(ylab)
        ax.legend(loc="best", fontsize=8)

        # landmark rules, from the harmful arm
        h = lm[(lm.model == model) & (lm.condition == "harmful_forced")]
        for name, colour in [("t_open", "grey"), ("t_star", "black")]:
            pos = h[h.landmark == name]["pos"]
            if len(pos):
                ax.axvline(float(pos.median()), ls=":", lw=1, color=colour)
                ax.annotate(name, (float(pos.median()), ax.get_ylim()[1]),
                            fontsize=7, color=colour, ha="center", va="top")

    fig.suptitle(f"Appendix: per-position values — {model}\n"
                 "P(refusal opener) between landmarks reflects SYNTACTIC AVAILABILITY "
                 "(a refusal cannot begin mid-word), not preference.", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="+", help="one or more collected parquets (globs ok)")
    ap.add_argument("--outdir", default="results/paper")
    args = ap.parse_args()

    paths = [Path(p) for pat in args.parquet for p in glob.glob(pat)] or \
            [Path(p) for p in args.parquet]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if "ctx_open" not in df.columns:
        sys.exit("!! parquet predates the t_open instrumentation -- recollect "
                 "(this is the column that separates format shift from semantic commitment).")

    lm = at_landmarks(df)
    lm.to_csv(outdir / "landmarks_per_prompt.csv", index=False)

    t = table1(lm)
    t.to_csv(outdir / "table1.csv", index=False)
    md = table1_markdown(t)
    (outdir / "table1.md").write_text(md, encoding="utf-8")
    print("\n===== Table 1 =====\n" + md)

    for model in sorted(df.model.unique()):
        figure1(lm, outdir / f"fig1_{_slug(model)}.png", model)
        figure_appendix_positions(df, lm, outdir / f"figA_positions_{_slug(model)}.png", model)

    # ---- the two supporting sentences ------------------------------------- #
    print("\n===== supporting sentences =====")
    for model, g in t.groupby("model"):
        star = g[g.landmark == "t_star"]
        if len(star):
            r = star.iloc[0]
            bound = np.log2(max(r.n_allowed_median, 1))
            print(f"[{model}] At the masked decision point the grammar allows "
                  f"{r.n_allowed_median:.0f} tokens, so served entropy is bounded by "
                  f"log2({r.n_allowed_median:.0f}) = {bound:.1f} bits (observed "
                  f"{r.H_post_harmful:.2f}); at forced literal positions the bound is 0 bits.")

    if "free_refused" in df.columns:
        fr = (df[df.condition == "free"]
              .groupby(["model", "prompt_id"])["free_refused"].first())
        for model, s in fr.groupby(level=0):
            print(f"[{model}] Under free decoding the model opens with a refusal on "
                  f"{100 * s.mean():.0f}% of harmful prompts (n={len(s)}).")

    # ---- the collapse decomposition, printed as a decision aid ------------- #
    print("\n===== collapse decomposition (harmful arm) =====")
    for model, g in lm[lm.condition == "harmful_forced"].groupby("model"):
        p = g.groupby("landmark")["P_refuse"].mean()
        t0, topen, tstar = p.get("t0"), p.get("t_open"), p.get("t_star")
        if None in (t0, topen, tstar) or any(pd.isna(x) for x in (t0, topen, tstar)):
            print(f"[{model}] incomplete landmarks -- cannot decompose")
            continue
        fmt_drop = (t0 - topen) / t0 if t0 else np.nan
        sem_drop = (topen - tstar) / t0 if t0 else np.nan
        print(f"[{model}] P_refuse  t0={t0:.4f}  t_open={topen:.4f}  t_star={tstar:.4f}")
        print(f"          format shift accounts for {100*fmt_drop:5.1f}% of the fall, "
              f"semantic commitment for {100*sem_drop:5.1f}%")
        verdict = ("FORMAT SHIFT (entering JSON)" if fmt_drop > 2 * sem_drop else
                   "SEMANTIC COMMITMENT (writing 'Step 1:')" if sem_drop > 2 * fmt_drop else
                   "MIXED -- run the neutral-scaffold arm to separate them")
        print(f"          -> dominant cause: {verdict}")

    print(f"\nwrote table1.{{md,csv}}, landmarks_per_prompt.csv, fig1_*.png -> {outdir}")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")


if __name__ == "__main__":
    main()
