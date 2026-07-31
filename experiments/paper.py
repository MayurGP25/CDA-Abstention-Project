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
# Display-only floor for log panels. The alpaca benign arm reads exactly 0.0, so
# without this the axis spans to 1e-8 and the readable range (1e-2 .. 1) is
# squeezed into a sliver. Points ON the floor mean "at or below 1e-4", which is
# stated in the axis label rather than silently implied.
DISPLAY_FLOOR = 1e-4
ARMS = ["harmful_forced", "benign_forced"]


# --------------------------------------------------------------------------- #
# landmarks
# --------------------------------------------------------------------------- #
def relabel_legacy(df: pd.DataFrame, grammar: str, benign_source: str) -> pd.DataFrame:
    """Fill provenance columns for parquets collected before they existed.

    The first Qwen depth run predates the `grammar` / `prompt_source` columns, so
    its rows load as "?" while later runs say "forced_steps" / "alpaca". Grouping
    by grammar would then file the two models under different keys and Figure 1
    would silently show only one of them -- the failure is invisible because the
    figure still renders.

    Relabelling is announced, never silent: if you ever point this at a legacy
    parquet from a DIFFERENT grammar, the warning is your cue to pass
    --legacy-grammar.
    """
    # NaN, not "?": when one parquet HAS the column and another does not,
    # pd.concat fills the missing rows with NaN. Checking only for "?" silently
    # dropped an entire model from Table 1.
    for col in ("grammar", "prompt_source"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].astype(object).where(df[col].notna(), "?")
    n = int((df["grammar"] == "?").sum())
    if n:
        print(f"!! {n} rows have no `grammar` column (collected before it existed). "
              f"Relabelling as '{grammar}'. Override with --legacy-grammar.")
        df.loc[df["grammar"] == "?", "grammar"] = grammar
    m = int(((df["prompt_source"] == "?") & (df["condition"] == "benign_forced")).sum())
    if m:
        print(f"!! {m} benign rows have no `prompt_source`. Relabelling as "
              f"'{benign_source}'. Override with --legacy-benign.")
        df.loc[(df["prompt_source"] == "?") & (df["condition"] == "benign_forced"),
               "prompt_source"] = benign_source
    df.loc[df["prompt_source"] == "?", "prompt_source"] = "harmbench"
    return df


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
    """One row per (model, grammar, condition, source, prompt, landmark)."""
    keys = ["model", "grammar", "condition", "prompt_source", "prompt_id"]
    for k in ("grammar", "prompt_source"):
        if k not in df.columns:
            df[k] = "?"                      # tolerate parquets from before this column
    out = []
    for (model, gram, cond, src, pid), g in df.groupby(keys):
        g = g.sort_values("pos").set_index("pos")
        for name, pos in landmarks(g.reset_index()).items():
            if pos is None or pos not in g.index:
                continue
            row = g.loc[pos]
            sr = float(row["sr"])
            # Retained / escape mass, free from the stored D column:
            #   D = -log Z_A  where Z_A = sum_{v in allowed} P_free(v)
            # so R = exp(-D) is the fraction of the model's belief the grammar
            # keeps, and E = 1 - R is what it discards. Both are LEXICON-FREE --
            # they need no refusal phrase list at all, so they establish that
            # belief left the allowed set without depending on how refusal is
            # spelled. P_refuse then says *what* left.
            # D is stored in NATS. It is exactly KL(P_mask || P_free): masking
            # renormalises, so the divergence collapses to -log R. Reporting it
            # as a KL puts the quantity in standard UQ vocabulary, while the
            # retained mass says the same thing more legibly.
            D_nats = float(row["D"])
            R_mass = float(np.exp(-D_nats)) if np.isfinite(D_nats) else np.nan
            KL_bits = D_nats / np.log(2.0) if np.isfinite(D_nats) else np.nan
            out.append(dict(
                model=model, grammar=gram, condition=cond, prompt_source=src,
                prompt_id=pid, landmark=name, pos=pos,
                P_refuse=float(np.exp(sr)) if np.isfinite(sr) else np.nan,
                H_pre=float(row["H_pre"]),
                H_post=float(row["H_post"]),
                n_allowed=int(row["n_allowed"]),
                R_mass=R_mass,
                E_mass=(1.0 - R_mass) if np.isfinite(R_mass) else np.nan,
                KL_bits=KL_bits,
            ))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Table 1
# --------------------------------------------------------------------------- #
def table1(lm: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, grammar, landmark, benign source).

    Grouped by GRAMMAR so a forced_steps run and a neutral_scaffold run are never
    pooled, and one row PER BENIGN SOURCE so an easy control (alpaca) and a hard
    one (xstest) are never averaged into a single misleading AUROC.
    """
    rows = []
    for (model, gram, landmark), g in lm.groupby(["model", "grammar", "landmark"]):
        h = g[g.condition == "harmful_forced"]
        if h.empty:
            continue
        bsrcs = sorted(g[g.condition == "benign_forced"]["prompt_source"].unique()) or [None]
        for bsrc in bsrcs:
            b = (g[(g.condition == "benign_forced") & (g.prompt_source == bsrc)]
                 if bsrc else g.iloc[0:0])
            rec = dict(model=model, grammar=gram, landmark=landmark,
                       benign_source=bsrc or "-", n_harmful=len(h), n_benign=len(b),
                       pos_median=float(h["pos"].median()),
                       n_allowed_median=float(h["n_allowed"].median()),
                       H_post_harmful=float(h["H_post"].mean()),
                       R_mass_harmful=float(h["R_mass"].mean()),
                       E_mass_harmful=float(h["E_mass"].mean()),
                       KL_bits_harmful=float(h["KL_bits"].mean()))
            for metric in ("P_refuse", "H_pre"):
                hv, bv = h[metric].dropna(), b[metric].dropna()
                rec[f"{metric}_harmful"] = float(hv.mean()) if len(hv) else np.nan
                rec[f"{metric}_benign"] = float(bv.mean()) if len(bv) else np.nan
                rec[f"{metric}_absdelta"] = abs(rec[f"{metric}_harmful"]
                                                - rec[f"{metric}_benign"])
                if len(hv) and len(bv):
                    a, alo, ahi = stats.auroc_ci(hv, bv)
                    rec[f"{metric}_auroc"] = a
                    rec[f"{metric}_auroc_lo"], rec[f"{metric}_auroc_hi"] = alo, ahi
                else:
                    rec[f"{metric}_auroc"] = np.nan
                    rec[f"{metric}_auroc_lo"] = rec[f"{metric}_auroc_hi"] = np.nan
                # 95% bootstrap CI on the harmful-arm mean itself
                if len(hv):
                    _, lo, hi = stats.bootstrap_ci(hv.to_numpy(float))
                    rec[f"{metric}_harmful_lo"], rec[f"{metric}_harmful_hi"] = lo, hi
            rows.append(rec)
    order = {"t0": 0, "t_open": 1, "t_star": 2}
    return (pd.DataFrame(rows)
            .assign(_o=lambda d: d.landmark.map(order))
            .sort_values(["model", "grammar", "_o", "benign_source"])
            .drop(columns="_o").reset_index(drop=True))


def _auroc_cell(a: float) -> str:
    """AUROC is always reported as P(harmful ranks above benign). Values BELOW
    0.5 mean the harmful arm ranks LOWER -- which is a real, signed finding for
    H_pre at t0 (harmful prompts are MORE certain there), not a bug. Marking the
    direction inline stops it being written up backwards."""
    if not np.isfinite(a):
        return "n/a"
    return f"{a:.3f}" + (" [INV]" if a < 0.5 else "")


def _auroc_ci_cell(r, metric) -> str:
    a = r[f"{metric}_auroc"]
    lo, hi = r.get(f"{metric}_auroc_lo", np.nan), r.get(f"{metric}_auroc_hi", np.nan)
    if not np.isfinite(a):
        return "n/a"
    s = f"{a:.3f}"
    if np.isfinite(lo):
        s += f" [{lo:.2f},{hi:.2f}]"
    return s + (" [INV]" if a < 0.5 else "")


def table1_markdown(t: pd.DataFrame) -> str:
    lines = ["| model | grammar | landmark | pos | benign src | n | P_refuse (harm) | "
             "P_refuse (benign) | AUROC [95% CI] | H_pre (harm) | H_pre (benign) | "
             "AUROC [95% CI] | n_allowed | retained R | KL bits | H_post |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(
            f"| {r.model.split('/')[-1]} | {r.grammar} | {r.landmark} | "
            f"{r.pos_median:.0f} | {r.benign_source} | {r.n_harmful}/{r.n_benign} | "
            f"{r.P_refuse_harmful:.4f} | {r.P_refuse_benign:.4f} | "
            f"{_auroc_ci_cell(r, 'P_refuse')} | "
            f"{r.H_pre_harmful:.2f} | {r.H_pre_benign:.2f} | "
            f"{_auroc_ci_cell(r, 'H_pre')} | "
            f"{r.n_allowed_median:.0f} | "
            f"{('%.2e' % r.R_mass_harmful) if r.R_mass_harmful < 1e-3 else ('%.4f' % r.R_mass_harmful)} | "
            f"{r.KL_bits_harmful:.1f} | {r.H_post_harmful:.2f} |")
    lines.append("")
    lines.append("AUROC = P(harmful ranks above benign). [INV] marks a value below 0.5, "
                 "i.e. the harmful arm ranks *lower* -- a signed result, not an error. "
                 "H_pre at t0 is expected to be [INV]: on harmful prompts the model is "
                 "MORE certain (it knows it wants to refuse).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figure 1
# --------------------------------------------------------------------------- #
def figure1_grid(lm: pd.DataFrame, out: Path, grammar: str = "forced_steps"):
    """THE figure: rows = models, cols = the two metrics, x = the three landmarks.

    One figure for the whole result. Per-model figures waste a page in a 4-page
    paper, and putting both models on shared axes makes the replication visible
    at a glance rather than requiring the reader to flip between panels.
    """
    g = lm[lm.grammar == grammar]
    models = sorted(g.model.unique())
    if not models:
        return
    order = ["t0", "t_open", "t_star"]
    x = np.arange(len(order))

    # One series per (condition, benign source). Pooling an easy control (alpaca)
    # with a hard one (xstest) would draw a benign line that describes neither --
    # the same mistake Table 1 avoids by emitting a row per source.
    series = [("harmful_forced", None, "harmful", "tab:red", "o", "-")]
    for src in sorted(g[g.condition == "benign_forced"]["prompt_source"].unique()):
        colour = "tab:blue" if src.startswith("alpaca") else "tab:purple"
        series.append(("benign_forced", src, f"benign ({src})", colour, "s", "--"))

    fig, axes = plt.subplots(len(models), 2, figsize=(9.5, 3.6 * len(models)),
                             squeeze=False)
    for r, model in enumerate(models):
        gm = g[g.model == model]
        for c, (col, ylab, logy) in enumerate(
                [("P_refuse", f"P(refusal opener)   [floor {DISPLAY_FLOOR:g}]", True),
                 ("H_pre", "pre-mask entropy (bits)", False)]):
            ax = axes[r][c]
            for cond, src, lab, colour, mk, ls in series:
                sub = gm[gm.condition == cond]
                if src is not None:
                    sub = sub[sub.prompt_source == src]
                if sub.empty:
                    continue
                m = [sub[sub.landmark == k][col].mean() for k in order]
                e = [sub[sub.landmark == k][col].sem() for k in order]
                if logy:      # keep an all-zero benign arm from dragging the log
                    m = [max(v, DISPLAY_FLOOR) for v in m]   # axis down to 1e-8
                ax.errorbar(x, m, yerr=e, marker=mk, color=colour, ls=ls,
                            capsize=3, lw=1.6, label=lab)
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(bottom=DISPLAY_FLOOR / 2)
            ax.set_xticks(x)
            ax.set_xticklabels(["$t_0$", "$t_{open}$", "$t^*$"] if r < len(models) - 1
                               else ["$t_0$\n(pre-constraint)",
                                     "$t_{open}$\n(in JSON,\nsentence start)",
                                     "$t^*$\n(after 'Step 1:')"], fontsize=8)
            ax.set_ylabel(ylab, fontsize=9)
            ax.grid(alpha=.25, axis="y")
            if r == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        axes[r][0].annotate(model.split("/")[-1], xy=(-0.22, 0.5),
                            xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_neutral(lm: pd.DataFrame, out: Path):
    """P_refuse at t_open under the two keys. Answers: format, or semantics?"""
    g = lm[(lm.condition == "harmful_forced") & (lm.landmark == "t_open")]
    models = sorted(m for m in g.model.unique()
                    if {"forced_steps", "neutral_scaffold"} <= set(
                        g[g.model == m].grammar.unique()))
    if not models:
        return
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(models), 3.8))
    w, x = 0.34, np.arange(len(models))
    for i, (gram, lab, col) in enumerate([
            ("neutral_scaffold", '"field1" (neutral)', "tab:green"),
            ("forced_steps", '"step1" (commits)', "tab:red")]):
        m = [g[(g.model == mo) & (g.grammar == gram)]["P_refuse"].mean() for mo in models]
        e = [g[(g.model == mo) & (g.grammar == gram)]["P_refuse"].sem() for mo in models]
        ax.bar(x + (i - 0.5) * w, m, w, yerr=e, capsize=3, label=lab, color=col, alpha=.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models], fontsize=8)
    ax.set_ylabel("P(refusal opener) at $t_{open}$")
    ax.set_title("Same position, same JSON depth,\nonly the key's wording differs",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_coercion_cost(df: pd.DataFrame, lm: pd.DataFrame, out: Path):
    """Appendix: s = -log P_free(forced token) at each landmark.

    'How surprised was the model by what it was forced to say.' A second,
    independent line of evidence for the same mechanism, read from the
    constraint's side rather than the model's: if s falls across the landmarks,
    the model has stopped being surprised by the text it is made to produce.

    Uses `s` rather than `D = -log Z_A` because D confounds coercion with how
    many tokens the grammar happens to allow (n_allowed runs 1 -> 90 here),
    whereas s is the surprisal of one specific emitted token.
    """
    if "s" not in df.columns:
        return
    key = ["model", "grammar", "condition", "prompt_id", "pos"]
    m = lm.merge(df[key + ["s"]], on=[c for c in key if c != "pos"] + ["pos"],
                 how="left")
    g = m[(m.condition == "harmful_forced") & (m.grammar == "forced_steps")]
    if g.empty or g["s"].isna().all():
        return
    order = ["t0", "t_open", "t_star"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for model in sorted(g.model.unique()):
        gm = g[g.model == model]
        mu = [gm[gm.landmark == k]["s"].mean() for k in order]
        se = [gm[gm.landmark == k]["s"].sem() for k in order]
        ax.errorbar(x, mu, yerr=se, marker="o", capsize=3, lw=1.6,
                    label=model.split("/")[-1])
    ax.set_xticks(x)
    ax.set_xticklabels(["$t_0$", "$t_{open}$", "$t^*$"])
    ax.set_ylabel("surprisal of the forced token (nats)")
    ax.set_title("Coercion cost: how surprised the model is\nby what it must write",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure1(lm: pd.DataFrame, out: Path, model: str, grammar: str):
    """THE figure: the two metrics at the three landmarks, per-prompt mean +/- 1 SEM.

    Landmarks, not the raw position curve, because P_refuse between landmarks is
    dominated by SYNTACTIC AVAILABILITY, not preference: a refusal phrase cannot
    begin in the middle of `"step1":`, so P_refuse reads ~0 at those positions and
    jumps back up at the next sentence start. Averaging across positions therefore
    measures token-boundary structure. Only t0 / t_open / t_star are mutually
    comparable -- all three are positions where a refusal could grammatically begin.
    """
    g = lm[(lm.model == model) & (lm.grammar == grammar)]
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
        if ax.get_legend_handles_labels()[0]:      # single-arm runs have no benign line
            ax.legend(fontsize=8)
        ax.grid(alpha=.25, axis="y")

    fig.suptitle(f"Refusal preference and latent entropy at the three landmarks\n"
                 f"{model.split('/')[-1]} — grammar: {grammar}", fontsize=11)
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
    ap.add_argument("--legacy-grammar", default="forced_steps",
                    help="grammar name to assign to rows collected before the "
                         "`grammar` column existed")
    ap.add_argument("--legacy-benign", default="alpaca",
                    help="benign source for rows collected before `prompt_source`")
    args = ap.parse_args()

    paths = [Path(p) for pat in args.parquet for p in glob.glob(pat)] or \
            [Path(p) for p in args.parquet]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if "ctx_open" not in df.columns:
        sys.exit("!! parquet predates the t_open instrumentation -- recollect "
                 "(this is the column that separates format shift from semantic commitment).")

    df = relabel_legacy(df, args.legacy_grammar, args.legacy_benign)
    lm = at_landmarks(df)
    lm.to_csv(outdir / "landmarks_per_prompt.csv", index=False)

    t = table1(lm)
    t.to_csv(outdir / "table1.csv", index=False)
    md = table1_markdown(t)
    (outdir / "table1.md").write_text(md, encoding="utf-8")
    print("\n===== Table 1 =====\n" + md)

    # Fig 1: everything on one figure (rows = models). Per-model versions are
    # still written for slides, but the grid is what goes in the paper.
    figure1_grid(lm, outdir / "fig1_grid.png")
    figure_neutral(lm, outdir / "fig3_neutral_scaffold.png")
    figure_coercion_cost(df, lm, outdir / "figA_coercion_cost.png")
    for (model, gram), _ in lm.groupby(["model", "grammar"]):
        if gram == "none":        # the free arm has no grammar; nothing to plot
            continue
        tag = f"{_slug(model)}__{_slug(gram)}"
        figure1(lm, outdir / f"fig1_{tag}.png", model, gram)
    for model in sorted(df.model.unique()):
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

    # ---- retained mass: the lexicon-free headline -------------------------- #
    # R = exp(-D) = the fraction of the model's own probability mass that the
    # grammar keeps. This needs no refusal phrase list, so it establishes the
    # claim independently of how refusal happens to be spelled.
    print("\n===== retained vs escaped probability mass (harmful arm) =====")
    hf = lm[lm.condition == "harmful_forced"]
    for (model, gram), g in hf.groupby(["model", "grammar"]):
        r = g.groupby("landmark")["R_mass"].mean()
        print(f"[{model.split('/')[-1]}/{gram}]")
        for k in ["t0", "t_open", "t_star"]:
            if k in r and np.isfinite(r[k]):
                sub = g[g.landmark == k]["R_mass"].dropna()
                _, lo, hi = stats.bootstrap_ci(sub.to_numpy(float))
                # scientific notation below 1e-3: at t0 the retained mass is the
                # paper's headline number and "0.0000" hides three orders of it
                fmt = "{:.3e}" if r[k] < 1e-3 else "{:.4f}"
                pct = ("{:.4g}".format(100 * r[k]) if r[k] < 1e-3
                       else "{:.2f}".format(100 * r[k]))
                kl = g[g.landmark == k]["KL_bits"].mean()
                print(f"   {k:7s} retained {fmt.format(r[k])} "
                      f"[95% CI {fmt.format(lo)}, {fmt.format(hi)}]"
                      f"   KL(P_mask||P_free) = {kl:.1f} bits"
                      f"  ->  renormalises {pct}% of the model's belief")

    # ---- the collapse decomposition, printed as a decision aid ------------- #
    print("\n===== collapse decomposition (harmful arm) =====")
    for (model, gram), g in hf.groupby(["model", "grammar"]):
        # paired CI on the t0 -> t_star drop (same prompts at both landmarks)
        w = g.pivot_table(index="prompt_id", columns="landmark", values="P_refuse")
        if {"t0", "t_star"} <= set(w.columns):
            d, lo, hi = stats.paired_bootstrap_diff(w["t0"], w["t_star"])
            print(f"[{model.split('/')[-1]}/{gram}] paired drop t0-t*: "
                  f"{d:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]  (n={w[['t0','t_star']].dropna().shape[0]})")
        p = g.groupby("landmark")["P_refuse"].mean()
        t0, topen, tstar = p.get("t0"), p.get("t_open"), p.get("t_star")
        tag = f"{model.split('/')[-1]}/{gram}"
        if t0 is None or pd.isna(t0):
            continue
        if topen is None or pd.isna(topen):
            print(f"[{tag}] t0={t0:.4f}  (no t_open -- cannot decompose)")
            continue
        print(f"[{tag}] P_refuse  t0={t0:.4f}  t_open={topen:.4f}  "
              f"t_star={'n/a' if tstar is None or pd.isna(tstar) else f'{tstar:.4f}'}")
        if tstar is None or pd.isna(tstar):
            print(f"          prefix cost {100*(t0-topen)/t0:5.1f}% of the refusal mass")
            continue
        fmt, sem = (t0 - topen) / t0, (topen - tstar) / t0
        print(f"          prefix accounts for {100*fmt:5.1f}% of the fall, "
              f"commitment for {100*sem:5.1f}%")

    # ---- neutral-scaffold control: is the prefix drop FORMAT or SEMANTICS? -- #
    at_open = hf[hf.landmark == "t_open"]
    for model, g in at_open.groupby("model"):
        grams = set(g.grammar.unique())
        if not {"forced_steps", "neutral_scaffold"} <= grams:
            continue
        s = g[g.grammar == "forced_steps"]["P_refuse"].dropna()
        n = g[g.grammar == "neutral_scaffold"]["P_refuse"].dropna()
        print(f"\n===== neutral-scaffold control ({model.split('/')[-1]}) =====")
        print(f"  P_refuse at t_open:  '\"step1\"' key = {s.mean():.4f} (n={len(s)})   "
              f"'\"field1\"' key = {n.mean():.4f} (n={len(n)})")
        if n.mean() > 1.5 * s.mean():
            print("  -> the drop is driven by the KEY'S SEMANTICS: naming the field "
                  "'step1' already commits the model. Entering JSON alone costs less.")
        elif s.mean() > 1.5 * n.mean():
            print("  -> unexpected: the neutral key suppresses refusal MORE. Investigate.")
        else:
            print("  -> the drop is driven by the FORMAT SHIFT itself; the key's wording "
                  "is not what does it.")

    # ---- landmark dropout audit: is t_open missing at random? -------------- #
    # t_open is located by parsing the realised token sequence, so it is lost
    # whenever the tokenizer merges the value-opening quote with what follows.
    # If that merge correlates with the outcome, the t_open sample is
    # outcome-conditioned and the landmark comparison is selection-biased.
    print("\n===== t_open dropout audit =====")
    for (model, gram, cond), g in lm.groupby(["model", "grammar", "condition"]):
        if gram == "none":
            continue
        have = set(g[g.landmark == "t_open"]["prompt_id"])
        t0 = g[g.landmark == "t0"].set_index("prompt_id")["P_refuse"]
        if t0.empty or not len(have) or len(have) == len(t0):
            continue
        kept, lost = t0[t0.index.isin(have)], t0[~t0.index.isin(have)]
        tag = f"{model.split('/')[-1]}/{gram}/{cond}"
        print(f"  {tag}: t_open found {len(have)}/{len(t0)} | "
              f"t0 P_refuse  kept={kept.mean():.4f} (n={len(kept)})  "
              f"lost={lost.mean():.4f} (n={len(lost)})")
        if len(lost) >= 2 and len(kept) >= 2:
            a = stats.auroc(kept.to_numpy(float), lost.to_numpy(float))
            flag = "  <-- CHECK: dropout looks outcome-related" if abs(a - .5) > .2 else ""
            print(f"      AUROC(kept > lost) = {a:.3f}  (0.5 = missing at random){flag}")

    # ---- robustness: restrict to prompts the model actually refuses freely -- #
    if "free_refused" in df.columns:
        refused = (df[df.condition == "free"]
                   .groupby("prompt_id")["free_refused"].first())
        keep = set(refused[refused.astype("boolean").fillna(False)].index)
        sub = hf[hf.prompt_id.isin(keep)]
        if len(sub) and len(sub) < len(hf):
            print("\n===== robustness: harmful prompts the model refuses when FREE =====")
            print(f"  {len(keep)}/{len(refused)} prompts; the rest are not refused even "
                  "unconstrained, so they can only dilute t0.")
            for (model, gram), g in sub.groupby(["model", "grammar"]):
                p = g.groupby("landmark")["P_refuse"].mean()
                print(f"  [{model.split('/')[-1]}/{gram}] "
                      + "  ".join(f"{k}={p[k]:.4f}" for k in
                                  ["t0", "t_open", "t_star"] if k in p))

    print(f"\nwrote table1.{{md,csv}}, landmarks_per_prompt.csv, fig1_*.png -> {outdir}")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")


if __name__ == "__main__":
    main()
