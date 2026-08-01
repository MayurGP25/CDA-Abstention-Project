"""Everything the paper could report, from every parquet on disk, in one run.

No GPU, no arguments. Reads every shard under results/ and prints the complete
set of numbers so a decision about what goes in the paper can be made from one
screen instead of six scripts.

    python3 experiments/dump_all.py                 # to stdout
    python3 experiments/dump_all.py > dump.txt 2>&1 # to paste somewhere

Each section is independently wrapped: a run directory with an unexpected
schema degrades that one section rather than killing the dump. Legacy parquets
predate the `fmt`, `grammar` and `prompt_source` columns, so those are filled
explicitly -- pd.concat fills missing columns with NaN, and a NaN group key is
DROPPED by groupby, which is how a whole model once vanished from a table with
no error.

Sections:
  0 inventory        what actually exists on disk
  1 precision audit  is any D at the 1e-12 clamp, i.e. censored?
  2 t0 table         the format sweep, per model x level x arm
  3 AUROC            which signals separate harmful from benign, incl. H_post
  4 landmarks        t0 / t_open / t*, from the depth runs
  5 free arm         behavioural refusal rate, unconstrained
  6 Reddy comparison alpha aggregated THEIR way, for a like-for-like number
"""
from __future__ import annotations

import glob
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from analysis.stats import auroc
except Exception:                                        # noqa: BLE001
    def auroc(pos, neg):                                 # Mann-Whitney U fallback
        pos = np.asarray(pos, float)[~np.isnan(np.asarray(pos, float))]
        neg = np.asarray(neg, float)[~np.isnan(np.asarray(neg, float))]
        if not len(pos) or not len(neg):
            return float("nan")
        r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
        return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

CLAMP_EPS = 1e-12                       # metrics.py Z_A.clamp(_EPS, 1.0)
CLAMP_D = -math.log(CLAMP_EPS)          # 27.631 nats == 39.86 bits
LEVELS = ["none", "neutral", "terse", "schema"]
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 400)


def hdr(n: int, title: str) -> None:
    print(f"\n\n{'=' * 78}\n== {n}. {title}\n{'=' * 78}")


def section(n, title, fn, *a, **kw):
    hdr(n, title)
    try:
        fn(*a, **kw)
    except Exception:                                    # noqa: BLE001
        print("!! section failed, continuing:")
        traceback.print_exc(limit=3)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def load_all() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "results" / "**" / "*.parquet"), recursive=True))
    # Prefer shards; a merged parquet duplicates rows already present as shards.
    sh = [f for f in files if f"{'shards'}{'/'}" in f.replace("\\", "/")]
    keep = sh if sh else files
    if not keep:
        raise SystemExit(f"no parquet under {ROOT / 'results'}")
    frames = []
    for f in keep:
        try:
            d = pd.read_parquet(f)
        except Exception as e:                           # noqa: BLE001
            print(f"  [skip] {f}: {e!r}")
            continue
        d["_run"] = Path(f).parent.parent.name if "shards" in f.replace("\\", "/") \
            else Path(f).parent.name
        d["_exp"] = Path(f).relative_to(ROOT / "results").parts[0]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # Legacy fills. NaN group keys are silently dropped by groupby.
    for col, fill in (("fmt", "none"), ("grammar", "forced_steps"),
                      ("prompt_source", "?"), ("condition", "?"), ("model", "?")):
        if col not in df.columns:
            df[col] = fill
        else:
            n_na = int(df[col].isna().sum())
            if n_na:
                print(f"  [legacy] {col}: filled {n_na} NaN with {fill!r}")
            df[col] = df[col].fillna(fill)

    df["model"] = df["model"].astype(str).str.split("/").str[-1]
    df["D"] = df["D"].astype(float)
    df["R"] = np.exp(-df["D"])
    with np.errstate(divide="ignore"):
        df["R_bits"] = df["D"] / math.log(2.0)           # == -log2 R
    df["P_refuse"] = np.exp(df["sr"].astype(float)) if "sr" in df.columns else np.nan
    df["arm"] = np.where(df["condition"] == "benign_forced",
                         "benign/" + df["prompt_source"].astype(str),
                         df["condition"])
    return df


def _iqr(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return (np.nan,) * 3
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


# ---------------------------------------------------------------------------
# 0 inventory
# ---------------------------------------------------------------------------
def s0(df):
    g = (df.groupby(["_exp", "_run", "model", "fmt", "grammar", "arm"])
           .agg(rows=("pos", "size"), prompts=("prompt_id", "nunique"),
                pos_max=("pos", "max"))
           .reset_index())
    print(g.to_string(index=False))
    print(f"\ntotal rows {len(df)}   run dirs {df['_run'].nunique()}   "
          f"models {sorted(df['model'].unique())}")
    print(f"fmt levels present: {sorted(df['fmt'].unique())}")
    cols = [c for c in ("pos", "D", "sr", "H_pre", "H_post", "n_allowed", "mu",
                        "alpha", "s", "t_star", "ctx_open", "free_refused")
            if c in df.columns]
    print(f"columns available: {cols}")
    missing = [c for c in ("t_star", "ctx_open", "free_refused") if c not in df.columns]
    if missing:
        print(f"NOT PRESENT (sections may be empty): {missing}")


# ---------------------------------------------------------------------------
# 1 precision audit -- is the headline number censored by the clamp?
# ---------------------------------------------------------------------------
def s1(df):
    print(f"metrics.py clamps Z_A at {CLAMP_EPS:g}, so D cannot exceed "
          f"{CLAMP_D:.3f} nats = {CLAMP_D / math.log(2):.2f} bits.")
    print("Any row AT that ceiling is censored: its true R is smaller than "
          "reported and the quartile is an artefact.\n")
    at = df["D"] >= CLAMP_D - 1e-6
    print(f"rows total          {len(df)}")
    print(f"rows at ceiling     {int(at.sum())}   <-- must be 0")
    print(f"max D observed      {df['D'].max():.4f} nats "
          f"({df['D'].max() / math.log(2):.2f} bits)")
    print(f"min R observed      {df['R'].min():.3e}")
    if at.any():
        print("\nCENSORED CELLS -- report these as '> {:.1f} bits', not as a number:"
              .format(CLAMP_D / math.log(2)))
        print(df[at].groupby(["model", "fmt", "arm"]).size().to_string())
    t0 = df[df["pos"] == 0]
    print("\nheadroom at t0 (margin in bits between the cell max and the ceiling;"
          "\nunder ~2 bits means the next model or seed could hit the floor):")
    h = (t0.groupby(["model", "fmt", "arm"])["D"]
           .max().to_frame("D_max_nats"))
    h["bits"] = h["D_max_nats"] / math.log(2)
    h["headroom_bits"] = CLAMP_D / math.log(2) - h["bits"]
    print(h.round(2).to_string())


# ---------------------------------------------------------------------------
# 2 t0 table
# ---------------------------------------------------------------------------
def s2(df):
    t0 = df[df["pos"] == 0].copy()
    rows = []
    for (m, f, a), g in t0.groupby(["model", "fmt", "arm"]):
        med, q1, q3 = _iqr(g["R"])
        rows.append(dict(
            model=m, fmt=f, arm=a, n=len(g),
            A=float(g["n_allowed"].median()),
            R_med=med, R_q1=q1, R_q3=q3,
            R_mean=float(np.nanmean(g["R"])),        # for the Reddy comparison
            bits_med=float(np.nanmedian(g["R_bits"])),
            H_pre=float(np.nanmean(g["H_pre"])),
            H_post=float(np.nanmean(g["H_post"])),
            logA=math.log2(max(float(g["n_allowed"].median()), 1)),
            P_ref=float(np.nanmean(g["P_refuse"])),
        ))
    t = pd.DataFrame(rows)
    t["_o"] = t["fmt"].map({k: i for i, k in enumerate(LEVELS)}).fillna(99)
    t = t.sort_values(["model", "_o", "arm"]).drop(columns="_o")

    print("| model | fmt | arm | n | |A| | R median [IQR] | R mean | -log2R | "
          "H_pre | H_post | log2|A| | P_refuse |")
    print("|" + "---|" * 12)
    for r in t.itertuples():
        print(f"| {r.model} | {r.fmt} | {r.arm} | {r.n} | {r.A:.0f} | "
              f"{r.R_med:.3e} [{r.R_q1:.1e}, {r.R_q3:.1e}] | {r.R_mean:.3e} | "
              f"{r.bits_med:.1f} | {r.H_pre:.2f} | {r.H_post:.2f} | "
              f"{r.logA:.2f} | {r.P_ref:.3f} |")
    print("\nH_post must be <= log2|A|. Violations (would mean a bug):")
    bad = t[t["H_post"] > t["logA"] + 1e-6]
    print("  none" if bad.empty else bad.to_string(index=False))


# ---------------------------------------------------------------------------
# 3 AUROC -- including H_post, which has never been computed
# ---------------------------------------------------------------------------
def s3(df):
    t0 = df[df["pos"] == 0]
    print("AUROC = P(harmful ranks ABOVE benign). 0.5 = no separation.")
    print("Far below 0.5 is also separation, in the other direction.")
    print("n=50/50 gives SE ~0.06, so do NOT compare two AUROCs to each other.\n")
    print("| model | fmt | vs | n_h/n_b | R_t | H_pre | H_post | P_refuse |")
    print("|" + "---|" * 8)
    for (m, f), g in t0.groupby(["model", "fmt"]):
        hf = g[g["condition"] == "harmful_forced"]
        if hf.empty:
            continue
        for src in sorted(g[g["condition"] == "benign_forced"]["prompt_source"].unique()):
            bf = g[(g["condition"] == "benign_forced") & (g["prompt_source"] == src)]
            if bf.empty:
                continue
            vals = []
            for col in ("R", "H_pre", "H_post", "P_refuse"):
                vals.append(auroc(hf[col].to_numpy(float), bf[col].to_numpy(float)))
            print(f"| {m} | {f} | {src} | {len(hf)}/{len(bf)} | "
                  + " | ".join(f"{v:.3f}" for v in vals) + " |")


# ---------------------------------------------------------------------------
# 4 landmarks
# ---------------------------------------------------------------------------
def s4(df):
    if "t_star" not in df.columns:
        print("no t_star column -- depth runs absent from results/")
        return
    d = df[df["pos"].notna()].copy()
    parts = []
    p0 = d[d["pos"] == 0].assign(landmark="t0")
    parts.append(p0)
    if "ctx_open" in d.columns:
        op = d[d["ctx_open"].fillna(False).astype(bool)]
        if not op.empty:
            first = op.sort_values("pos").groupby(
                ["_run", "model", "condition", "prompt_id"], as_index=False).first()
            parts.append(first.assign(landmark="t_open"))
    ts = d[(d["t_star"].fillna(-1) >= 0) & (d["pos"] == d["t_star"])]
    if not ts.empty:
        parts.append(ts.assign(landmark="t_star"))
    L = pd.concat(parts, ignore_index=True)

    rows = []
    for (m, f, a, lm), g in L.groupby(["model", "fmt", "arm", "landmark"]):
        med, q1, q3 = _iqr(g["R"])
        rows.append(dict(model=m, fmt=f, arm=a, landmark=lm, n=len(g),
                         pos_med=float(np.nanmedian(g["pos"])),
                         A=float(g["n_allowed"].median()),
                         R_med=med, R_q1=q1, R_q3=q3,
                         H_pre=float(np.nanmean(g["H_pre"])),
                         H_post=float(np.nanmean(g["H_post"])),
                         P_ref=float(np.nanmean(g["P_refuse"]))))
    t = pd.DataFrame(rows)
    order = {"t0": 0, "t_open": 1, "t_star": 2}
    t["_o"] = t["landmark"].map(order)
    t = t.sort_values(["model", "fmt", "arm", "_o"]).drop(columns="_o")
    print("| model | fmt | arm | landmark | n | pos | |A| | R median [IQR] | "
          "H_pre | H_post | P_refuse |")
    print("|" + "---|" * 11)
    for r in t.itertuples():
        print(f"| {r.model} | {r.fmt} | {r.arm} | {r.landmark} | {r.n} | "
              f"{r.pos_med:.0f} | {r.A:.0f} | "
              f"{r.R_med:.3e} [{r.R_q1:.1e}, {r.R_q3:.1e}] | "
              f"{r.H_pre:.2f} | {r.H_post:.2f} | {r.P_ref:.3f} |")
    print("\nt* coverage (generations where the anchor was found):")
    if "t_star" in df.columns:
        cov = (df.groupby(["model", "_exp"])["t_star"]
                 .apply(lambda s: f"{(s >= 0).sum()}/{len(s)} rows"))
        print(cov.to_string())


# ---------------------------------------------------------------------------
# 5 free arm
# ---------------------------------------------------------------------------
def s5(df):
    if "free_refused" not in df.columns:
        print("no free_refused column -- the free arm was not collected here")
        return
    fr = df[df["condition"] == "free"]
    if fr.empty:
        print("free arm present as a column but no rows")
        return
    u = fr.groupby(["model", "_exp", "prompt_id"], as_index=False)["free_refused"].first()
    print(u.groupby(["model", "_exp"])["free_refused"]
           .agg(n="size", refused="sum",
                rate=lambda s: float(np.nanmean(s.astype(float)))).round(3).to_string())
    print("\nThis is the BEHAVIOURAL number: refusal rate with no grammar.")
    print("Contrast with P_refuse at t0 under the grammar in section 2.")


# ---------------------------------------------------------------------------
# 6 Reddy et al. like-for-like
# ---------------------------------------------------------------------------
def s6(df):
    print("Reddy et al. (ICML 2026) Table 10, MATH500, sequence-averaged alpha:")
    print("    Qwen2.5-7B-Instruct   0.8019 (CD)     0.9156 (DCCD)")
    print("    Llama-3.1-8B-Instruct 0.7144 (CD)     0.8822 (DCCD)")
    print("Their KL identity: KL(p_con || p_free) = log(1/alpha) = -log alpha,")
    print("reverse direction because the forward KL is infinite. Our metrics.py")
    print("computes D = -log Z_A, which is the SAME quantity. Only aggregation")
    print("and units differ, and both matter:\n")
    print("  * they average alpha over TOKENS in a sequence, then over prompts;")
    print("    we read one landmark and take the median over prompts.")
    print("  * mean(-log a) != -log(mean a)  [Jensen], so an alpha comparison and")
    print("    a KL comparison are not interchangeable. Quote alpha, not bits,")
    print("    when comparing to their table.\n")
    d = df[df["condition"].isin(["harmful_forced", "benign_forced"])]
    have_depth = d.groupby(["model", "_exp", "arm"])["pos"].max()
    multi = have_depth[have_depth > 0]
    if multi.empty:
        print("No multi-position runs on disk, so a sequence-averaged alpha")
        print("cannot be formed. Compare t0 mean R in section 2 instead, and")
        print("state the position difference explicitly.")
        return
    print("SEQUENCE-AVERAGED alpha, aggregated exactly their way")
    print("(mean over all generated positions within a generation, then mean")
    print(" over generations). This is the only directly comparable number:\n")
    dd = d[d.set_index(["model", "_exp", "arm"]).index.isin(multi.index)]
    per_gen = dd.groupby(["model", "_exp", "arm", "prompt_id"], as_index=False)["R"].mean()
    out = per_gen.groupby(["model", "_exp", "arm"])["R"].agg(
        n="size", alpha_seq_mean="mean", alpha_seq_median="median").round(4)
    print(out.to_string())
    print("\nAnd the same runs at t0 only, for the position contrast:")
    t0 = dd[dd["pos"] == 0].groupby(["model", "_exp", "arm"])["R"].agg(
        n="size", alpha_t0_mean="mean", alpha_t0_median="median")
    print(t0.to_string())


def main():
    print("=" * 78)
    print("== dump_all: every number currently on disk")
    print("=" * 78)
    df = load_all()
    section(0, "INVENTORY", s0, df)
    section(1, "PRECISION AUDIT (is the headline censored?)", s1, df)
    section(2, "t0 TABLE -- the format sweep", s2, df)
    section(3, "AUROC -- which signal separates", s3, df)
    section(4, "LANDMARKS -- t0 / t_open / t*", s4, df)
    section(5, "FREE ARM -- behavioural refusal", s5, df)
    section(6, "REDDY ET AL. LIKE-FOR-LIKE", s6, df)
    print("\n\ndone.")


if __name__ == "__main__":
    main()
