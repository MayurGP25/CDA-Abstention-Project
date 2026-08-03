"""Everything the paper could report, from every parquet on disk, in one run.

No GPU, no arguments. Reads every shard under results/ and prints the complete
set of numbers so a decision about what goes in the paper can be made from one
screen instead of six scripts.

    python3 experiments/dump_all.py                 # to stdout
    python3 experiments/dump_all.py --refresh       # after collecting new data

Reads the MERGED parquet in each run directory rather than its shards. On a
network filesystem the cost is per file opened, so 150 shard opens per run cost
minutes while the one merged file they were merged into costs milliseconds. The
concatenated frame is then cached to results/_dump_cache.parquet, so every run
after the first is instant. Pass --refresh once new data lands.

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

import argparse
import math
import sys
import time
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
def _run_dirs(res: Path):
    """One entry per run directory: (run, exp, files, used_merged).

    MERGED FIRST, and this matters a great deal on a cluster. results/ is on a
    network mount where the cost is per-file-open, not per-byte: ~100ms of
    latency times a few thousand shards is minutes of apparent hang. Each run
    directory also holds a merged parquet with exactly the same rows, written by
    collect.py at the end of every invocation, so one open replaces 150.

    Shards are the fallback for a directory that has none, which means a run
    that was interrupted before its merge. Recover it with
        python3 experiments/collect.py --model M --bench B --exp E --merge-only
    """
    out = []
    for exp in sorted(p for p in res.iterdir() if p.is_dir()):
        for run in sorted(p for p in exp.iterdir() if p.is_dir()):
            merged = sorted(f for f in run.glob("*.parquet") if not f.name.startswith("."))
            if merged:
                out.append((run.name, exp.name, merged, True))
                continue
            sd = run / "shards"
            if sd.is_dir():
                sh = sorted(f for f in sd.glob("*.parquet") if not f.name.endswith(".tmp"))
                if sh:
                    out.append((run.name, exp.name, sh, False))
    return out


def load_all(cache: Path | None, refresh: bool,
             only: set[str] | None = None, skip: set[str] | None = None) -> pd.DataFrame:
    # One cache FILE PER SELECTION. A single shared cache holds whatever the
    # last run happened to load, so --only either has to be ignored or has to
    # force a rebuild -- and rebuilding costs ~2 minutes of network filesystem
    # latency every time. Keying the filename lets each selection keep its own.
    if cache is not None and (only or skip):
        tag = "+".join(sorted(only or ["all"]))
        if skip:
            tag += "-x-" + "+".join(sorted(skip))
        cache = cache.with_name(f"{cache.stem}__{tag}{cache.suffix}")
    if cache is not None and cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        print(f"[cache] {len(df)} rows from {cache.name} "
              f"-- pass --refresh to rebuild from results/")
        return df

    res = ROOT / "results"
    if not res.is_dir():
        raise SystemExit(f"no results/ under {ROOT}")
    units = _run_dirs(res)
    if only:
        units = [u for u in units if u[1] in only]
    if skip:
        units = [u for u in units if u[1] not in skip]
    if not units:
        raise SystemExit(f"no parquet under {res}")

    frames = []
    t_start = time.time()
    for run, exp, files, merged in units:
        t0 = time.time()
        got = []
        for f in files:
            try:
                got.append(pd.read_parquet(f))
            except Exception as e:                       # noqa: BLE001
                print(f"  [skip] {f.name}: {e!r}")
        if not got:
            continue
        d = pd.concat(got, ignore_index=True) if len(got) > 1 else got[0]
        d["_run"], d["_exp"] = run, exp
        frames.append(d)
        # Progress, because silence on a network mount is indistinguishable
        # from a hang and the first version of this script got Ctrl-C'd.
        print(f"  [{'merged' if merged else 'shards'}] {exp}/{run:<44} "
              f"{len(d):>7} rows  {len(files):>4} file(s)  {time.time() - t0:5.1f}s",
              flush=True)
    df = pd.concat(frames, ignore_index=True)
    print(f"  loaded {len(df)} rows from {len(units)} run dirs in "
          f"{time.time() - t_start:.1f}s")

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
    df["_key"] = df["_exp"] + "/" + df["_run"]      # `_run` alone collides: the
    # same directory name exists under depth/ and fmt/, which silently merged
    # two runs into one group.

    # The depth and neutral runs predate `prompt_source` and used the default
    # benign set, so their '?' is Alpaca. Leaving it as '?' puts the SAME
    # measurement in the table twice under two labels.
    legacy_ben = (df["prompt_source"] == "?") & (df["condition"] == "benign_forced")
    if legacy_ben.any():
        print(f"  [legacy] prompt_source: relabelled {int(legacy_ben.sum())} "
              f"benign rows '?' -> 'alpaca' (pre-dates the column; default source)")
        df.loc[legacy_ben, "prompt_source"] = "alpaca"
    df["D"] = df["D"].astype(float)
    df["R"] = np.exp(-df["D"])
    with np.errstate(divide="ignore"):
        df["R_bits"] = df["D"] / math.log(2.0)           # == -log2 R
    df["P_refuse"] = np.exp(df["sr"].astype(float)) if "sr" in df.columns else np.nan
    df["arm"] = np.where(df["condition"] == "benign_forced",
                         "benign/" + df["prompt_source"].astype(str),
                         df["condition"])
    if cache is not None:
        try:
            df.to_parquet(cache, index=False)
            print(f"[cache] wrote {cache} -- later runs are instant")
        except Exception as e:                           # noqa: BLE001
            print(f"[cache] not written ({e!r}); harmless")
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

    # BY EXPERIMENT, because this decides what has to be recollected and what
    # does not. A run whose censoring is confined to position 0 can source its
    # t0 row from the cheap fmt sweep and keep its later positions; a run
    # censored at a landmark has to be run again.
    if at.any():
        print("\nWHICH RUNS ARE AFFECTED (this decides what to recollect):")
        by = (df[at].groupby(["_exp", "model", "fmt", "grammar", "arm"])
                    .agg(censored=("D", "size"),
                         pos_min=("pos", "min"), pos_max=("pos", "max")))
        tot = (df.groupby(["_exp", "model", "fmt", "grammar", "arm"])
                 .size().to_frame("rows"))
        print(by.join(tot).to_string())
        print("\npos_min > 0 would mean a landmark is censored, not just t0.")

        print("\nCENSORED ROWS BY POSITION (t0 only is the good case):")
        print(df[at].groupby(["_exp", "pos"]).size().to_string())

    # Landmark-level verdict. t0 is cheap to recollect (one forward pass per
    # prompt); the depth runs are not, so what matters is whether anything
    # AFTER position 0 is at the floor.
    print("\nPER-LANDMARK MINIMUM HEADROOM (bits below the ceiling; 0.00 = censored):")
    ceil_bits = CLAMP_D / math.log(2)
    marks = {"t0": df["pos"] == 0}
    if "ctx_open" in df.columns:
        marks["t_open(pos>0)"] = (df["pos"] > 0) & \
            df["ctx_open"].fillna(False).astype(bool)
    if "t_star" in df.columns:
        marks["t_star"] = (df["t_star"].fillna(-1) >= 0) & (df["pos"] == df["t_star"])
    marks["all pos>0"] = df["pos"] > 0
    rows = []
    for name, sel in marks.items():
        g = df[sel]
        if g.empty:
            rows.append(dict(landmark=name, n=0, worst_bits=np.nan,
                             headroom=np.nan, censored=0))
            continue
        worst = float(g["D"].max()) / math.log(2)
        rows.append(dict(landmark=name, n=len(g), worst_bits=round(worst, 2),
                         headroom=round(ceil_bits - worst, 2),
                         censored=int((g["D"] >= CLAMP_D - 1e-6).sum())))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nVERDICT: if `censored` is 0 for every landmark except t0, the depth")
    print("runs keep their later positions and only t0 needs recollecting, which")
    print("the fmt sweep already provides at one forward pass per prompt.")


# ---------------------------------------------------------------------------
# 2 t0 table
# ---------------------------------------------------------------------------
def s2(df):
    t0 = df[df["pos"] == 0].copy()
    rows = []
    for (m, f, gr, a), g in t0.groupby(["model", "fmt", "grammar", "arm"]):
        med, q1, q3 = _iqr(g["R"])
        rows.append(dict(
            model=m, fmt=f, gram=gr, arm=a, n=len(g),
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
    t = t.sort_values(["model", "_o", "gram", "arm"]).drop(columns="_o")

    print("| model | fmt | grammar | arm | n | |A| | R median [IQR] | R mean | "
          "-log2R | H_pre | H_post | log2|A| | P_refuse |")
    print("|" + "---|" * 13)
    for r in t.itertuples():
        print(f"| {r.model} | {r.fmt} | {r.gram} | {r.arm} | {r.n} | {r.A:.0f} | "
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
    print("| model | fmt | grammar | vs | n_h/n_b | R_t | H_pre | H_post | P_refuse |")
    print("|" + "---|" * 9)
    for (m, f, gr), g in t0.groupby(["model", "fmt", "grammar"]):
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
            print(f"| {m} | {f} | {gr} | {src} | {len(hf)}/{len(bf)} | "
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
        op = d[(d["pos"] > 0) & (d["ctx_open"].fillna(False).astype(bool))]
        if not op.empty:
            first = op.sort_values("pos").groupby(
                ["_key", "model", "condition", "prompt_id"], as_index=False).first()
            parts.append(first.assign(landmark="t_open"))
    ts = d[(d["t_star"].fillna(-1) >= 0) & (d["pos"] == d["t_star"])]
    if not ts.empty:
        parts.append(ts.assign(landmark="t_star"))
    L = pd.concat(parts, ignore_index=True)

    rows = []
    for (m, f, gr, a, lm), g in L.groupby(["model", "fmt", "grammar", "arm", "landmark"]):
        med, q1, q3 = _iqr(g["R"])
        rows.append(dict(model=m, fmt=f, gram=gr, arm=a, landmark=lm, n=len(g),
                         pos_med=float(np.nanmedian(g["pos"])),
                         A=float(g["n_allowed"].median()),
                         R_med=med, R_q1=q1, R_q3=q3,
                         H_pre=float(np.nanmean(g["H_pre"])),
                         H_post=float(np.nanmean(g["H_post"])),
                         P_ref=float(np.nanmean(g["P_refuse"]))))
    t = pd.DataFrame(rows)
    order = {"t0": 0, "t_open": 1, "t_star": 2}
    t["_o"] = t["landmark"].map(order)
    t = t.sort_values(["model", "fmt", "gram", "arm", "_o"]).drop(columns="_o")
    print("| model | fmt | grammar | arm | landmark | n | pos | |A| | "
          "R median [IQR] | H_pre | H_post | P_refuse |")
    print("|" + "---|" * 12)
    for r in t.itertuples():
        print(f"| {r.model} | {r.fmt} | {r.gram} | {r.arm} | {r.landmark} | {r.n} | "
              f"{r.pos_med:.0f} | {r.A:.0f} | "
              f"{r.R_med:.3e} [{r.R_q1:.1e}, {r.R_q3:.1e}] | "
              f"{r.H_pre:.2f} | {r.H_post:.2f} | {r.P_ref:.3f} |")
    print("\nt* coverage (generations where the anchor was found):")
    if "t_star" in df.columns:
        gen = df[df["condition"] != "free"].groupby(
            ["model", "_exp", "_run", "condition", "prompt_id"], as_index=False)["t_star"].first()
        cov = (gen.groupby(["model", "_exp", "condition"])["t_star"]
                  .apply(lambda s: f"{(s >= 0).sum()}/{len(s)} generations"))
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
    have_depth = d.groupby(["model", "_exp", "grammar", "arm"])["pos"].max()
    multi = have_depth[have_depth > 0]
    if multi.empty:
        print("No multi-position runs on disk, so a sequence-averaged alpha")
        print("cannot be formed. Compare t0 mean R in section 2 instead, and")
        print("state the position difference explicitly.")
        return
    print("SEQUENCE-AVERAGED alpha, aggregated exactly their way")
    print("(mean over all generated positions within a generation, then mean")
    print(" over generations). This is the only directly comparable number:\n")
    dd = d[d.set_index(["model", "_exp", "grammar", "arm"]).index.isin(multi.index)]
    per_gen = dd.groupby(["model", "_exp", "arm", "prompt_id"], as_index=False)["R"].mean()
    out = per_gen.groupby(["model", "_exp", "grammar", "arm"])["R"].agg(
        n="size", alpha_seq_mean="mean", alpha_seq_median="median").round(4)
    print(out.to_string())
    print("\nAnd the same runs at t0 only, for the position contrast:")
    t0 = dd[dd["pos"] == 0].groupby(["model", "_exp", "grammar", "arm"])["R"].agg(
        n="size", alpha_t0_mean="mean", alpha_t0_median="median")
    print(t0.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild from results/ instead of reading the cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache", default=str(ROOT / "results" / "_dump_cache.parquet"))
    ap.add_argument("--only", default=None,
                    help="comma-separated results/ subdirs to include, e.g. fmt2,depth. "
                         "Two sweeps of the same conditions must never be pooled: "
                         "they would appear as one arm with double n.")
    ap.add_argument("--skip", default=None, help="comma-separated subdirs to exclude")
    args = ap.parse_args()

    print("=" * 78)
    print("== dump_all: every number currently on disk")
    print("=" * 78)
    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else None
    df = load_all(None if args.no_cache else Path(args.cache), args.refresh, only, skip)
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
