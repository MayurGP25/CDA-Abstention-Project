"""Figure 1: the five admitted tokens before and after masking, for one harmful
and one benign prompt.

This is the paper's explanatory picture. On a typical harmful prompt the admitted
tokens carry near-equal negligible mass, so renormalising them leaves an almost
flat distribution and the served entropy is high. On a typical benign prompt the
model has a real preference among the SAME five tokens, and renormalisation
preserves it.

PROMPT SELECTION IS NOT ARBITRARY, and this matters. An earlier version took the
first prompt of each arm, which produced a figure showing benign post-mask
entropy ABOVE harmful, the reverse of the paper's claim and of Table 1. Single
prompts vary. We therefore pick, from the already-collected measurements, the
prompt in each arm whose post-mask entropy is closest to that arm's median, so
the panel illustrates the typical case by construction. The script then checks
the chosen pair reproduces the class ordering and refuses to write a figure that
contradicts the table.

Needs a GPU, but only two forward passes.

    python3 experiments/fig2_slope.py --only fmt2      # populates the cache
    GPU=7 python3 experiments/fig1_distributions.py --only fmt2

Writes UncertainNLP-paper/fig1_distributions.pdf.

RESPONSIBLE USE. Nothing decoded from the harmful prompt is written to disk or
drawn. The figure shows five JSON structural tokens and their probabilities, and
the prompts are identified by benchmark id only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_T0 = time.time()


def step(msg):
    """Progress with a wall clock, flushed.

    Everything slow here is slow because it touches a network filesystem: the
    torch and xgrammar imports, and then fifteen gigabytes of weights out of the
    shared HF cache. Without these lines the script prints nothing for minutes
    and is indistinguishable from a hang, which is exactly how it was first
    reported.
    """
    print("[%6.1fs] %s" % (time.time() - _T0, msg), flush=True)


step("importing torch and numpy")
import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

step("importing matplotlib and xgrammar")
import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import xgrammar as xgr                                   # noqa: E402

step("importing project modules")
from abstention import runner                            # noqa: E402
from abstention.decode_loop import _bool_allowed         # noqa: E402
from abstention.model_loader import encode_chat          # noqa: E402
from abstention.prompting import build_messages          # noqa: E402
from dump_all import load_all                            # noqa: E402
step("imports done")

C_HARM, C_BEN = "#B2182B", "#2166AC"


def token_label(t: str) -> str:
    """Short ASCII label for a structural token.

    The raw decode of these five is a brace plus whitespace variants, so a naive
    escape gives axis labels like `{\\n\\n\\n` that are unreadable at 6pt and
    dominate the panel. NL is compact and the caption spells it out.
    """
    n = t.count("\n")
    body = t.replace("\n", "").replace(" ", "_")
    if n == 0:
        return body or "_"
    return "%s+%dNL" % (body, n) if body else "%dNL" % n


def pick_representative(cache, only, model_key, fmt="none"):
    """prompt_id per arm whose post-mask entropy is nearest that arm's median."""
    df = load_all(Path(cache), False, set(only.split(",")) if only else None)
    d = df[(df["pos"] == 0) & (df["fmt"] == fmt)
           & (df["grammar"] == "forced_steps")
           & (df["model"].str.contains(model_key, case=False))
           & (df["condition"].isin(["harmful_forced", "benign_forced"]))
           & (df["prompt_source"].isin(["harmbench", "alpaca"]))].copy()
    d = d.drop_duplicates(subset=["condition", "prompt_id"])
    out = {}
    for cond in ("harmful_forced", "benign_forced"):
        g = d[d["condition"] == cond]
        if g.empty:
            raise SystemExit("no %s rows for %s/%s in the cache" % (cond, model_key, fmt))
        med = g["H_post"].median()
        row = g.iloc[(g["H_post"] - med).abs().argsort().iloc[0]]
        out[cond] = (str(row["prompt_id"]), float(row["H_pre"]), float(row["H_post"]))
        print("  %-15s median H_post %.3f -> picked %s (H_pre %.3f, H_post %.3f)"
              % (cond, med, row["prompt_id"], row["H_pre"], row["H_post"]))
    return out


@torch.no_grad()
def first_step(lm, grammar, prompt, schema, fmt):
    """Pre-mask and post-mask probabilities over the admitted set at t=0.

    Uses the SAME build_messages and encode_chat as the collection path
    (conditions.py), since a hand-rolled user turn applies a different chat
    template and would silently disagree with Table 1.
    """
    ids = encode_chat(lm.tokenizer, build_messages(prompt, fmt, schema), lm.device)
    logits = lm.model(input_ids=ids, use_cache=False).logits[0, -1, :].float()

    matcher = xgr.GrammarMatcher(grammar)
    bitmask = xgr.allocate_token_bitmask(1, lm.full_vocab)
    matcher.fill_next_token_bitmask(bitmask)
    allowed = _bool_allowed(bitmask, lm.full_vocab, lm.device)

    p_free = torch.softmax(logits, -1).double()
    idx = torch.nonzero(allowed, as_tuple=True)[0]
    pre = p_free[idx].cpu().numpy()
    R = float(pre.sum())
    post = pre / R
    toks = [lm.tokenizer.decode([int(i)]) for i in idx]
    H_post = float(-(post * np.log2(np.clip(post, 1e-300, None))).sum())
    return toks, pre, post, R, H_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen25-7b")
    ap.add_argument("--bench", default="harmbench")
    ap.add_argument("--only", default="fmt2", help="results/ subdirs for selection")
    ap.add_argument("--cache", default=str(ROOT / "results" / "_dump_cache.parquet"))
    ap.add_argument("--harmful-id", default=None, help="override the chosen prompt")
    ap.add_argument("--benign-id", default=None)
    ap.add_argument("--force", action="store_true",
                    help="write the figure even if the pair contradicts Table 1")
    ap.add_argument("--out", default=str(ROOT / "UncertainNLP-paper" / "fig1_distributions.pdf"))
    args = ap.parse_args()

    step("selecting representative prompts from measured data")
    key = "qwen" if "qwen" in args.model.lower() else "llama"
    picks = pick_representative(args.cache, args.only, key)
    want_h = args.harmful_id or picks["harmful_forced"][0]
    want_b = args.benign_id or picks["benign_forced"][0]

    # The long pause. Weights come off the shared cache, so this is minutes on a
    # cold mount and seconds once the page cache is warm.
    step("loading %s (this is the slow part, weights come off the shared cache)"
         % args.model)
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset, torch will pick>")
    step("  CUDA_VISIBLE_DEVICES=%s" % dev)
    lm, grammar, schema, exp_cfg, _ = runner.setup(args.model)
    step("model loaded on %s" % getattr(lm, "device", "?"))
    fmt = runner.format_instruction(exp_cfg)
    harmful = {p["id"]: p for p in runner.harmful_prompts(exp_cfg, args.bench)}
    benign = {p["id"]: p for p in runner.benign_prompts(exp_cfg)}
    for want, pool, name in ((want_h, harmful, "harmful"), (want_b, benign, "benign")):
        if want not in pool:
            raise SystemExit("%s id %r not in the prompt pool" % (name, want))

    panels = []
    for label, p, colour in (("harmful", harmful[want_h], C_HARM),
                             ("benign", benign[want_b], C_BEN)):
        toks, pre, post, R, H_post = first_step(
            lm, grammar, p["prompt"], schema, fmt)
        panels.append((label, p["id"], toks, pre, post, R, H_post, colour))
        print("\n%-8s id=%s" % (label, p["id"]))
        print("  R_t = %.3e   H_post = %.3f" % (R, H_post))
        print("  tokens: %s" % [token_label(t) for t in toks])
        print("  post:   %s" % np.array2string(post, precision=3))

    # A figure that shows benign above harmful would illustrate the reverse of
    # the paper's claim. That happened with arbitrary index selection and is
    # worth failing on rather than shipping.
    h_post, b_post = panels[0][6], panels[1][6]
    if h_post <= b_post and not args.force:
        raise SystemExit(
            "\nREFUSING TO WRITE: harmful H_post %.3f <= benign H_post %.3f, which\n"
            "is the reverse of Table 1. Pick a different pair with --harmful-id /\n"
            "--benign-id, or pass --force if you intend to show an atypical case."
            % (h_post, b_post))

    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.9), sharex="col")
    for row, (label, pid, toks, pre, post, R, H_post, colour) in enumerate(panels):
        x = np.arange(len(toks))
        names = [token_label(t) for t in toks]

        # Log axis before the mask: these masses span orders of magnitude and a
        # linear axis renders every bar as zero.
        ax = axes[row][0]
        ax.bar(x, np.clip(pre, 1e-16, None), color=colour, width=0.62)
        ax.set_yscale("log")
        ax.set_ylim(1e-14, 1.0)
        ax.set_ylabel("%s\np(token)" % label, fontsize=7)
        ax.text(0.03, 0.90, "$R_t$=%.0e" % R, transform=ax.transAxes,
                ha="left", va="top", fontsize=6.5)
        if row == 0:
            ax.set_title("before mask", fontsize=8)

        ax = axes[row][1]
        ax.bar(x, post, color=colour, width=0.62)
        ax.set_ylim(0, 1.0)
        ax.text(0.97, 0.90, "$H$=%.2f" % H_post, transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5)
        if row == 0:
            ax.set_title("after mask", fontsize=8)

        for ax in axes[row]:
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=5.5)
            ax.tick_params(labelsize=6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote %s" % out)
    print("Caption should name the two ids: harmful %s, benign %s" % (want_h, want_b))


if __name__ == "__main__":
    main()
