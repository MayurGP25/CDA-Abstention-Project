"""Figure 1: the five admitted tokens before and after masking, averaged over
each arm.

AVERAGED, not a single pair, and the reason matters. Two earlier versions plotted
one harmful and one benign prompt. The first picked index 0 and produced a figure
whose benign post-mask entropy sat ABOVE the harmful one, the reverse of Table 1.
The second picked the prompt nearest each arm's median, which fixed the direction
but still showed two nearly identical panels, because a 0.1 bit gap between two
single prompts is invisible.

The measured within-arm rank correlation between the two entropies is
approximately zero, so an individual prompt carries almost no signal about the
effect: it is a between-arm effect and only an average can show it. This script
therefore runs every prompt in both arms and plots the mean post-mask probability
per admitted token with its interquartile range across prompts.

One forward pass per prompt, so a hundred in total at the default settings.

    GPU=7 python3 experiments/fig1_distributions.py

Writes UncertainNLP-paper/fig1_distributions.pdf.

RESPONSIBLE USE. Nothing decoded from the harmful prompts is written to disk or
drawn. The figure shows five JSON structural tokens and their probabilities.
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
step("imports done")

C_HARM, C_BEN = "#B2182B", "#2166AC"


def token_label(i: int, t: str) -> str:
    """Axis label for the i-th admitted token.

    Earlier versions printed the raw decode, giving unreadable runs of escaped
    newlines, then a "+kNL" shorthand that still needed explaining. Positional
    labels keep the axis clean and the caption carries the identities, which is
    the usual convention and needs no notation at all.
    """
    return "T%d" % (i + 1)


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
    ap.add_argument("--n", type=int, default=50, help="prompts per arm to average")
    ap.add_argument("--out", default=str(ROOT / "UncertainNLP-paper" / "fig1_distributions.pdf"))
    args = ap.parse_args()

    step("loading %s (weights come off the shared cache, so this is the slow part)"
         % args.model)
    step("  CUDA_VISIBLE_DEVICES=%s"
         % os.environ.get("CUDA_VISIBLE_DEVICES", "<unset, torch will pick>"))
    lm, grammar, schema, exp_cfg, _ = runner.setup(args.model)
    step("model loaded")
    fmt = runner.format_instruction(exp_cfg)
    arms = (("harmful", runner.harmful_prompts(exp_cfg, args.bench)[:args.n], C_HARM),
            ("benign", runner.benign_prompts(exp_cfg)[:args.n], C_BEN))

    panels = []
    for label, prompts, colour in arms:
        pres, posts, Hs, toks = [], [], [], None
        for i, p in enumerate(prompts):
            t, pre, post, R, H = first_step(lm, grammar, p["prompt"], schema, fmt)
            toks = toks or t
            pres.append(pre); posts.append(post); Hs.append(H)
            if (i + 1) % 10 == 0:
                step("  %s %d/%d" % (label, i + 1, len(prompts)))
        pres, posts = np.array(pres), np.array(posts)
        panels.append((label, toks, pres, posts, np.array(Hs), colour))
        step("%s done: mean H_post %.3f over n=%d" % (label, np.mean(Hs), len(Hs)))

    h_mean, b_mean = panels[0][4].mean(), panels[1][4].mean()
    if h_mean <= b_mean:
        raise SystemExit("REFUSING TO WRITE: harmful mean H_post %.3f <= benign %.3f, "
                         "which contradicts Table 1." % (h_mean, b_mean))

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
    x = np.arange(len(panels[0][1]))
    w = 0.38
    for k, (label, toks, pres, posts, Hs, colour) in enumerate(panels):
        off = (k - 0.5) * w
        # Mean post-mask probability per admitted token, with the interquartile
        # range across prompts. A single prompt cannot show a between-arm effect,
        # which the near-zero within-arm rank correlation makes explicit, so the
        # figure has to average.
        m = posts.mean(0)
        lo = m - np.percentile(posts, 25, axis=0)
        hi = np.percentile(posts, 75, axis=0) - m
        axes[1].bar(x + off, m, width=w, color=colour, label=label,
                    yerr=np.vstack([np.clip(lo, 0, None), np.clip(hi, 0, None)]),
                    error_kw=dict(lw=0.6, capsize=1.5, ecolor="0.35"))
        axes[0].bar(x + off, np.clip(pres.mean(0), 1e-16, None), width=w,
                    color=colour, label=label)

    axes[0].set_yscale("log"); axes[0].set_ylim(1e-14, 1.0)
    axes[0].set_title("before mask", fontsize=8)
    axes[0].set_ylabel("p(token)", fontsize=7)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("after mask", fontsize=8)
    axes[1].text(0.97, 0.95, "$H$: %.2f vs %.2f" % (h_mean, b_mean),
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=6.5)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["T%d" % (i + 1) for i in x], fontsize=6.5)
        ax.tick_params(labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[1].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()

    out = Path(args.out)
    fig.savefig(out, bbox_inches="tight")
    step("wrote %s" % out)
    for label, toks, pres, posts, Hs, colour in panels:
        print("  %-8s mean post-mask p: %s" % (label, np.array2string(posts.mean(0), precision=3)))


if __name__ == "__main__":
    main()
