"""Figure 1: the five admitted tokens before and after masking, for one harmful
and one benign prompt.

This is the paper's central picture. On the harmful prompt the admitted tokens
carry near-equal negligible mass, so renormalising them leaves an almost flat
distribution and the served entropy is high. On the benign prompt the model has a
real preference among the SAME five tokens, and renormalisation preserves it.

Needs a GPU, but only two forward passes: one prompt per class, one token each.

    GPU=7 python3 experiments/fig1_distributions.py

Writes UncertainNLP-paper/fig1_distributions.pdf.

RESPONSIBLE USE. Nothing decoded from the harmful prompt is written to disk or
drawn. The figure shows five JSON structural tokens and their probabilities, and
the harmful prompt is identified in the caption only by its benchmark id.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import xgrammar as xgr                                   # noqa: E402

from abstention import runner                            # noqa: E402
from abstention.decode_loop import _bool_allowed         # noqa: E402
from abstention.model_loader import encode_chat          # noqa: E402
from abstention.prompting import build_messages          # noqa: E402

C_HARM, C_BEN = "#B2182B", "#2166AC"


@torch.no_grad()
def first_step(lm, grammar, prompt, schema, fmt):
    """Pre-mask and post-mask probabilities over the admitted set at t=0.

    Uses the SAME build_messages and encode_chat the collection path uses
    (conditions.py:85). A hand-rolled user turn here would apply a different chat
    template and quietly produce a figure that does not match Table 1.
    """
    messages = build_messages(prompt, fmt, schema)
    ids = encode_chat(lm.tokenizer, messages, lm.device)
    logits = lm.model(input_ids=ids, use_cache=False).logits[0, -1, :].float()

    matcher = xgr.GrammarMatcher(grammar)
    bitmask = xgr.allocate_token_bitmask(1, lm.full_vocab)
    matcher.fill_next_token_bitmask(bitmask)
    allowed = _bool_allowed(bitmask, lm.full_vocab, lm.device)

    p_free = torch.softmax(logits, -1)
    idx = torch.nonzero(allowed, as_tuple=True)[0]
    pre = p_free[idx].double().cpu().numpy()
    post = pre / pre.sum()
    toks = [lm.tokenizer.decode([int(i)]) for i in idx]
    H_pre = float(-(p_free.double() * p_free.double().clamp_min(1e-300).log2()).sum())
    H_post = float(-(post * np.log2(np.clip(post, 1e-300, None))).sum())
    return toks, pre, post, pre.sum(), H_pre, H_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen25-7b")
    ap.add_argument("--bench", default="harmbench")
    ap.add_argument("--harmful-index", type=int, default=0)
    ap.add_argument("--benign-index", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "UncertainNLP-paper" / "fig1_distributions.pdf"))
    args = ap.parse_args()

    lm, grammar, schema, exp_cfg, _ = runner.setup(args.model)
    harmful = runner.harmful_prompts(exp_cfg, args.bench)[args.harmful_index]
    benign = runner.benign_prompts(exp_cfg)[args.benign_index]
    fmt = runner.format_instruction(exp_cfg)

    panels = []
    for label, p, colour in (("harmful", harmful, C_HARM), ("benign", benign, C_BEN)):
        toks, pre, post, R, H_pre, H_post = first_step(
            lm, grammar, p["prompt"], schema, fmt)
        panels.append((label, p["id"], toks, pre, post, R, H_pre, H_post, colour))
        print("%-8s id=%-24s R=%.3e  H_pre=%.3f  H_post=%.3f"
              % (label, p["id"], R, H_pre, H_post))
        print("         tokens: %s" % [t.replace("\n", "\\n") for t in toks])
        print("         pre:    %s" % np.array2string(pre, precision=2))
        print("         post:   %s" % np.array2string(post, precision=3))

    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.0), sharex="col")
    for row, (label, pid, toks, pre, post, R, H_pre, H_post, colour) in enumerate(panels):
        x = np.arange(len(toks))
        # ASCII only: a space token rendered as a space is invisible on the axis,
        # and a non-ASCII glyph risks an encoding failure on the run host.
        names = [t.replace("\n", "\\n").replace(" ", "_") or "?" for t in toks]

        # Pre-mask on a log axis: these masses span orders of magnitude and a
        # linear axis would render every bar as zero.
        ax = axes[row][0]
        ax.bar(x, np.clip(pre, 1e-16, None), color=colour, width=0.65)
        ax.set_yscale("log")
        ax.set_ylim(1e-14, 1.0)
        ax.set_ylabel("%s\np(token)" % label, fontsize=7)
        if row == 0:
            ax.set_title("before mask", fontsize=8)

        ax = axes[row][1]
        ax.bar(x, post, color=colour, width=0.65)
        ax.set_ylim(0, 1.0)
        if row == 0:
            ax.set_title("after mask", fontsize=8)
        ax.text(0.97, 0.92, "$H$=%.2f" % H_post, transform=ax.transAxes,
                ha="right", va="top", fontsize=7)

        for ax in axes[row]:
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=6, rotation=0)
            ax.tick_params(labelsize=6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote %s" % out)
    print("Quote the two prompt ids in the caption, not the prompt text.")


if __name__ == "__main__":
    main()
