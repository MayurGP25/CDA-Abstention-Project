"""Validate the refusal prefix set -- the one metric-definition gap in the study.

P_refuse is the probability mass on a fixed set of 14 refusal phrases (28 token
sequences with space variants). That set was hand-written and never checked
against the refusals the models actually produce. If it misses common forms, the
metric under-counts and every reported P_refuse is a lower bound with unknown
slack.

This measures the slack. No GPU: the free arm's generated tokens are already in
the parquet as `forced_id`, so the generations can be reconstructed with a
tokenizer alone. (Only the free arm is usable -- in the forced arms the grammar
forbids a refusal from being emitted at all, so there is nothing to count.)

Reports:
  * coverage c  = fraction of genuine refusals whose opener is in the 28-set
  * the misses, so the slack can be attributed to specific phrasings
  * a per-model breakdown, since lexicons differ across model families

Usage:
  python experiments/coverage.py results/depth/*/*.parquet
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abstention.refusal_score import (  # noqa: E402
    PREFIX_SET_VERSION, REFUSAL_PREFIXES, prefix_set_hash)
from abstention.refusal_set import _looks_like_refusal  # noqa: E402


def _starts_with_prefix_set(text: str) -> str | None:
    """Return the matching phrase from the FROZEN 28-sequence set, or None.
    Case-sensitive on the first letter is too brittle across chat templates, so
    we compare case-insensitively on the stripped opener -- the token-level set
    already carries both space variants."""
    t = text.strip().lower()
    for p in REFUSAL_PREFIXES:
        if t.startswith(p.lower()):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="+")
    ap.add_argument("--model-map", default=None,
                    help="optional HF id override if the parquet's model column "
                         "is not a loadable name")
    args = ap.parse_args()

    paths = [Path(p) for pat in args.parquet for p in glob.glob(pat)] or \
            [Path(p) for p in args.parquet]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    free = df[df.condition == "free"]
    if free.empty:
        sys.exit("no free-arm rows: coverage cannot be measured without generations")

    from transformers import AutoTokenizer                      # noqa: PLC0415

    print(f"prefix set {PREFIX_SET_VERSION} (hash {prefix_set_hash()}), "
          f"{len(REFUSAL_PREFIXES)} phrases\n")

    for model, g in free.groupby("model"):
        tok = AutoTokenizer.from_pretrained(args.model_map or model)
        refusals, covered, missed = 0, 0, Counter()
        for pid, gp in g.groupby("prompt_id"):
            ids = gp.sort_values("pos")["forced_id"].astype(int).tolist()
            text = tok.decode(ids, skip_special_tokens=True)
            if not _looks_like_refusal(text):
                continue                       # not a refusal; nothing to cover
            refusals += 1
            hit = _starts_with_prefix_set(text)
            if hit:
                covered += 1
            else:
                missed[" ".join(text.strip().split()[:4])] += 1

        if not refusals:
            print(f"[{model}] no free-arm refusals detected")
            continue
        c = covered / refusals
        print(f"[{model}]")
        print(f"  free-arm refusals            : {refusals}/{g.prompt_id.nunique()}")
        print(f"  caught by the 28-sequence set: {covered}  ->  coverage c = {c:.3f}")
        print(f"  => P_refuse under-counts refusal mass by a factor of at most "
              f"{1/c:.2f}x; every reported value is a LOWER BOUND.")
        if missed:
            print("  missed openers (add to the set only BEFORE a run, never after):")
            for opener, n in missed.most_common(8):
                print(f"      {n:>3}x  {opener!r}")
        print()

    print("Note: only the free arm can be measured. Under the forcing grammar a "
          "refusal cannot be emitted at all, so no forced-arm text exists to "
          "count against. Whether the model's refusal LEXICON shifts inside the "
          "JSON scaffold is therefore not testable from these artifacts -- state "
          "it as a limitation, and note that retained mass R = exp(-D) measures "
          "the same collapse without any lexicon at all.")


if __name__ == "__main__":
    main()
