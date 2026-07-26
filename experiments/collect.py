"""Main data collection. Runs the three conditions and writes one tidy parquet
that powers E1 (mu_0), E2 (depth), E3 (coercion decomposition), and E4 (control).

  free           : harmful prompts, no grammar
  harmful_forced : harmful prompts, forcing grammar
  benign_forced  : benign  prompts, same forcing grammar (matched control)

Usage:
  python experiments/collect.py --model llama31-8b --bench advbench --n 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abstention import runner  # noqa: E402
from abstention.conditions import run_condition  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model key from configs/models.yaml")
    ap.add_argument("--bench", default="advbench")
    ap.add_argument("--n", type=int, default=None, help="override prompt count")
    ap.add_argument("--exp", default="depth", help="results subdir")
    args = ap.parse_args()

    lm, grammar, _schema, exp_cfg, _ = runner.setup(args.model)

    harmful = runner.harmful_prompts(exp_cfg, args.bench)
    benign = runner.benign_prompts(exp_cfg)
    if args.n:
        harmful, benign = harmful[: args.n], benign[: args.n]

    refusal_ids = runner.get_refusal_ids(lm, exp_cfg, harmful)
    T = exp_cfg["max_new_tokens"]

    rows: list[dict] = []
    for p in tqdm(harmful, desc="harmful"):
        rows += run_condition(lm, condition="free", prompt=p["prompt"],
                              prompt_id=p["id"], refusal_ids=refusal_ids, max_new_tokens=T)
        rows += run_condition(lm, condition="harmful_forced", prompt=p["prompt"],
                              prompt_id=p["id"], refusal_ids=refusal_ids,
                              compiled_grammar=grammar, max_new_tokens=T)
    for p in tqdm(benign, desc="benign"):
        rows += run_condition(lm, condition="benign_forced", prompt=p["prompt"],
                              prompt_id=p["id"], refusal_ids=refusal_ids,
                              compiled_grammar=grammar, max_new_tokens=T)

    out = runner.results_dir(args.exp) / f"{args.model}__{args.bench}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
