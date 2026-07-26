"""E5 -- restoration (constructive, safety-positive).

Re-allow R u {EOS} into the valid set at position p during the attacked
(harmful_forced) run and measure the fraction of prompts where the model exits
into a refusal. Sweeping p shows how deep the recoverable abstention persists.
Recovery is bounded by the free-decoding refusal mass at p.

Usage:
  python experiments/e5_restoration.py --model llama31-8b --n 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abstention import runner  # noqa: E402
from abstention.decode_loop import restoration_probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default="advbench")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    lm, grammar, _schema, exp_cfg, _ = runner.setup(args.model)
    harmful = runner.harmful_prompts(exp_cfg, args.bench)[: args.n]
    refusal_ids = runner.get_refusal_ids(lm, exp_cfg, harmful)
    positions = exp_cfg["restoration_positions"]
    T = exp_cfg["max_new_tokens"]

    rows = []
    for p in tqdm(harmful, desc="restoration"):
        msgs = [{"role": "user", "content": p["prompt"]}]
        for rpos in positions:
            r = restoration_probe(lm, msgs, refusal_ids, grammar,
                                  restore_pos=rpos, max_new_tokens=T)
            r.update(model=lm.model_id, prompt_id=p["id"])
            rows.append(r)

    df = pd.DataFrame(rows)
    out = runner.results_dir("e5") / f"{args.model}__{args.bench}.parquet"
    df.to_parquet(out)

    print("\n===== E5 restoration =====")
    summary = df.groupby("restore_pos").agg(
        escape_rate=("escaped", "mean"),
        mean_recovered=("recovered_mass", "mean"),
        mean_mu=("mu_at_pos", "mean"),
    )
    print(summary.to_string())
    print(f"parquet -> {out}")


if __name__ == "__main__":
    main()
