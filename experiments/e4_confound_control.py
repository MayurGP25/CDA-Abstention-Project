"""E4 -- confound control. RUN THIS FIRST: it is the kill-switch.

Question: does the mu/D/s signal reflect refusal *coercion*, or merely "being
constrained"? We compare harmful_forced vs benign_forced (identical grammar,
harmful vs benign prompt) and test whether the model's free-distribution
surprisal s_t at forced tokens separates the two.

  AUROC(s_t) >> 0.5  -> the model registers harmful coercion specifically; the
                       instrument is valid, proceed to depth profiles.
  AUROC(s_t) ~  0.5  -> signals reflect generic constraint; the framing is in
                       trouble. Stop and rethink before investing in E1-E3.

Usage:
  python experiments/e4_confound_control.py --model llama31-8b --n 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abstention import runner  # noqa: E402
from abstention.conditions import run_condition  # noqa: E402
from analysis import stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default="advbench")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    lm, grammar, _schema, exp_cfg, _ = runner.setup(args.model)
    harmful = runner.harmful_prompts(exp_cfg, args.bench)[: args.n]
    benign = runner.benign_prompts(exp_cfg)[: args.n]
    refusal_ids = runner.get_refusal_ids(lm, exp_cfg, harmful)
    T = exp_cfg["max_new_tokens"]

    rows = []
    for p in tqdm(harmful, desc="harmful_forced"):
        rows += run_condition(lm, condition="harmful_forced", prompt=p["prompt"],
                              prompt_id=p["id"], refusal_ids=refusal_ids,
                              compiled_grammar=grammar, max_new_tokens=T)
    for p in tqdm(benign, desc="benign_forced"):
        rows += run_condition(lm, condition="benign_forced", prompt=p["prompt"],
                              prompt_id=p["id"], refusal_ids=refusal_ids,
                              compiled_grammar=grammar, max_new_tokens=T)

    df = pd.DataFrame(rows)
    out = runner.results_dir("e4") / f"{args.model}__{args.bench}.parquet"
    df.to_parquet(out)

    harmful_s = df[df.condition == "harmful_forced"]["s"]
    benign_s = df[df.condition == "benign_forced"]["s"]
    auc_s = stats.auroc(harmful_s, benign_s)
    harmful_mu0 = df[(df.condition == "harmful_forced") & (df.pos == 0)]["mu"]
    benign_mu0 = df[(df.condition == "benign_forced") & (df.pos == 0)]["mu"]

    print("\n===== E4 confound control =====")
    print(f"model            : {lm.model_id}")
    print(f"AUROC(s_t) harmful-vs-benign forced : {auc_s:.3f}   (want >> 0.5)")
    print(f"mean mu_0 harmful_forced : {harmful_mu0.mean():.3f}")
    print(f"mean mu_0 benign_forced  : {benign_mu0.mean():.3f}")
    verdict = ("PROCEED" if auc_s == auc_s and auc_s > 0.65
               else "WEAK — inspect before proceeding")
    print(f"VERDICT: {verdict}")
    print(f"parquet -> {out}")


if __name__ == "__main__":
    main()
