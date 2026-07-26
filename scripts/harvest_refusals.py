"""Harvest and cache the refusal-token set R for a model. Prints coverage.

Usage:
  python scripts/harvest_refusals.py --model llama31-8b --bench advbench
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abstention import runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default="advbench")
    args = ap.parse_args()
    lm, _g, _s, exp_cfg, _ = runner.setup(args.model)
    harmful = runner.harmful_prompts(exp_cfg, args.bench)
    ids = runner.get_refusal_ids(lm, exp_cfg, harmful)
    print(f"cached R ({exp_cfg['refusal_set']}) for {lm.model_id}: |R|={len(ids)}")


if __name__ == "__main__":
    main()
