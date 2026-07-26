#!/usr/bin/env bash
# End-to-end run on the GPU box. De-risked order: smoke -> E4 gate -> collect -> report -> E5.
set -euo pipefail

MODEL="${1:-llama31-8b}"
BENCH="${2:-advbench}"

echo "== 0. metric unit tests (CPU) =="
pytest -q tests/test_metrics.py

echo "== 1. smoke slice =="
python scripts/smoke_test.py --model "$MODEL"

echo "== 2. E4 confound control (KILL-SWITCH: needs AUROC >> 0.5) =="
python experiments/e4_confound_control.py --model "$MODEL" --bench "$BENCH" --n 40

echo "== 3. collect depth data (all conditions) =="
python experiments/collect.py --model "$MODEL" --bench "$BENCH"

echo "== 4. headline numbers =="
python experiments/report.py "results/depth/${MODEL}__${BENCH}.parquet"

echo "== 5. figures =="
python analysis/figures.py "results/depth/${MODEL}__${BENCH}.parquet" --model "$MODEL"

echo "== 6. restoration (E5) =="
python experiments/e5_restoration.py --model "$MODEL" --bench "$BENCH" --n 50

echo "done."
