#!/usr/bin/env bash
# Re-collect the t0 format sweep with D computed in log space.
#
#   GPU=7 bash scripts/rerun_fmt_logspace.sh
#
# WHY. metrics.py summed the allowed probability mass and took a log, with the
# sum clamped at 1e-12. On Qwen at position 0 the true mass is below that, so 73
# rows reported exactly the clamp and several lower quartiles were pinned to a
# value the grammar never produced. D is now logsumexp(logits) -
# logsumexp(logits[allowed]), which is exact and has no floor.
#
# Lands in results/fmt2/ rather than overwriting results/fmt/, so the censored
# numbers stay on disk for the before/after comparison. Read them apart:
#   python3 experiments/dump_all.py --only fmt2,depth,neutral,xstest,restore
#
# Cheap: --max-new-tokens 1 means one forward pass per prompt. ~25 min for all
# of it, including the schema arm that needed the OOM fix.
set -uo pipefail

cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/shared/scratch/0/home/v_mayur_parvatikar/hf_cache}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-7}}"
PY="${PY:-python3}"
N="${N:-50}"
BENCH="${BENCH:-harmbench}"
EXP="${EXP:-fmt2}"

echo "rerun_fmt_logspace: gpu=$CUDA_VISIBLE_DEVICES exp=$EXP n=$N"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/  gpu /' || true

# Guard: a stale .pyc or an un-pulled worktree would silently reproduce the old
# numbers, and the whole point of this run is that the numbers change.
"$PY" - <<'CHECK' || exit 1
import sys, torch
sys.path.insert(0, "src")
from abstention.metrics import compute_metrics
V = 200
lg = torch.zeros(V); lg[1:6] = -41.45; lg[6:] = -1e4
al = torch.zeros(V, dtype=torch.bool); al[1:6] = True
pf = torch.softmax(lg, -1)
m = compute_metrics(pf, pf, al, torch.tensor([0]), 1, logits=lg)
if m.D < 30:
    print(f"STOP: metrics.py is the OLD probability-space version (D={m.D:.3f}, "
          f"expected ~39.8). git pull, and delete src/abstention/__pycache__.")
    raise SystemExit(1)
print(f"  log-space check OK (D={m.D:.3f} nats, past the old 27.631 clamp)")
CHECK

for M in qwen25-7b llama31-8b; do
  for F in none neutral terse schema; do
    echo -e "\n### $M / fmt=$F / alpaca    $(date -u +%FT%TZ)"
    "$PY" experiments/collect.py --model "$M" --bench "$BENCH" --n "$N" \
      --exp "$EXP" --format-instruction "$F" --max-new-tokens 1 \
      --conditions harmful_forced benign_forced || echo "  FAILED, continuing"
  done
  for F in terse schema; do
    echo -e "\n### $M / fmt=$F / xstest    $(date -u +%FT%TZ)"
    "$PY" experiments/collect.py --model "$M" --bench "$BENCH" --n "$N" \
      --exp "$EXP" --format-instruction "$F" --max-new-tokens 1 \
      --benign-source xstest --conditions benign_forced || echo "  FAILED, continuing"
  done
done

echo -e "\n### tables    $(date -u +%FT%TZ)"
"$PY" experiments/dump_all.py --only "$EXP,depth,neutral,xstest,restore" 2>&1 \
  | tee "dump_${EXP}.txt"
echo "### wrote dump_${EXP}.txt"
