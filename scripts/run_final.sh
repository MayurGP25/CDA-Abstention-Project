#!/usr/bin/env bash
# Every remaining run for the paper, in one command.
#
#   GPU=7 bash scripts/run_final.sh
#   GPU=3 bash scripts/run_final.sh 2      # start from stage 2
#
# Pick a free device first:
#   nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
#
# Note: once CUDA_VISIBLE_DEVICES is set, PyTorch renumbers your chosen device
# to "cuda:0". An OOM message naming "GPU 0" therefore does NOT mean the export
# was ignored -- check the process ids in the message against nvidia-smi.
#
# Every stage is resumable and idempotent: finished units are skipped, so it is
# always safe to re-run after a crash or a preemption. NOT `set -e`, so one
# failed stage does not abandon the rest.
set -uo pipefail

cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/shared/scratch/0/home/v_mayur_parvatikar/hf_cache}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-7}}"
PY="${PY:-python3}"
N="${N:-50}"
BENCH="${BENCH:-harmbench}"
START="${1:-1}"
TOTAL=3

echo "run_final: gpu=$CUDA_VISIBLE_DEVICES bench=$BENCH n=$N from stage $START"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/  gpu /' || true

stage () {                       # stage <num> <name> <cmd...>
  local num=$1 name=$2; shift 2
  if [ "$num" -lt "$START" ]; then
    echo -e "\n### [$num/$TOTAL] $name -- SKIPPED"; return 0
  fi
  echo -e "\n############################################################"
  echo "### [$num/$TOTAL] $name    $(date -u +%FT%TZ)"
  echo "############################################################"
  local t0=$SECONDS rc=0
  "$@" || rc=$?
  local el=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then
    echo "### [$num/$TOTAL] OK in $((el/60))m$((el%60))s"
  else
    echo "### [$num/$TOTAL] FAILED rc=$rc -- continuing. Retry: GPU=$CUDA_VISIBLE_DEVICES bash $0 $num" >&2
  fi
  return 0
}

# ---------------------------------------------------------------------------
# 1. Llama format sweep. The headline (R_t = 1.00 benign vs 0.28 harmful under
#    an aligned prompt) currently rests on one row of one model. ~22 min.
#    `schema` is deliberately omitted: it OOMs on a 24 GB card and `terse`
#    already answers the question it was there to answer.
# ---------------------------------------------------------------------------
stage 1 "llama31-8b format sweep (none/neutral/terse)" bash -c '
  for F in none neutral terse; do
    echo "--- fmt=$F ---"
    '"$PY"' experiments/collect.py --model llama31-8b --bench "'"$BENCH"'" --n '"$N"' \
      --exp fmt --format-instruction "$F" --max-new-tokens 1 \
      --conditions harmful_forced benign_forced
  done
'

# ---------------------------------------------------------------------------
# 2. Hard negative under the aligned prompt. AUROC 0.985 is against Alpaca,
#    which is the easy control; XSTest prompts are lexically alarming but safe,
#    so this decides whether R_t tracks harm or just vocabulary. ~8 min.
#    Lands in its own directory thanks to the __ben-xstest suffix.
# ---------------------------------------------------------------------------
stage 2 "XSTest under fmt=terse, both models" bash -c '
  for M in qwen25-7b llama31-8b; do
    echo "--- $M ---"
    '"$PY"' experiments/collect.py --model "$M" --bench "'"$BENCH"'" --n '"$N"' \
      --exp fmt --format-instruction terse --max-new-tokens 1 \
      --benign-source xstest --conditions benign_forced
  done
'

# ---------------------------------------------------------------------------
# 3. Tables. No GPU; safe to re-run at any time.
# ---------------------------------------------------------------------------
stage 3 "tables" bash -c '
  '"$PY"' experiments/fmt_table.py --csv results/fmt/t0.csv
  echo
  '"$PY"' experiments/paper.py results/depth/*/*.parquet results/neutral/*/*.parquet \
      results/xstest/*/*.parquet --outdir results/paper 2>/dev/null | head -40 || true
'

echo -e "\n############################################################"
echo "### run_final finished $(date -u +%FT%TZ)"
echo "### t0 table: results/fmt/t0.csv"
echo "############################################################"
