#!/usr/bin/env bash
# All remaining runs for the UncertaiNLP paper, in priority order, on one GPU.
#
#   ./scripts/run_paper.sh          # everything still outstanding
#   ./scripts/run_paper.sh 3        # start from stage 3
#
# Every stage is RESUMABLE and IDEMPOTENT: re-running skips finished units, so it
# is always safe to re-launch after a crash, preemption, or dropped connection.
# Stages run sequentially because they share one GPU.
#
# Launch detached:
#   nohup ./scripts/run_paper.sh > run_paper.log 2>&1 &
#   echo "PID $!"
#   tail -f run_paper.log
#
# (scripts/run_all.sh is the older E4/E5 pipeline and is not used by the paper.)
set -uo pipefail          # NOT -e: one failed stage must not abandon the rest

cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/shared/scratch/0/home/v_mayur_parvatikar/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
PY="${PY:-python3}"
N="${N:-50}"
BENCH="${BENCH:-harmbench}"
START="${1:-1}"
TOTAL=6

stage () {                 # stage <num> <name> <cmd...>
  local num=$1 name=$2; shift 2
  if [ "$num" -lt "$START" ]; then
    echo -e "\n### [$num/$TOTAL] $name -- SKIPPED (starting from stage $START)"
    return 0
  fi
  echo -e "\n############################################################"
  echo "### [$num/$TOTAL] $name"
  echo "### $(date -u +%FT%TZ)   GPU=$CUDA_VISIBLE_DEVICES"
  echo "############################################################"
  local t0=$SECONDS rc=0
  "$@" || rc=$?
  local el=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then
    echo "### [$num/$TOTAL] $name OK in $((el/60))m$((el%60))s"
  else
    echo "### [$num/$TOTAL] $name FAILED rc=$rc after $((el/60))m -- continuing." >&2
    echo "###   retry just this stage: ./scripts/run_paper.sh $num" >&2
  fi
  return 0
}

echo "run_paper: bench=$BENCH n=$N gpu=$CUDA_VISIBLE_DEVICES from stage $START"
echo "started $(date -u +%FT%TZ)"

# ---------------------------------------------------------------------------
# 1. Llama-3.1-8B landmarks. NON-NEGOTIABLE -- one model is not a result.
#    ~3.7h. Gated on HF: needs an accepted licence and `huggingface-cli login`.
# ---------------------------------------------------------------------------
stage 1 "llama31-8b landmarks" \
  $PY experiments/collect.py --model llama31-8b --bench "$BENCH" --n "$N"

# ---------------------------------------------------------------------------
# 2. Restoration probe, Qwen. Answers Steindl et al.'s call for uncertainty-based
#    defences: is abstention still ENFORCEABLE at depth, or only detectable?
#    8 positions x N prompts, short decodes. ~1h.
# ---------------------------------------------------------------------------
stage 2 "restoration probe (qwen25-7b)" \
  $PY experiments/restore.py --model qwen25-7b --bench "$BENCH" --n "$N"

# ---------------------------------------------------------------------------
# 3. Neutral-scaffold control, Qwen. Separates "entering JSON" from "being asked
#    for steps", currently confounded because the key is literally the word
#    "step1". Harmful arm only, 16 tokens. ~50m.
#    --anchor '' : no "Step 1:" exists here, so t_star correctly reports -1 and
#    paper.py omits that landmark instead of inventing one.
# ---------------------------------------------------------------------------
stage 3 "neutral-scaffold control (qwen25-7b)" \
  $PY experiments/collect.py --model qwen25-7b --bench "$BENCH" --n "$N" \
      --exp neutral --grammar neutral_scaffold --anchor '' --max-new-tokens 16 \
      --conditions harmful_forced

# ---------------------------------------------------------------------------
# 4. Hard negative, Qwen. Alpaca reads P_refuse = 0.0000 everywhere, so AUROC
#    1.000 shows the task was easy, not that the signal is strong. XSTest gives
#    safe-but-scary prompts that are not trivially separable. ~1.8h.
# ---------------------------------------------------------------------------
stage 4 "hard negative: xstest (qwen25-7b)" \
  $PY experiments/collect.py --model qwen25-7b --bench "$BENCH" --n "$N" \
      --exp xstest --benign-source xstest --conditions benign_forced

# ---------------------------------------------------------------------------
# 5. Restoration probe, Llama. Only informative if stage 2 showed a clean
#    deadline; harmless to run either way.
# ---------------------------------------------------------------------------
stage 5 "restoration probe (llama31-8b)" \
  $PY experiments/restore.py --model llama31-8b --bench "$BENCH" --n "$N"

# ---------------------------------------------------------------------------
# 6. Analysis. No GPU needed; safe to re-run any time.
# ---------------------------------------------------------------------------
stage 6 "analysis" bash -c '
  shopt -s nullglob
  PQ=(results/depth/*/*.parquet results/neutral/*/*.parquet results/xstest/*/*.parquet)
  if [ ${#PQ[@]} -eq 0 ]; then echo "no parquets yet"; exit 1; fi
  '"$PY"' experiments/paper.py "${PQ[@]}" --outdir results/paper
  for d in results/restore/*/; do
    m=$(basename "$d"); m=${m%%__*}
    '"$PY"' experiments/restore.py --model "$m" --bench "'"$BENCH"'" --summary-only
  done
'

echo -e "\n############################################################"
echo "### run_paper finished $(date -u +%FT%TZ)"
echo "### table + figures -> results/paper/"
echo "############################################################"
