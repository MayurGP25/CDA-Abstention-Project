#!/usr/bin/env bash
# Format-instruction sweep at t0 only.
#
# Answers the one question the paper cannot currently answer: is retained mass
# at t0 a fact about constrained decoding, or an artefact of the prompt never
# mentioning that JSON was coming? Reddy et al. (2026) run their constrained
# baseline WITH a JSON system prompt, so without this sweep our 1e-11 is not
# comparable to their alpha and a reviewer can read it as a mismatch result.
#
# Cheap by construction: --max-new-tokens 1 means one forward pass per prompt
# plus the teacher-forced refusal score. No landmark parsing, no depth. The
# `free` arm is skipped -- at t0 no grammar has acted, so its P_free is
# identical to harmful_forced's within a format level (see conditions.py).
#
# `none` is re-run rather than reused. It must reproduce the t0 cells of the
# existing depth runs exactly; if it does not, the new prompting path changed
# something it should not have, and that is worth 3 minutes to find out.
#
#   bash scripts/run_fmt_sweep.sh qwen25-7b harmbench
#   bash scripts/run_fmt_sweep.sh llama31-8b harmbench
set -euo pipefail

MODEL="${1:?usage: run_fmt_sweep.sh <model-key> [bench]}"
BENCH="${2:-harmbench}"
N="${N:-50}"
# The GPU box has python3 only, no `python` on PATH. Override with PY=... if
# your interpreter is named something else.
PY="${PY:-python3}"

# Guard first: never start new collection if the finished runs' digests moved.
$PY scripts/check_fingerprint_compat.py

for LEVEL in none neutral terse schema; do
  echo "=== ${MODEL} / ${BENCH} / fmt=${LEVEL} ==="
  $PY experiments/collect.py \
    --model "$MODEL" --bench "$BENCH" --n "$N" \
    --exp fmt \
    --format-instruction "$LEVEL" \
    --max-new-tokens 1 \
    --conditions harmful_forced benign_forced
done

echo "done. parquets under results/fmt/${MODEL}__${BENCH}__fmt-*/"
