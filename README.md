# Masked but Not Silent — abstention under constrained decoding

Logit-level measurement of whether a model's **refusal/abstention intent survives
constrained decoding**. We reuse grammar-constrained jailbreaks (EnumAttack-style
forcing) purely as an *instrument*: mask the refusal channel, then read the
**pre-mask** next-token distribution to ask whether abstention is *latent*
(present in the logits but masked) or *surface* (gone once its opening tokens are
unavailable).

Training-free, dataset-free. Targets a 4-page UncertaiNLP @ EMNLP short paper.

## Why this needs a GPU + open weights (not an API)

The whole method reads the model's logits **before** the grammar mask is applied.
Hosted APIs (OpenAI/Gemini/Groq) never expose that, and vLLM's sampler fuses
masking into sampling and hides the seam. So we run **open-weight models locally**
with a **HuggingFace manual decode loop + xgrammar**, which exposes the seam
(`fill_next_token_bitmask` → read → `apply_token_bitmask_inplace`).

Models: Llama-3.1-8B-Instruct (primary), Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3.
A single 24 GB GPU is enough; the full matrix is a few GPU-hours per model.

## Setup (Vast / Linux + CUDA)

```bash
git clone <this-repo> && cd abstention-under-constraint
python -m venv .venv && source .venv/bin/activate

# 1) CUDA-matched torch FIRST (pick the wheel for your driver):
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 2) the rest:
pip install -r requirements.txt
pip install -e .

# 3) gated Llama: accept the license on HF, then:
huggingface-cli login
```

## Run order (de-risked — two cheap kill-switches first)

```bash
# 0. metric math (CPU, runs anywhere, ~1s) — guards the corrected KL etc.
pytest -q tests/test_metrics.py

# 1. day-1 vertical slice: one prompt, both conditions, four metrics
python scripts/smoke_test.py --model llama31-8b

# 2. KILL-SWITCH E4: is the signal refusal-coercion or just "constrained"?
#    Needs AUROC(s_t) >> 0.5 to proceed.
python experiments/e4_confound_control.py --model llama31-8b --n 40

# 3. collect the full depth data (free / harmful_forced / benign_forced)
python experiments/collect.py --model llama31-8b --bench advbench

# 4. headline numbers (E1 mu_0, E2 depth t*, E3 alpha, E4 AUROC)
python experiments/report.py results/depth/llama31-8b__advbench.parquet

# 5. figures (centerpiece depth profile, decision bar, coercion area)
python analysis/figures.py results/depth/llama31-8b__advbench.parquet --model llama31-8b

# 6. E5 restoration sweep (safety-positive result)
python experiments/e5_restoration.py --model llama31-8b --n 50
```

Or the whole thing: `bash scripts/run_all.sh llama31-8b advbench`.
Repeat steps 3–6 for `qwen25-7b` and `mistral-7b`.

## What each metric is (paper §3, KL direction corrected)

| Metric | Meaning | Note |
|---|---|---|
| `mu`  (M1) | `Σ_{v∈R} P_free(v)` — latent refusal mass | read from pre-mask logits; the central quantity |
| `D`   (M2) | `KL(P_con ‖ P_free)` over the allowed support | **reverse** KL — finite; forward diverges on masked tokens |
| `alpha` (M3) | share of masked-away mass that was refusal | in `[0,1]` |
| `s`   (M4) | `-log P_free(forced token)` — coercion surprisal | drives the E4 AUROC |

## Design choices that matter (and why)

- **Baseline is `benign_forced`, not `free`.** At `t=0` the pre-mask logits don't
  depend on the grammar, so `μ₀` is identical in free and forced conditions —
  E1 is a *magnitude* result, not a free-vs-forced test (`report.py` prints this
  as a sanity check). For `t>0`, free and forced condition on divergent prefixes,
  so the clean matched comparison is harmful-forced vs benign-forced (identical
  grammar, harmful vs benign prompt). That is why E4 is the kill-switch.
- **`multi_step` grammar** re-forces an affirmative opener at each step, giving
  several masked decision points at increasing depth — a better depth probe than
  one prefix + free tail.
- **vocab-size fix** (`config.vocab_size`, not `tokenizer.vocab_size`) is applied
  once in `model_loader.py`; getting it wrong silently corrupts every metric.

## Layout

```
src/abstention/     metrics · model_loader · grammars · refusal_set · decode_loop · conditions · data · runner
experiments/        collect · e4_confound_control · e5_restoration · report
analysis/           stats · figures
tests/              test_metrics (CPU) · test_grammar
scripts/            smoke_test · harvest_refusals · run_all.sh
configs/            models · experiment · grammars
```

## Pivot plan

`μ` high & persistent under attack → **latent** abstention (headline + E5 restoration).
`μ` collapses → **surface** abstention (sharp negative result).
Mixed across models → model-specific depth (comparative study). E4 + E1 decide the
branch from data in the first runs.

## Ethics

The forcing grammar elicits harmful continuations from open models; we measure
logits and **do not persist decoded harmful text** (only numeric per-step
records). Published attack used solely as an instrument, standard benchmarks, no
new capability, restoration is safety-positive. See the paper's Ethical
considerations section.
```
