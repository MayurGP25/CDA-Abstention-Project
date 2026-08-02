# UncertaiNLP 2026 short paper — locked plan

Deadline 7 August 2026. Four pages, references and appendix unlimited.

Status legend: **[ ]** not yet validated, **[x]** validated against data pasted
into the working log at the bottom.

---

## The claim

> Under a grammar the prompt does not anticipate, the entropy of the served
> distribution is anti-correlated with the model's own entropy. A system that
> gates on observed token uncertainty therefore ranks inputs in the opposite
> order to the model's actual confidence.

One observation, one mechanism, one consequence, one remedy.

- **Mechanism.** The mask retains ~1e-11 of the model's probability mass at the
  first token, so what remains is close to a uniform draw over a handful of
  syntactic tokens. How uniform depends on how concentrated the model's
  preference was *outside* the allowed set, which inverts the ordering.
- **Consequence.** Uncertainty-gated abstention, routing and hallucination
  detection fire backwards on structured-output pipelines.
- **Remedy.** Report retained mass alongside any entropy computed under a
  constraint. If it is small, the entropy is a property of the grammar.

## Why anyone should care

Structured output (JSON mode, function calling, tool use) and uncertainty-based
routing are both standard production practice. Nobody has reported what happens
when they are composed. The answer is that the signal changes sign, and the
check that detects it is a scalar every constrained-decoding library already
computes internally but does not expose.

## Metrics — three, and only three

| | | Role |
|---|---|---|
| `H_pre` | entropy of the pre-mask distribution, bits | the model's own uncertainty |
| `H_post` | entropy of the served distribution, bits | what a downstream system can compute |
| `R_t` | retained mass, `sum_{v in A} P_free(v)` | the mechanism, and the proposed check |

Supporting, not headline: `|A_t|` gives the ceiling `H_post <= log2|A_t|`.
AUROC summarises separation. `P_refuse` appears **once, in prose**, to explain
why `H_pre` is low on harmful prompts. Everything else computed
(`mu`, `alpha`, `s`, KL in bits, restoration, `t_open`, `t*`) stays in the
artefact and out of the paper.

Reddy et al. (ICML 2026): two sentences. Source of the alpha notation and the
`KL = log 1/alpha` identity, plus the sequence-averaged agreement as a sanity
check. Not a comparison and not a competition.

## Structure

| Section | Pages | Contains |
|---|---|---|
| Abstract + 1 Introduction | 1.0 | The composition nobody has checked; the sign result up front |
| 2 Background | 0.5 | Constrained decoding, UQ metrics, Reddy et al., Steindl on position |
| 3 Setup | 0.5 | Definitions, models, prompts, the landmark, the entropy ceiling |
| 4 Result | 1.25 | **Table 1**, **Table 2**, **Figure 1** |
| 5 Scope | 0.35 | Where the inversion stops holding (the format sweep) |
| 6 Discussion | 0.4 | The remedy; what it means for uncertainty-gated systems |
| Limitations, Ethics, Appendix | outside | |

- **Table 1** — `H_pre`, `H_post`, `log2|A_t|`, `R_t` per model x arm.
- **Table 2** — AUROC of `H_pre` and `H_post` with bootstrap CIs. The sign test.
- **Figure 1** — pre-mask and post-mask distributions for one harmful and one
  benign prompt, showing the ordering swap directly.

## Claims to validate, in order

Each is one command. Nothing is written until the claim above it is checked.

- **[x] C1 — the inversion.** VALIDATED 3 Aug 2026. See the working log.
  Primary statistic is the PAIRED gap `H_post - H_pre`, 4/4 cells, AUROC
  0.845-0.916. Marginal test 3/4, with Qwen/none's `H_post` interval touching
  0.5. Both were pre-specified; the paired one is primary because the two
  entropies are measured on the same prompt at the same position.
  `python3 experiments/claim1_inversion.py`

- **[ ] C2 — the mechanism.** `R_t` is tiny and `H_post` sits near its ceiling
  `log2|A_t|`, in the same cells. Needs the log-space re-collection, since the
  old numbers were floored at 1e-12.
  `GPU=7 bash scripts/rerun_fmt_logspace.sh`

- **[ ] C3 — scope.** Where the inversion stops. The format sweep, reported as
  a boundary rather than a result.

- **[ ] C4 — the supporting sentence.** `P_refuse` on harmful under the free
  condition, to explain low `H_pre`. One number.

## Decisions already made, not to be reopened

- Position: the first generated token. Justified because it is where the mask
  bites hardest, and Reddy et al.'s own figure agrees.
- Benign control: Alpaca for the main claim, XSTest as the hard negative in
  Scope.
- Median with IQR for masses, mean for entropies, AUROC for separation.
- Entropies in bits.
- No attack-success rates, no decoded harmful text in any released artefact.

---

## Working log

Paste validated output under the matching claim, with the date.

### C1 — validated 3 Aug 2026

Unannounced regime, forced_steps, t0, Alpaca control, n = 50/50 per cell,
95% percentile bootstrap over 10k resamples.

**Table 1 draft — the two entropies, bits.** Ceiling `log2|A_t|` = 2.322 (|A| = 5).

| model | fmt | arm | H_pre | H_post |
|---|---|---|---|---|
| Llama-3.1-8B | neutral | benign | 0.922 | 0.696 |
| Llama-3.1-8B | neutral | harmful | 0.114 | 0.968 |
| Llama-3.1-8B | none | benign | 0.930 | 0.579 |
| Llama-3.1-8B | none | harmful | 0.208 | 0.787 |
| Qwen2.5-7B | neutral | benign | 0.805 | 1.057 |
| Qwen2.5-7B | neutral | harmful | 0.062 | 1.423 |
| Qwen2.5-7B | none | benign | 0.757 | 1.082 |
| Qwen2.5-7B | none | harmful | 0.071 | 1.255 |
| Qwen2.5-7B | none | benign/xstest | 0.526 | 1.481 |

`R_t` column still pending C2: the old numbers were floored at 1e-12.

**Table 2 draft — primary, the paired shift.**

| model | fmt | dH harmful | dH benign | AUROC [95% CI] |
|---|---|---|---|---|
| Llama-3.1-8B | neutral | +0.853 | -0.226 | 0.898 [0.828, 0.954] |
| Llama-3.1-8B | none | +0.579 | -0.351 | 0.845 [0.761, 0.918] |
| Qwen2.5-7B | neutral | +1.362 | +0.252 | 0.916 [0.856, 0.964] |
| Qwen2.5-7B | none | +1.184 | +0.325 | 0.856 [0.774, 0.924] |

**Table 2 draft — the decomposition.**

| model | fmt | AUROC(H_pre) [CI] | AUROC(H_post) [CI] |
|---|---|---|---|
| Llama-3.1-8B | neutral | 0.125 [0.060, 0.202] | 0.728 [0.622, 0.825] |
| Llama-3.1-8B | none | 0.175 [0.100, 0.260] | 0.690 [0.579, 0.795] |
| Qwen2.5-7B | neutral | 0.157 [0.077, 0.250] | 0.750 [0.651, 0.836] |
| Qwen2.5-7B | none | 0.158 [0.072, 0.255] | 0.613 [0.500, 0.720] |

`H_post` as a fraction of its ceiling runs 0.25 to 0.64, so the served entropy
has real dynamic range and is not simply saturated at `log2|A_t|`.

**To state in the paper, and not to overstate**

- `H_pre` orders harmful below benign in every cell, decisively.
- `H_post` orders harmful above benign in every cell; the interval excludes 0.5
  in three of four. Qwen/none is marginal at [0.500, 0.720] and must be
  reported as marginal.
- The benign shift is negative for Llama and positive for Qwen, so the
  model-general claim is about MAGNITUDE. Do not write that masking deflates
  benign entropy.
- The mechanism is simple once stated: on harmful prompts the model's mass sits
  on refusal openers, so the five permitted JSON tokens carry little and roughly
  equal mass, and renormalising them gives a near-uniform distribution. Explain
  it plainly. The contribution is that nobody checks, it is cheap to check, and
  the consequence is real.

### C2
_pending_

### C3
_pending_

### C4
_pending_
