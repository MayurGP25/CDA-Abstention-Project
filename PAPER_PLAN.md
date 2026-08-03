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

- **[x] C2 — the mechanism.** VALIDATED 3 Aug 2026. See the working log.
  `R_t` is 1e-9 (Llama) to 1e-11 (Qwen) and does NOT separate the two
  conditions, which is the point. Log-space confirmed live: max reading 51.94
  bits against the old 39.86-bit clamp ceiling.
  `GPU=7 bash scripts/rerun_fmt_logspace.sh`

- **[x] C3 — scope.** VALIDATED 3 Aug 2026. Inversion holds in the unannounced
  regime and in none of the eight aligned cells. Three regimes governed by
  `R_t`. See the working log.
  `python3 experiments/claim3_scope.py --only fmt2,depth`

- **[x] C4 — the supporting sentences.** VALIDATED 3 Aug 2026. `P_refuse` on
  harmful at t0 is 0.83 (Llama) and 0.77 (Qwen) unannounced; the unconstrained
  refusal rate is 0.88 on both models (44/50).

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

### C2 — validated 3 Aug 2026

Recollected with `D` in log space into `results/fmt2/`. **Cross-check passed
exactly**: every entropy and every AUROC reproduced the pre-recollection value
to the printed digit, confirming the change touched `D` and nothing else.

Censoring confirmed cured: max reading 51.94 bits against the old 39.86-bit
clamp ceiling. Medians were never affected (Qwen none harmful is 3.551e-11 both
before and after); the lower quartiles were, e.g. Qwen neutral benign Q1 moved
off exactly 1.0e-12 to 8.2e-13.

**Table 1, complete.** t0, unannounced regime, n = 50 per cell.
Ceiling `log2|A_t|` = 2.322 throughout (|A| = 5).

| model | fmt | arm | H_pre | H_post | H_post/ceiling | R_t median [IQR] | -log2 R_t |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | none | benign | 0.930 | 0.579 | 0.249 | 4.271e-09 [9.9e-10, 1.8e-08] | 27.8 |
| Llama-3.1-8B | none | harmful | 0.208 | 0.787 | 0.339 | 7.743e-09 [1.7e-09, 3.1e-08] | 26.9 |
| Llama-3.1-8B | neutral | benign | 0.922 | 0.696 | 0.300 | 1.455e-09 [3.5e-10, 6.9e-09] | 29.4 |
| Llama-3.1-8B | neutral | harmful | 0.114 | 0.968 | 0.417 | 1.232e-09 [1.5e-10, 5.7e-09] | 29.6 |
| Qwen2.5-7B | none | benign | 0.757 | 1.082 | 0.466 | 1.593e-11 [1.4e-12, 1.8e-10] | 35.9 |
| Qwen2.5-7B | none | harmful | 0.071 | 1.255 | 0.541 | 3.551e-11 [6.8e-12, 8.1e-11] | 34.7 |
| Qwen2.5-7B | neutral | benign | 0.805 | 1.057 | 0.455 | 9.456e-12 [8.2e-13, 7.5e-11] | 36.6 |
| Qwen2.5-7B | neutral | harmful | 0.062 | 1.423 | 0.613 | 7.394e-12 [1.5e-12, 2.5e-11] | 37.0 |

**The mechanism, stated correctly.** `R_t` is nearly identical across the two
conditions (IQRs overlap in every pair), so the AMOUNT of mass removed does not
distinguish them. It is a property of the grammar, which is the paper's point.

The inversion comes from the SHAPE of what survives:

- On harmful prompts the model's mass sits on refusal openers, so all five
  permitted JSON tokens carry tiny and roughly EQUAL mass. Renormalising a flat
  tail gives a near-uniform distribution, hence high `H_post`.
- On benign prompts the model has genuine preference among those same five
  tokens, so the renormalised distribution is peaked and `H_post` is lower.

So `R_t`'s role is to establish that the served distribution is a renormalised
near-zero tail **in both conditions**. It is the enabling condition, not the
discriminator. State it this way: it pre-empts "isn't `R_t` doing the work?"

Note also that Llama's `R_t` is two to three orders of magnitude larger than
Qwen's and both show the inversion, so the effect is not sensitive to the exact
magnitude of the retained mass.

### C3 — validated 3 Aug 2026

Inversion holds in the unannounced regime and in **none** of the eight aligned
cells. 3 of 12 across the whole axis, and the three are exactly the unannounced
ones (Qwen/none marginal at [0.499, 0.719]).

**Sign test across the format axis.**

| model | fmt | vs | AUROC(H_pre) [CI] | AUROC(H_post) [CI] | inverted |
|---|---|---|---|---|---|
| Llama | none | alpaca | 0.175 [0.100, 0.260] | 0.690 [0.579, 0.795] | YES |
| Llama | neutral | alpaca | 0.125 [0.060, 0.202] | 0.728 [0.622, 0.825] | YES |
| Llama | terse | alpaca | 0.026 [0.005, 0.059] | 0.499 [0.382, 0.618] | no |
| Llama | terse | xstest | 0.392 [0.276, 0.511] | 0.733 [0.627, 0.830] | no |
| Llama | schema | alpaca | 0.049 [0.014, 0.096] | 0.439 [0.327, 0.554] | no |
| Llama | schema | xstest | 0.301 [0.201, 0.411] | 0.478 [0.362, 0.594] | no |
| Qwen | none | alpaca | 0.158 [0.073, 0.252] | 0.613 [0.499, 0.719] | marginal |
| Qwen | neutral | alpaca | 0.157 [0.076, 0.250] | 0.750 [0.650, 0.838] | YES |
| Qwen | terse | alpaca | 0.914 [0.851, 0.965] | 0.798 [0.704, 0.882] | no |
| Qwen | terse | xstest | 0.562 [0.443, 0.678] | 0.402 [0.275, 0.532] | no |
| Qwen | schema | alpaca | 0.757 [0.655, 0.849] | 0.339 [0.235, 0.450] | no |
| Qwen | schema | xstest | 0.403 [0.290, 0.517] | 0.277 [0.176, 0.381] | no |

**Retained mass across the same axis**, harmful / benign-alpaca medians:

| model | none | neutral | terse | schema |
|---|---|---|---|---|
| Llama | 7.7e-09 / 4.3e-09 | 1.2e-09 / 1.5e-09 | 1.3e-08 / **0.206** | 1.6e-08 / **6.8e-03** |
| Qwen | 3.6e-11 / 1.6e-11 | 7.4e-12 / 9.5e-12 | **0.278** / **0.9995** | **0.841** / **0.663** |

**THREE REGIMES, governed by one scalar.** This is the scope paragraph, and it
demonstrates the paper's recommendation rather than asserting it.

1. **Both `R_t` tiny** (unannounced, both models). The served distribution is a
   renormalised near-zero tail. `H_post` inverts relative to `H_pre`.
2. **Both `R_t` large** (Qwen under terse/schema). The served distribution is
   close to the model's own. `H_pre` and `H_post` now AGREE (0.914 and 0.798),
   which is the predicted non-inverted behaviour.
3. **Mixed** (Llama under terse: benign 0.206, harmful 1.3e-08). `H_pre` still
   separates strongly (0.026) but `H_post` carries no signal at all (0.499).

**Why the two models leave the regime differently**, and this is the honest
explanation the section needs. Under an explicit format instruction Llama keeps
its refusal disposition while Qwen's collapses:

| P_refuse on harmful | none | neutral | terse | schema |
|---|---|---|---|---|
| Llama | 0.830 | 0.884 | 0.892 | 0.869 |
| Qwen | 0.769 | 0.660 | 0.071 | 0.000 |

So for Qwen under an aligned prompt the premise of the claim fails outright:
there is no forbidden preference left to invert. For Llama the preference
survives but the mask stops being the dominant effect on the benign side.

**Also state:** Qwen's XSTest `H_post` (1.481 unannounced) exceeds its harmful
`H_post` (1.255). Safe-but-alarming prompts look more uncertain than harmful
ones, because `H_post` inherits the model's over-refusal. A caveat, not a
counterexample, and it belongs in Scope.

### C4 — validated 3 Aug 2026

Both numbers are prose, never table columns.

- **`P_refuse` on harmful at t0, unannounced:** 0.830 (Llama), 0.769 (Qwen).
  This is why `H_pre` is low there. Without it, "the model is more certain on
  harmful prompts" is an unexplained fact.
- **Unconstrained refusal rate:** 44/50 = **0.88 on both models**. The
  disposition being measured is one the model acts on when free to.
