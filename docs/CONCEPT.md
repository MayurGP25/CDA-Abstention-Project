# Concept: what this project measures and why

Reference explainer for the project and the seed of the paper's intro +
related-work framing. Plain-language; see `src/abstention/metrics.py` for the
formal definitions.

## One-line thesis

Grammar-constrained decoding can *mask a model's refusal tokens out of its valid
output space*. We use that as a controlled instrument to ask: when the refusal
**channel** is gagged, does the refusal **intent** still exist in the model's
next-token distribution (**latent abstention**) or does it vanish (**surface
abstention**)? We read the answer directly off the pre-mask logits.

## Why constrained decoding is the instrument

Structured-output APIs let a caller supply a grammar (e.g. a JSON schema). During
inference the serving stack does **constrained decoding**: at each step it
computes the model's full next-token distribution, then **masks** (sets to −∞)
every token the grammar forbids, and samples from what remains. The mask is
applied *after* the logits are computed. So if the grammar's valid outputs
exclude refusals (e.g. it forces an affirmative "Step 1: …"), the refusal tokens
are set to zero probability **post-hoc** — the model didn't *choose* to comply,
the decoder removed its ability to decline.

That gives a precise scalpel: **silence the refusal channel, then read from the
pre-mask distribution whether the model still wanted to use it.**

## Data plane vs control plane

- **Data plane** = the prompt text (what the user asks).
- **Control plane** = the grammar / schema that shapes the output.

Refusal normally happens on the data plane (the model reads a harmful prompt and
says "I can't"). Our attack lives on the **control plane** — the grammar removes
the refusal from the valid token set. We keep the data-plane prompt fixed so the
only thing changing between conditions is the grammar (or the prompt's
harmfulness, in the control).

## The four metrics (per generation step t)

Let `P_free` be the model's unconstrained next-token distribution and `P_con` the
grammar-masked one actually sampled. `R` = the refusal-initiating token ids.

| Metric | Formula | Reads as |
|---|---|---|
| **μ** (latent refusal mass) | `Σ_{v∈R} P_free(v)` | *How much did the model want to refuse here?* 1.0 = certain. Read from **pre-mask** logits, so measurable even when R is masked. |
| **D** (coercion force) | `KL(P_con ‖ P_free)` over allowed support | *How hard is the grammar shoving it off course?* 0 = not fighting; large = forcing disfavored tokens. **Reverse** KL (finite; the forward direction diverges on masked tokens). |
| **α** (abstention-attributable coercion) | masked-away refusal mass ÷ total masked-away mass | *Is the force aimed at the refusal specifically?* 1.0 = everything blocked was the refusal. |
| **s** (coercion surprisal) | `-log P_free(forced token)` | *How shocked is the model at the token it was forced to emit?* ~0 = would've said it anyway; large = would essentially never. |

`n_allowed` (size of the grammar-allowed set) is context, not a core metric:
1–5 = a forced/structural position (refusals masked); ~vocab-size = an open
region (almost anything allowed, refusals too).

## R — the refusal token set

Harvested **per model** (`refusal_set.harvest`): curated refusal-opener seeds
("I", "I'm", "Sorry", "As an", "I cannot", …) encoded to first-token ids, unioned
with the most frequent first-tokens of the model's actual refusals. **Coverage**
= share of the model's real refusal mass (at position 0) captured by R; we aim
for ≥ 0.9. R is a *heuristic set of refusal openers*, not a ground-truth refusal
vocabulary — hence the required **R-sensitivity ablation** (small vs large R).

## t — generation depth

`t` is the token position in the response. LLMs generate autoregressively, so at
each `t` there's a fresh distribution. We measure the metrics at **every** `t`
because the whole question is about **depth**: does refusal intent live only at
`t=0` (the decision point) and die — *surface* — or persist / reassert at higher
`t` — *latent*? The **μ-vs-t** curve is the centerpiece figure.

## Conditions

| Condition | Prompt | Grammar | Purpose |
|---|---|---|---|
| **free** | harmful | none | reference: what the model *wanted* (it refuses) |
| **harmful_forced** | harmful | forcing grammar | the attack: refusals masked, affirmative forced |
| **benign_forced** | benign | same forcing grammar | **confound control** — isolates refusal-coercion from generic "being constrained" |
| **escape_blocked** *(planned)* | harmful | forcing grammar **+ R masked everywhere** | tests whether abstention survives *inescapable* suppression (see below) |
| **dictattack** *(planned)* | benign keys | dictionary schema | tests whether refusal responds to *assembled* meaning, not literal harmful words |

## What is measured, and is it correct?

Correct **mathematically** (μ is probability mass; reverse-KL is finite —
unit-tested; μ is grammar-independent at t=0 — verified `μ₀ free = forced`).
Correct **conceptually with controls**, given these honest caveats:

1. μ is an **output-level** proxy (refusal-opener mass), not the model's full
   internal state (which could also live in activations we deliberately don't
   read — a scope choice).
2. **t=0 μ is near-tautological** (prompt-determined); the *signal* is the depth
   dynamics — so the paper leads with the depth profile, not t=0.
3. Where the grammar is permissive (`n_allowed ≈ vocab`), a μ spike is the model
   *freely* refusing, not surviving suppression → motivates **escape_blocked**.
4. A μ drop could be "abstention died" **or** "model wrote benign filler" →
   motivates the **benign_forced** control (E4) and the content-degeneracy check.

## The two attack flavors in our framing

- **EnumAttack-style (current):** harmful prompt sent directly; grammar forces an
  affirmative structure and masks refusals. The model clearly "sees" the harmful
  intent, so μ is high at t=0. Clean, primary instrument.
- **DictAttack-style (planned):** the harmful query is split into benign keys
  (data plane) + a benign dictionary (control plane); the model must *reconstruct*
  the query. Here the harmful words never appear literally — so this tests whether
  the refusal signal tracks **assembled meaning** vs surface lexical cues, and we
  expect μ to appear **later** (after reconstruction) rather than at t=0. A
  stronger test of what the abstention signal responds to.

## Related work — what they measured vs us

| Work | Measured | Level | Difference from us |
|---|---|---|---|
| CDA / EnumAttack, DictAttack | ASR (harmful content produced) | output content | they ask "did the attack succeed"; we ask "did refusal intent survive" — we reuse their attack as an instrument |
| Qi et al. (shallow alignment) | per-position KL(aligned ‖ base) | two models | we do free-vs-constrained within **one** model, reading refusal mass directly |
| Arditi et al. (refusal direction) | refusal as a linear direction | activations | we stay at the **logit** level (no probes) |
| Furina | token/semantic entropy | output dist | they build an attack from entropy; we probe an existing attack, measuring refusal mass |
| Circuit Breakers | ASR + utility after intervention | behavior + activations | a defense; we're a measurement |

**The open niche:** refusal probability mass in the *unconstrained* distribution
while the grammar masks those very tokens — free-vs-constrained, one model, across
depth. That exact quantity (μ under masking) is what's new.
