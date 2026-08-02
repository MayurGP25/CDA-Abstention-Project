# Metrics reference

Everything the paper reports, what it means, and why it is there. This becomes
section 3 (Setup) of the paper. Companion to [PAPER_PLAN.md](PAPER_PLAN.md).

## Notation

| Symbol | Meaning |
|---|---|
| `V` | the full vocabulary |
| `t` | decoding step. `t = 0` is the first generated token, the only position the main claim uses |
| `h_t` | context so far: chat-templated prompt plus tokens already emitted |
| `P_free` | the model's own next-token distribution, `softmax(logits)`. What the forward pass produced |
| `A_t` | the set of tokens the grammar permits at step `t`, from the xgrammar bitmask |
| `P_mask` | the distribution the decoder actually samples from: `P_free` restricted to `A_t` and renormalised |

Grammar-constrained decoding is exactly two operations, applied to the logits
after the forward pass: **restrict** to `A_t`, then **renormalise**. Every
quantity below is a functional of `P_free` and `A_t`.

## The metrics

| Symbol | Formula | In plain terms | Why it is in the paper |
|---|---|---|---|
| `H_pre` | `-sum_{v in V} P_free(v) log2 P_free(v)` | How uncertain the **model** is, over the whole vocabulary | The uncertainty you actually want to know about. Computed exactly, with no top-k truncation, because we hold the full distribution |
| `H_post` | `-sum_{v in A_t} P_mask(v) log2 P_mask(v)` | How uncertain the **served distribution** looks | The uncertainty a downstream system can compute. This is the number practitioners read and call model confidence |
| `dH` | `H_post - H_pre` | How far masking moved the entropy, **for one prompt** | **The primary statistic.** Both terms are measured on the same prompt at the same position, so pairing removes prompt-level variance that a two-sample comparison leaves in |
| `R_t` | `sum_{v in A_t} P_free(v)` | The share of the model's probability the grammar admits | The mechanism, and the paper's recommendation. One scalar that fully determines the relationship between `P_free` and `P_mask`, since masking is only restriction plus renormalisation |
| `\|A_t\|` | count of permitted tokens | How many options the grammar leaves | Context, not a claim. It sets the ceiling `H_post <= log2\|A_t\|`, a bound the **grammar author** chose and the model did not |
| AUROC | `P(harmful ranks above benign)` | Ordering agreement, 0.5 = none | The claim is about **ordering** ("ranks inputs backwards"), not calibration. AUROC is the ordering statistic and is scale-free, so the two entropies compare without either being rescaled |

### It is one metric, not three

`H_pre` and `H_post` are the **same quantity** — Shannon entropy of the
next-token distribution — read at two points in one pipeline. The design is not
"we picked several metrics"; it is "we picked one metric and varied where it is
read." `R_t` is not a third metric but the parameter separating those two points.

### Identities worth stating once

```
P_mask(v)  = P_free(v) / R_t          for v in A_t,  0 otherwise
H_post     = H_A / R_t + log2 R_t     where H_A = -sum_{v in A_t} P_free(v) log2 P_free(v)
KL(P_mask || P_free) = -ln R_t
```

The KL is the reverse direction because the forward one diverges: `P_free` puts
mass where `P_mask` is zero. The closed form is Reddy et al. (ICML 2026) Eq. 5,
where `R_t` is written `alpha(h_t)`. We adopt their notation and cite it rather
than rederiving, and we do **not** report the KL, because `R_t` and `log 1/R_t`
are one quantity printed twice.

## Supporting numbers (prose only, never a table column)

| | What it is | Why it is only a sentence |
|---|---|---|
| `P_refuse` | `exp(S_R)`, where `S_R = log sum_r P(r \| h_t)` over a frozen, prefix-free set of 28 refusal openers, scored by teacher forcing | Explains **why** `H_pre` is low on harmful prompts: the model's mass is concentrated on refusal. Promoting it to a column would make this a safety paper judged by the wrong standard |
| unconstrained refusal rate | fraction of harmful prompts refused with no grammar applied | Establishes that the disposition being measured is one the model acts on when free to. One number: 0.88 on both models |

## The `fmt` axis

`fmt` is the **format instruction**: how much the prompt tells the model to
expect structured output, before the grammar is applied. Four levels, set in
`src/abstention/prompting.py`:

| Level | Prompt content | Purpose |
|---|---|---|
| `none` | no system turn at all | The grammar arrives **unannounced**. The regime the main claim is about |
| `neutral` | a system turn with generic assistant text and no format content | Controls for the mere presence of a system turn, so the effect cannot be attributed to that |
| `terse` | "Output ONLY a single JSON object conforming to the provided schema." | The prompt anticipates the format. Wording follows Reddy et al. Appendix H so the comparison is like for like |
| `schema` | the same, plus the schema literal | The most explicit case |

**Why this axis exists.** "The grammar arrives unannounced" was an unstated
constant in the original design. Varying it turns a hidden assumption into a
reported variable, and it is what produces the Scope section: the inversion is a
property of the unannounced regime and weakens once the prompt anticipates the
schema. A reviewer would otherwise, quite rightly, ask.

## Other terms

| Term | Meaning |
|---|---|
| `forced_steps` | The grammar under test. A JSON object of the shape `{"step1": "Step 1: ..."}`. At `t = 0` it permits 5 tokens |
| `neutral_scaffold` | A differently shaped control grammar, used for a non-replication check. Reported as a negative result |
| `harmful_forced` | Harmful prompt, grammar applied. The measurement arm |
| `benign_forced` | Benign prompt, **same** grammar. The control that rules out the grammar alone producing the effect |
| `free` | Harmful prompt, no grammar, short generation. Gives the behavioural refusal rate only |
| HarmBench | Source of harmful prompts |
| Alpaca | Easy benign control. Ordinary instructions |
| XSTest | Hard benign control. Safe prompts that are lexically alarming, so separation cannot come from vocabulary alone |
| `t0` | Position 0, the first generated token. Where the mask bites hardest |

## Aggregation

| Quantity | How | Why |
|---|---|---|
| entropies | mean, in bits | Roughly symmetric; bits match the UQ literature |
| `R_t` | **median with IQR** | Spans orders of magnitude. The mean ran 8-9x the median and once reversed a comparison outright |
| AUROC | point estimate with 95% percentile bootstrap, 10k resamples, both arms resampled at their design sizes | n = 50 per arm. Any AUROC without an interval is unreadable at this n |
| `R_t` in text | also as `-log2 R_t` bits | Precision lives in the exponent. A bf16 logit is good to ~0.06 nats, so a 53-bit reading carries ~0.09 bits regardless of how small the probability is. Nobody argues about 35 bits; people argue about 3.6e-11 |

## Computed but deliberately not reported

`mu` (single-token refusal mass, superseded by the sequence-level score),
`alpha` (refusal share of removed mass), `s` (forced-token surprisal), KL in
bits (identical to `R_t`), the restoration probe, and the `t_open` / `t*`
landmarks. All remain in the artefact. Eight quantities are computed; five reach
the paper, and only three are load-bearing.

## Standing constraints

Generated harmful completions are inspected on screen and never persisted.
Released artefacts contain token identifiers and distributional statistics, not
decoded harmful text. No attack is introduced and no attack-success rates are
reported.
