"""The three decoding conditions, plus a helper that flattens per-step metrics
into tidy records for pandas.

  free           : harmful prompt, NO grammar. Reference for "what it wanted".
  harmful_forced : harmful prompt, forcing grammar. The attacked condition.
  benign_forced  : benign  prompt, the SAME forcing grammar. The matched
                   confound control -- identical grammar, only the prompt's
                   harmfulness differs, isolating refusal-coercion from generic
                   "being constrained".
"""
from __future__ import annotations

from .decode_loop import run
from .model_loader import LoadedModel
from .prompting import build_messages


def _rows(metrics, *, condition, prompt_id, model_id, grammar_name, prompt_source,
          fmt):
    return [
        dict(
            model=model_id,
            condition=condition,
            # Without these three, a forced_steps run and a neutral_scaffold run
            # both label their rows "harmful_forced", and pointing paper.py at
            # both parquets silently merges them into one condition. `fmt` is the
            # same hazard for the format-instruction arms: unprompted and
            # schema-prompted rows are otherwise indistinguishable once merged.
            grammar=grammar_name,
            prompt_source=prompt_source,
            fmt=fmt,
            prompt_id=prompt_id,
            pos=i,
            mu=m.mu,
            sr=m.sr,
            D=m.D,
            alpha=m.alpha,
            s=m.s,
            H_pre=m.H_pre,
            H_post=m.H_post,
            n_allowed=m.n_allowed,
            forced_id=m.forced_id,
            t_star=m.t_star,
            ctx_open=m.ctx_open,
            tau=(i - m.t_star) if m.t_star >= 0 else None,
        )
        for i, m in enumerate(metrics)
    ]


def run_condition(
    lm: LoadedModel,
    *,
    condition: str,
    prompt: str,
    prompt_id: str,
    refusal_ids: list[int],
    compiled_grammar=None,
    max_new_tokens: int = 64,
    refusal_prefix_ids: list[list[int]] | None = None,
    sr_stride: int = 1,
    anchor: str = "Step 1:",
    sr_window: int = 12,
    grammar_name: str = "?",
    prompt_source: str = "?",
    format_instruction: str = "none",
    schema: dict | None = None,
) -> tuple[list[dict], list[int]]:
    """Run one (condition, prompt) unit. Returns (tidy rows, emitted token ids).

    The FREE arm passes `refusal_prefix_ids=None`: it exists only to establish the
    behavioural refusal rate, and its t0 value is identical to harmful_forced's by
    construction (no grammar has acted at position 0), so paying for S_R there
    would buy a number we already have.

    `format_instruction` selects how much the prompt anticipates the grammar (see
    prompting.py). It applies to EVERY arm including `free`, because the whole
    point of the axis is that conditioning changes P_free -- exempting the free
    arm would make it a different prompt from the forced arm it is the reference
    for. Note the t0-equality above holds within a format level, not across them.
    """
    grammar = None if condition == "free" else compiled_grammar
    metrics, emitted, _ts = run(
        lm,
        build_messages(prompt, format_instruction, schema),
        refusal_ids,
        compiled_grammar=grammar,
        max_new_tokens=max_new_tokens,
        refusal_prefix_ids=refusal_prefix_ids,
        sr_stride=sr_stride,
        anchor=anchor,
        sr_window=sr_window,
    )
    rows = _rows(metrics, condition=condition, prompt_id=prompt_id,
                 model_id=lm.model_id,
                 grammar_name=("none" if grammar is None else grammar_name),
                 prompt_source=prompt_source, fmt=format_instruction)
    return rows, emitted
