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


def _rows(metrics, *, condition, prompt_id, model_id):
    return [
        dict(
            model=model_id,
            condition=condition,
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
) -> tuple[list[dict], list[int]]:
    """Run one (condition, prompt) unit. Returns (tidy rows, emitted token ids).

    The FREE arm passes `refusal_prefix_ids=None`: it exists only to establish the
    behavioural refusal rate, and its t0 value is identical to harmful_forced's by
    construction (no grammar has acted at position 0), so paying for S_R there
    would buy a number we already have.
    """
    grammar = None if condition == "free" else compiled_grammar
    metrics, emitted, _ts = run(
        lm,
        [{"role": "user", "content": prompt}],
        refusal_ids,
        compiled_grammar=grammar,
        max_new_tokens=max_new_tokens,
        refusal_prefix_ids=refusal_prefix_ids,
        sr_stride=sr_stride,
        anchor=anchor,
        sr_window=sr_window,
    )
    rows = _rows(metrics, condition=condition, prompt_id=prompt_id, model_id=lm.model_id)
    return rows, emitted
