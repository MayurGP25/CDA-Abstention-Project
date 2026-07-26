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
            D=m.D,
            alpha=m.alpha,
            s=m.s,
            n_allowed=m.n_allowed,
            forced_id=m.forced_id,
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
) -> list[dict]:
    grammar = None if condition == "free" else compiled_grammar
    metrics, _emitted = run(
        lm,
        [{"role": "user", "content": prompt}],
        refusal_ids,
        compiled_grammar=grammar,
        max_new_tokens=max_new_tokens,
    )
    return _rows(metrics, condition=condition, prompt_id=prompt_id, model_id=lm.model_id)
