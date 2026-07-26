"""The instrumented decode loop -- the whole method in one place.

We own the generation loop so we can read the model's PRE-MASK logits at every
step (the seam that hosted APIs and vLLM's fused sampler hide). At each step:

    1. forward pass -> raw logits            -> P_free
    2. xgrammar fills the per-token bitmask   -> allowed set
    3. apply the bitmask to a copy of logits  -> P_con
    4. compute mu, D, alpha, s from (P_free, P_con, allowed, R)
    5. emit the constrained-greedy token, advance the matcher + KV cache

`compiled_grammar=None` runs the FREE condition: no mask, P_con == P_free,
greedy over the free distribution, D == 0. This shares the identical loop so the
per-position bookkeeping matches the constrained conditions.
"""
from __future__ import annotations

import torch
import xgrammar as xgr

from .metrics import StepMetrics, compute_metrics
from .model_loader import LoadedModel, encode_chat
from .refusal_score import sequence_refusal_logprob


def _bool_allowed(bitmask: torch.Tensor, vocab: int, device) -> torch.Tensor:
    """True = grammar-allowed, shape (vocab,)."""
    bm = xgr.testing.bitmask_to_bool_mask(bitmask)[0]
    return bm[:vocab].to(device)


@torch.no_grad()
def run(
    lm: LoadedModel,
    messages: list[dict],
    refusal_ids: list[int],
    *,
    compiled_grammar=None,
    max_new_tokens: int = 64,
    refusal_prefix_ids: list[list[int]] | None = None,
    sr_stride: int = 1,
) -> tuple[list[StepMetrics], list[int]]:
    """Run one prompt under one condition; return (per-step metrics, token ids).

    If `refusal_prefix_ids` is given, the sequence-level refusal score S_R is
    computed every `sr_stride` positions (NaN elsewhere). The generated token ids
    are returned for OPTIONAL redacted logging only -- do not persist decoded
    harmful text (see RESPONSIBLE_USE / ethics).
    """
    model, tok, dev = lm.model, lm.tokenizer, lm.device
    V = lm.full_vocab
    R = torch.tensor(sorted(set(refusal_ids)), device=dev, dtype=torch.long)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    input_ids = encode_chat(tok, messages, dev)

    matcher = None
    bitmask = None
    if compiled_grammar is not None:
        matcher = xgr.GrammarMatcher(compiled_grammar)
        bitmask = xgr.allocate_token_bitmask(1, V)

    # Prime the KV cache with the full prompt.
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1, :].float()

    records: list[StepMetrics] = []
    emitted: list[int] = []

    for t in range(max_new_tokens):
        p_free = torch.softmax(logits, dim=-1)

        if matcher is None:
            # FREE condition: no mask.
            allowed = torch.ones(V, dtype=torch.bool, device=dev)
            p_con = p_free
            next_id = int(torch.argmax(p_free).item())
        else:
            matcher.fill_next_token_bitmask(bitmask)
            allowed = _bool_allowed(bitmask, V, dev)
            masked = logits.clone().unsqueeze(0)                 # (1, V)
            xgr.apply_token_bitmask_inplace(masked, bitmask.to(dev))
            masked = masked.squeeze(0)
            p_con = torch.softmax(masked, dim=-1)
            next_id = int(torch.argmax(p_con).item())

        m = compute_metrics(p_free, p_con, allowed, R, next_id)
        # Sequence-level refusal score at this position (context = prompt + emitted so far).
        if refusal_prefix_ids is not None and (t % sr_stride == 0):
            ctx = (torch.cat([input_ids[0], torch.tensor(emitted, device=dev, dtype=torch.long)])
                   if emitted else input_ids[0])
            m.sr = sequence_refusal_logprob(model, ctx, refusal_prefix_ids, dev, pad_id=pad_id)
        records.append(m)
        emitted.append(next_id)

        if matcher is not None:
            matcher.accept_token(next_id)
            if matcher.is_terminated():
                break
        else:
            if next_id == tok.eos_token_id:
                break

        step = model(
            input_ids=torch.tensor([[next_id]], device=dev),
            past_key_values=past,
            use_cache=True,
        )
        past = step.past_key_values
        logits = step.logits[0, -1, :].float()

    return records, emitted


@torch.no_grad()
def restoration_probe(
    lm: LoadedModel,
    messages: list[dict],
    refusal_ids: list[int],
    compiled_grammar,
    *,
    restore_pos: int,
    max_new_tokens: int = 64,
) -> dict:
    """E5: decode under the forcing grammar up to `restore_pos`, then re-allow
    R u {EOS} and ask whether the model would exit into a refusal.

    Returns dict with:
      escaped        : bool, argmax of the re-allowed distribution is in R u {EOS}
      recovered_mass : free-dist mass on R u {EOS} at restore_pos
      mu_at_pos      : latent refusal mass at restore_pos (bounds recovery)
    """
    model, tok, dev = lm.model, lm.tokenizer, lm.device
    V = lm.full_vocab
    R = torch.tensor(sorted(set(refusal_ids)), device=dev, dtype=torch.long)
    eos = tok.eos_token_id

    input_ids = encode_chat(tok, messages, dev)
    matcher = xgr.GrammarMatcher(compiled_grammar)
    bitmask = xgr.allocate_token_bitmask(1, V)

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1, :].float()

    for pos in range(max_new_tokens):
        matcher.fill_next_token_bitmask(bitmask)
        allowed = _bool_allowed(bitmask, V, dev)
        p_free = torch.softmax(logits, dim=-1)

        if pos == restore_pos:
            allowed_r = allowed.clone()
            allowed_r[R] = True
            if eos is not None:
                allowed_r[eos] = True
            masked = torch.where(allowed_r, logits, torch.full_like(logits, float("-inf")))
            p = torch.softmax(masked, dim=-1)
            argmax_id = int(torch.argmax(p).item())
            escaped = bool(argmax_id in set(refusal_ids) or argmax_id == eos)
            recovered = float(p_free[R].sum() + (p_free[eos] if eos is not None else 0.0))
            return dict(
                restore_pos=restore_pos,
                escaped=escaped,
                recovered_mass=recovered,
                mu_at_pos=float(p_free[R].sum()),
            )

        masked = logits.clone().unsqueeze(0)
        xgr.apply_token_bitmask_inplace(masked, bitmask.to(dev))
        next_id = int(torch.argmax(masked.squeeze(0)).item())
        matcher.accept_token(next_id)
        if matcher.is_terminated():
            break
        step = model(input_ids=torch.tensor([[next_id]], device=dev),
                     past_key_values=past, use_cache=True)
        past = step.past_key_values
        logits = step.logits[0, -1, :].float()

    return dict(restore_pos=restore_pos, escaped=False,
               recovered_mass=float("nan"), mu_at_pos=float("nan"))
