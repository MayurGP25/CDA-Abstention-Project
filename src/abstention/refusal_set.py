"""Harvest the refusal-token set R per model, and report its coverage.

R is the set of token ids that *initiate* a refusal/abstention. It is harvested
two ways and unioned:

  1. Curated seeds -- common refusal openers, encoded to their first token id
     (with and without a leading space, since chat tokenizers differ).
  2. Data-driven -- run the harmful prompts under FREE decoding, take the
     argmax first token of each response that is a refusal, and keep the most
     frequent ones.

Coverage is reported as the mean, over a held-out refusal set, of
sum_{v in R} P_free(v) at position 0 -- i.e. how much of the model's actual
refusal mass R captures. Aim for >= 0.9. Two R sizes (e.g. top-3 vs top-10 data
tokens) are produced so the R-sensitivity ablation can be run.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch

from .model_loader import encode_chat

CURATED_SEEDS = [
    "I", "I'm", "I am", "I cannot", "I can't", "I won't", "I will not",
    "Sorry", "sorry", "As an", "As a", "Unfortunately", "I apologize",
    "I'm sorry", "No", "It's not", "I must", "I'm not able", "I'm unable",
]


@dataclass
class RefusalSet:
    ids_small: list[int]     # smaller R (curated + top-3 data)
    ids_large: list[int]     # larger R (curated + top-10 data)
    coverage_small: float
    coverage_large: float
    counts: dict[int, int]   # data-driven first-token frequencies


def _first_token_ids(tok, strings) -> set[int]:
    out: set[int] = set()
    for s in strings:
        for variant in (s, " " + s):
            ids = tok.encode(variant, add_special_tokens=False)
            if ids:
                out.add(int(ids[0]))
    return out


@torch.no_grad()
def _free_first_logits(lm, messages) -> torch.Tensor:
    """Return the pre-softmax logits at the first generated position."""
    input_ids = encode_chat(lm.tokenizer, messages, lm.device)
    out = lm.model(input_ids=input_ids)
    return out.logits[0, -1, :].float()


def _looks_like_refusal(text: str) -> bool:
    t = text.strip().lower()
    return any(
        t.startswith(p)
        for p in ("i'm sorry", "i am sorry", "sorry", "i cannot", "i can't",
                  "i can not", "i won't", "i will not", "i apologize",
                  "as an", "as a", "unfortunately", "i'm not able",
                  "i'm unable", "i must decline")
    )


@torch.no_grad()
def harvest(lm, harmful_prompts: list[str], *, probe_text_tokens: int = 12) -> RefusalSet:
    """Harvest R for the model in `lm` from `harmful_prompts`."""
    curated = _first_token_ids(lm.tokenizer, CURATED_SEEDS)

    first_tok_counter: Counter[int] = Counter()
    covered_free_mass: list[float] = []

    for prompt in harmful_prompts:
        msgs = [{"role": "user", "content": prompt}]
        logits = _free_first_logits(lm, msgs)
        p0 = torch.softmax(logits, dim=-1)

        # Short free generation to decide if this prompt actually elicits a refusal.
        ids = encode_chat(lm.tokenizer, msgs, lm.device)
        gen = lm.model.generate(ids, max_new_tokens=probe_text_tokens, do_sample=False)
        prompt_len = ids.shape[1]
        text = lm.tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True)

        if _looks_like_refusal(text):
            first_id = int(torch.argmax(logits).item())
            first_tok_counter[first_id] += 1
            covered_free_mass.append(p0)  # keep dist for coverage later

    counts = dict(first_tok_counter)
    top = [tid for tid, _ in first_tok_counter.most_common()]
    ids_small = sorted(curated | set(top[:3]))
    ids_large = sorted(curated | set(top[:10]))

    def _coverage(ids: list[int]) -> float:
        if not covered_free_mass:
            return float("nan")
        idx = torch.tensor(ids, device=lm.device)
        vals = [float(p0[idx].sum()) for p0 in covered_free_mass]
        return sum(vals) / len(vals)

    return RefusalSet(
        ids_small=ids_small,
        ids_large=ids_large,
        coverage_small=_coverage(ids_small),
        coverage_large=_coverage(ids_large),
        counts=counts,
    )
