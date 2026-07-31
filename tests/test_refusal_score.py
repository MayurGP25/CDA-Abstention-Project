"""sequence_refusal_logprob: the chunked / logits_to_keep path must return
exactly what the naive full-logits path returned.

The rewrite exists because the `schema` arm of the format sweep OOMed: a long
system prompt makes the (P, L+maxk, |V|) fp32 logit tensor tens of gigabytes.
Two things changed at once -- trailing-position logits and chunking -- and both
touch the index arithmetic that maps a logit row to the token it predicts. That
is worth pinning against a reference rather than trusting.
"""
import torch

from abstention.refusal_score import sequence_refusal_logprob


class FakeOut:
    def __init__(self, logits):
        self.logits = logits


class FakeLM:
    """Deterministic stand-in. Logits are a fixed function of position and token
    id, so the reference and the implementation must agree exactly, and
    `logits_to_keep` is honoured the way transformers honours it: the LAST k
    positions."""

    def __init__(self, vocab=37, support_keep=True):
        self.vocab = vocab
        self.support_keep = support_keep
        self.calls = []

    def _full(self, input_ids):
        b, s = input_ids.shape
        pos = torch.arange(s, dtype=torch.float32).view(1, s, 1)
        tok = torch.arange(self.vocab, dtype=torch.float32).view(1, 1, self.vocab)
        seed = input_ids.float().unsqueeze(-1)
        return torch.sin(pos * 0.7 + tok * 0.13 + seed * 0.31)

    def __call__(self, input_ids=None, attention_mask=None, logits_to_keep=None):
        if logits_to_keep is not None and not self.support_keep:
            raise TypeError("unexpected keyword argument 'logits_to_keep'")
        full = self._full(input_ids)
        self.calls.append(input_ids.shape[0])
        if logits_to_keep:
            return FakeOut(full[:, -logits_to_keep:, :])
        return FakeOut(full)


def _reference(model, ctx_ids, prefixes, pad_id=0):
    """The original implementation: one forward, full logits, absolute indices."""
    import torch.nn.functional as F
    P, L = len(prefixes), int(ctx_ids.shape[0])
    maxk = max(len(r) for r in prefixes)
    batch = torch.full((P, L + maxk), pad_id, dtype=torch.long)
    attn = torch.zeros((P, L + maxk), dtype=torch.long)
    for p, r in enumerate(prefixes):
        batch[p, :L] = ctx_ids
        batch[p, L:L + len(r)] = torch.tensor(r)
        attn[p, :L + len(r)] = 1
    logprobs = F.log_softmax(model(input_ids=batch, attention_mask=attn).logits.float(), -1)
    out = torch.empty(P)
    for p, r in enumerate(prefixes):
        pos = torch.arange(L - 1, L - 1 + len(r))
        out[p] = logprobs[p, pos, torch.tensor(r)].sum()
    return float(torch.logsumexp(out, 0))


PREFIXES = [[3, 4], [5], [6, 7, 8], [9, 10], [11], [12, 13, 14, 15]]
CTX = torch.tensor([1, 2, 3, 4, 5, 6, 7])


def test_matches_the_naive_implementation():
    m = FakeLM()
    ref = _reference(FakeLM(), CTX, PREFIXES)
    got = sequence_refusal_logprob(m, CTX, PREFIXES, device="cpu", chunk=8)
    assert abs(got - ref) < 1e-5


def test_chunking_does_not_change_the_value():
    ref = _reference(FakeLM(), CTX, PREFIXES)
    for c in (1, 2, 3, len(PREFIXES), len(PREFIXES) + 5):
        got = sequence_refusal_logprob(FakeLM(), CTX, PREFIXES, device="cpu", chunk=c)
        assert abs(got - ref) < 1e-5, f"chunk={c} diverged"


def test_chunk_actually_splits_the_batch():
    m = FakeLM()
    sequence_refusal_logprob(m, CTX, PREFIXES, device="cpu", chunk=2)
    assert m.calls == [2, 2, 2], m.calls          # 6 prefixes, 3 forwards
    m2 = FakeLM()
    sequence_refusal_logprob(m2, CTX, PREFIXES, device="cpu", chunk=99)
    assert m2.calls == [6]


def test_falls_back_when_logits_to_keep_unsupported():
    """Older transformers raise TypeError. The fallback slices the full tensor
    and must land on the same rows."""
    ref = _reference(FakeLM(), CTX, PREFIXES)
    old = FakeLM(support_keep=False)
    got = sequence_refusal_logprob(old, CTX, PREFIXES, device="cpu", chunk=4)
    assert abs(got - ref) < 1e-5


def test_empty_prefix_set_is_nan():
    import math
    assert math.isnan(sequence_refusal_logprob(FakeLM(), CTX, [], device="cpu"))
