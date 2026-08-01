"""CPU-only unit tests for the metric core. No GPU / model / xgrammar needed.

These guard exactly the bug class that sank the draft proposal (a divergent KL)
and the boundedness of mu/alpha. Run anywhere: `pytest tests/test_metrics.py`.
"""
import math

import torch

from abstention.metrics import compute_metrics


def _dists(V, masked_ids):
    """Build a free dist and a constrained dist that zeros out masked_ids."""
    logits = torch.randn(V)
    p_free = torch.softmax(logits, dim=-1)
    allowed = torch.ones(V, dtype=torch.bool)
    allowed[masked_ids] = False
    masked_logits = torch.where(allowed, logits, torch.full_like(logits, float("-inf")))
    p_con = torch.softmax(masked_logits, dim=-1)
    return p_free, p_con, allowed


def test_kl_is_finite_when_refusals_masked():
    V = 128
    R = torch.tensor([3, 4, 5])
    p_free, p_con, allowed = _dists(V, masked_ids=R.tolist())
    forced = int(torch.argmax(p_con).item())
    m = compute_metrics(p_free, p_con, allowed, R, forced)
    # The whole point: reverse KL stays finite even though refusal tokens are
    # masked (P_con=0 there while P_free>0). The forward direction would be +inf.
    assert math.isfinite(m.D)
    assert m.D >= -1e-9


def test_mu_matches_free_mass_on_R():
    V = 64
    R = torch.tensor([1, 2, 10])
    p_free, p_con, allowed = _dists(V, masked_ids=[1, 2, 10])
    forced = int(torch.argmax(p_con).item())
    m = compute_metrics(p_free, p_con, allowed, R, forced)
    assert abs(m.mu - float(p_free[R].sum())) < 1e-6


def test_alpha_in_unit_interval():
    V = 64
    R = torch.tensor([1, 2, 3])
    p_free, p_con, allowed = _dists(V, masked_ids=[1, 2, 3, 20, 21])
    forced = int(torch.argmax(p_con).item())
    m = compute_metrics(p_free, p_con, allowed, R, forced)
    assert 0.0 - 1e-9 <= m.alpha <= 1.0 + 1e-9


def test_free_condition_has_zero_kl_and_undefined_alpha():
    V = 32
    R = torch.tensor([0, 1])
    logits = torch.randn(V)
    p_free = torch.softmax(logits, dim=-1)
    allowed = torch.ones(V, dtype=torch.bool)  # nothing masked
    m = compute_metrics(p_free, p_free, allowed, R, int(torch.argmax(p_free)))
    assert abs(m.D) < 1e-6  # -log(sum softmax); ~0 up to float precision
    # alpha is 0/0 here. It must be NaN, not 0.0 -- a 0.0 would be silently
    # averaged into condition means and pull them toward zero.
    assert math.isnan(m.alpha)


def test_alpha_one_when_only_refusals_masked():
    V = 40
    R = torch.tensor([7, 8])
    p_free, p_con, allowed = _dists(V, masked_ids=[7, 8])  # only R masked
    forced = int(torch.argmax(p_con).item())
    m = compute_metrics(p_free, p_con, allowed, R, forced)
    assert abs(m.alpha - 1.0) < 1e-6


def test_entropy_pre_matches_manual_and_is_bounded_in_bits():
    V = 128
    R = torch.tensor([3, 4, 5])
    p_free, p_con, allowed = _dists(V, masked_ids=R.tolist())
    forced = int(torch.argmax(p_con).item())
    m = compute_metrics(p_free, p_con, allowed, R, forced)
    manual_bits = float(-(p_free.double() * p_free.double().log2()).sum())
    assert abs(m.H_pre - manual_bits) < 1e-6
    assert 0.0 <= m.H_pre <= math.log2(V) + 1e-6    # bounded by log2|V| = 7 bits


def test_h_post_is_bounded_by_log2_n_allowed():
    """The Discussion claim: served entropy can never exceed a bound the GRAMMAR
    author picks. 0 bits at a literal, log2(k) with k allowed tokens."""
    V = 256
    R = torch.tensor([0, 1])
    for k in (1, 4, 90):
        logits = torch.randn(V)
        allowed = torch.zeros(V, dtype=torch.bool)
        allowed[torch.randperm(V)[:k]] = True
        masked = torch.where(allowed, logits, torch.full_like(logits, float("-inf")))
        p_free, p_con = torch.softmax(logits, -1), torch.softmax(masked, -1)
        m = compute_metrics(p_free, p_con, allowed, R, int(p_con.argmax()))
        assert m.H_post <= math.log2(k) + 1e-6


def test_served_entropy_can_EXCEED_latent_entropy():
    """Masking shrinks the support but RENORMALISES, so H_post is NOT bounded by
    H_pre. When the mask removes the peak, what remains is flatter and served
    entropy goes UP.

    This is not a corner case -- it is what the grammar does at t0. Observed on
    Qwen2.5-7B/HarmBench: H_pre = 0.03 bits (the model is near-certain it wants
    to refuse) while H_post = 1.02 bits over the 5 allowed JSON-opening tokens.
    Output-side UQ there reports 34x MORE uncertainty than the model has.
    """
    V = 8
    R = torch.tensor([0])
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # peaked on 0
    p_free = torch.softmax(logits, dim=-1)
    allowed = torch.ones(V, dtype=torch.bool)
    allowed[0] = False                                   # mask the peak away
    masked = torch.where(allowed, logits, torch.full_like(logits, float("-inf")))
    p_con = torch.softmax(masked, dim=-1)
    m = compute_metrics(p_free, p_con, allowed, R, 1)
    assert m.H_pre < 0.1                                 # model is near-certain
    assert m.H_post > m.H_pre                            # served entropy is HIGHER
    assert m.H_post <= math.log2(int(allowed.sum())) + 1e-6   # the real bound


def test_forced_literal_position_has_zero_served_entropy():
    # A single-token allowed set (a literal/enum position) => H_post == 0 while
    # the pre-mask distribution can still be uncertain (H_pre > 0). This is the
    # served-vs-latent decoupling the paper measures.
    V = 32
    R = torch.tensor([0, 1])
    logits = torch.randn(V)
    p_free = torch.softmax(logits, dim=-1)
    allowed = torch.zeros(V, dtype=torch.bool)
    allowed[7] = True                                # exactly one token allowed
    masked = torch.where(allowed, logits, torch.full_like(logits, float("-inf")))
    p_con = torch.softmax(masked, dim=-1)
    m = compute_metrics(p_free, p_con, allowed, R, 7)
    assert m.H_post < 1e-6
    assert m.H_pre > 1e-3


def test_free_condition_entropy_equal_pre_post():
    V = 32
    R = torch.tensor([0, 1])
    logits = torch.randn(V)
    p_free = torch.softmax(logits, dim=-1)
    allowed = torch.ones(V, dtype=torch.bool)
    m = compute_metrics(p_free, p_free, allowed, R, int(torch.argmax(p_free)))
    assert abs(m.H_pre - m.H_post) < 1e-6           # no mask => served == latent


# ---------------------------------------------------------------------------
# D in log space. The probability-space path floors Z_A at 1e-12, which on Qwen
# at position 0 censored 73 rows and pinned lower quartiles to a value the
# grammar never produced. These pin the replacement.
# ---------------------------------------------------------------------------
def _mk(logits, allowed_idx):
    V = logits.numel()
    allowed = torch.zeros(V, dtype=torch.bool)
    allowed[torch.tensor(allowed_idx)] = True
    p_free = torch.softmax(logits, -1)
    masked = logits.clone()
    masked[~allowed] = float("-inf")
    return p_free, torch.softmax(masked, -1), allowed


def test_log_space_agrees_with_probability_space_when_both_are_valid():
    """Well above the clamp the two paths must be indistinguishable."""
    torch.manual_seed(0)
    logits = torch.randn(500) * 3.0
    p_free, p_con, allowed = _mk(logits, [1, 2, 3, 4, 5])
    R = torch.tensor([7, 8])
    a = compute_metrics(p_free, p_con, allowed, R, 1)                    # probs
    b = compute_metrics(p_free, p_con, allowed, R, 1, logits=logits)     # logs
    assert abs(a.D - b.D) < 1e-6, (a.D, b.D)


def test_log_space_resolves_below_the_probability_clamp():
    """A mass of ~1e-18 is the regime that actually occurs at t0. The old path
    reports the clamp; the new one reports the true value."""
    V = 200
    logits = torch.zeros(V)
    logits[0] = 0.0                      # the bulk
    gap = 41.45                          # exp(-41.45) ~ 1e-18
    logits[1:6] = -gap
    logits[6:] = -1e4                    # negligible
    p_free, p_con, allowed = _mk(logits, [1, 2, 3, 4, 5])
    R = torch.tensor([0])

    old = compute_metrics(p_free, p_con, allowed, R, 1).D
    new = compute_metrics(p_free, p_con, allowed, R, 1, logits=logits).D

    assert abs(old - 27.631) < 1e-3, f"expected the clamp, got {old}"
    expected = gap - math.log(5.0)       # 5 allowed tokens each exp(-gap)
    assert abs(new - expected) < 1e-3, (new, expected)
    assert new > old, "log space must resolve past the clamp"


def test_log_space_is_invariant_to_logit_shift():
    """Softmax is shift-invariant, so D must be too. A logsumexp that lost its
    max-subtraction would fail this.

    float64 inputs on purpose. In float32, `logits + 500` discards mantissa bits
    of a value of magnitude 5 BEFORE compute_metrics sees it, which shows up as
    ~7e-6 of drift. That is the input being quantised, not the algorithm, and
    testing in float32 would measure the wrong thing.
    """
    torch.manual_seed(1)
    logits = (torch.randn(300) * 5.0).double()
    p_free, p_con, allowed = _mk(logits, [10, 11, 12])
    R = torch.tensor([0])
    base = compute_metrics(p_free, p_con, allowed, R, 10, logits=logits).D
    for shift in (-500.0, -50.0, 50.0, 500.0):
        s = logits + shift
        pf, pc, al = _mk(s, [10, 11, 12])
        got = compute_metrics(pf, pc, al, R, 10, logits=s).D
        assert abs(got - base) < 1e-9, f"shift {shift}: {got} vs {base}"


def test_D_is_zero_when_nothing_is_masked():
    torch.manual_seed(2)
    logits = torch.randn(50)
    p_free = torch.softmax(logits, -1)
    allowed = torch.ones(50, dtype=torch.bool)
    m = compute_metrics(p_free, p_free, allowed, torch.tensor([0]), 0, logits=logits)
    assert abs(m.D) < 1e-9, m.D
