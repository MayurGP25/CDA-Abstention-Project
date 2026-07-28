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


def test_masking_reduces_served_entropy():
    # Masking shrinks the support, so served (post-mask) entropy <= latent.
    V = 64
    R = torch.tensor([1, 2, 3])
    p_free, p_con, allowed = _dists(V, masked_ids=[1, 2, 3, 20, 21])
    forced = int(torch.argmax(p_con).item())
    m = compute_metrics(p_free, p_con, allowed, R, forced)
    assert m.H_post <= m.H_pre + 1e-9


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
