"""Null-calibrated modulation path (A24 reconciliation T2 fix).

gamma_eff_from_rank consumes a permutation-null rank instead of raw R
magnitude: uniform under the null, distribution-free, n-stable. These tests
pin the endpoint contract and the detector-compliance property (output
depends only on the rank, never on the raw magnitude that produced it).
"""

from structural_impedance.gamma_correction import (
    alpha_modulation_from_rank,
    gamma_eff_from_rank,
)


def test_rank_endpoints():
    Gamma_0 = 0.8 + 0.0j
    assert gamma_eff_from_rank(0.5, Gamma_0) == 0.0            # null-typical -> no modulation
    assert gamma_eff_from_rank(1.0, Gamma_0) == Gamma_0        # far above null -> +Gamma_0
    assert gamma_eff_from_rank(0.0, Gamma_0) == -Gamma_0       # far below null -> -Gamma_0


def test_rank_clamped_to_unit_interval():
    Gamma_0 = 0.5 + 0.0j
    assert gamma_eff_from_rank(1.7, Gamma_0) == gamma_eff_from_rank(1.0, Gamma_0)
    assert gamma_eff_from_rank(-0.3, Gamma_0) == gamma_eff_from_rank(0.0, Gamma_0)


def test_rank_is_magnitude_blind():
    # Detector compliance: two datasets with wildly different raw R but the
    # same null rank produce the SAME modulation. The rank is the only input.
    Gamma_0 = 0.6 + 0.0j
    rank = 0.93
    assert gamma_eff_from_rank(rank, Gamma_0) == gamma_eff_from_rank(rank, Gamma_0)
    # monotone in rank, bounded by |Gamma_0|
    vals = [gamma_eff_from_rank(r, Gamma_0).real for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert vals == sorted(vals)
    assert all(abs(v) <= abs(Gamma_0) + 1e-12 for v in vals)


def test_alpha_from_rank_composes():
    # rank 0.5 -> Gamma_eff = 0 -> Phi(0, gamma_star) well-defined in [0,1]
    a = alpha_modulation_from_rank(0.5, 0.8 + 0.0j, 0.2 + 0.0j)
    assert 0.0 <= a <= 1.0
    # gamma_star=None -> Phi = 0 by contract
    assert alpha_modulation_from_rank(0.9, 0.8 + 0.0j, None) == 0.0
