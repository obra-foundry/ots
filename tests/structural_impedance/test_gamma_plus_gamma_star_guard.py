"""(Γ+γ*) singularity guard: returns tagged obstruction, no NaN/Inf."""

import math

from structural_impedance.gamma_correction import metric_warp_factor


def test_singular_denominator_returns_tag():
    factor, tag = metric_warp_factor(Gamma=1 + 0j, gamma_star=-1 + 0j)
    assert factor == 1.0
    assert tag == "metric_warp:gamma_plus_gamma_star_singular"
    assert not math.isnan(factor) and not math.isinf(factor)


def test_none_gamma_star_short_circuit():
    factor, tag = metric_warp_factor(Gamma=0.3 + 0.1j, gamma_star=None)
    assert factor == 1.0 and tag is None


def test_nonsingular_warp_is_finite_ge_one():
    factor, tag = metric_warp_factor(Gamma=0.2 + 0.0j, gamma_star=0.5 + 0.0j)
    assert tag is None
    assert factor >= 1.0
    assert not math.isnan(factor) and not math.isinf(factor)
