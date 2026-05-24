"""Φ endpoints: γ*=None ⇒ 0; Γ=γ* ⇒ 0; |Γ|→1 ⇒ →1; output in [0,1]."""

from structural_impedance.gamma_correction import phi_modulation


def test_none_gamma_star():
    assert phi_modulation(0.4 + 0.2j, None) == 0.0


def test_impedance_match():
    assert phi_modulation(0.4 + 0.2j, 0.4 + 0.2j) == 0.0


def test_total_reflection_limit():
    # |Γ| -> 1 ⇒ denom -> 0 ⇒ Φ -> 1.0 (total-reflection branch)
    assert phi_modulation(1 + 0j, 0.3 + 0j) == 1.0


def test_output_in_unit_interval():
    for g, gs in [(0.1 + 0j, 0.5 + 0j), (0.3 + 0.2j, -0.1 + 0.4j), (0.9 + 0j, 0.0 + 0j)]:
        phi = phi_modulation(g, gs)
        assert 0.0 <= phi <= 1.0
