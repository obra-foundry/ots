"""Per-component admission (axioms §0.5.1): large κ_3 cannot mask a near-zero κ_4."""

import torch

from structural_impedance.cumulant import admit_per_component, cross_cumulant_residual_perK

torch.manual_seed(0)


def test_large_k3_cannot_mask_small_k4():
    # Construct X,Y where the 3rd-order cross signature is large but the 4th is tiny,
    # then set tau_4 above the observed ||k4|| so aggregate masking would wrongly admit.
    n = 40000
    X = torch.randn(n, 1, dtype=torch.float64)
    Y = (X ** 2) + 0.05 * torch.randn(n, 1, dtype=torch.float64)  # strong odd-order coupling

    n3 = cross_cumulant_residual_perK(X, Y, 3).norm().item()
    n4 = cross_cumulant_residual_perK(X, Y, 4).norm().item()

    # pick a τ_4 strictly between (small) n4 budget and (large) n3 so per-component blocks
    tau_4 = n4 * 10.0
    tau_3 = n3 * 0.5
    assert n3 > tau_3 and n4 <= tau_4   # precondition: aggregate would have been fooled

    admit, meta = admit_per_component(X, Y, tau_3=tau_3, tau_4=tau_4)
    assert admit is False
    assert meta["blocking_component"] == "k4"


def test_admit_when_both_exceed():
    n = 40000
    X = torch.randn(n, 1, dtype=torch.float64)
    Y = X + 0.1 * torch.randn(n, 1, dtype=torch.float64)
    admit, meta = admit_per_component(X, Y, tau_3=1e-6, tau_4=1e-6)
    assert admit is True
    assert meta["blocking_component"] is None
