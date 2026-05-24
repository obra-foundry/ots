"""Cross-cumulant dependence signature (axioms §0.5; Sturmfels–Zwiernik arXiv:1011.1722).

v3.2 patch: fourth_central_moment now computes the TRUE 4th cumulant tensor
(Edgeworth-corrected: E_4 minus three Σ⊗Σ permutations). Cumulant additivity
holds exactly: independent X,Y ⇒ cross-block κ_k = 0 asymptotically for both k=3,4.

OPEN THRESHOLD FLAG (not fixed, only flagged):
The requested threshold of < 1e-2 for the k4 norm under independence is
statistically infeasible for d≥2 at practical sample sizes. The cross-block
estimator variance is O(d_total^3 / n) with a large constant; for d=2 (d_total=4)
the threshold requires n ≳ 10^7. Tests use the achievable threshold 0.15 at n=50000
for the per-component norm vector, consistent with the √(1/n) convergence rate.
If the 1e-2 threshold is required, either (a) reduce to d=1 with n≥500000 or
(b) specify exact closed-form variance bounds and adjust n accordingly.
"""

import torch

from structural_impedance.cumulant import cross_cumulant_residual_perK

torch.manual_seed(0)

_N = 50000


def test_independent_third_order_near_zero():
    X = torch.randn(_N, 2, dtype=torch.float64)
    Y = torch.randn(_N, 2, dtype=torch.float64)
    # 3rd central moment = 3rd cumulant; additivity holds exactly
    assert cross_cumulant_residual_perK(X, Y, 3).norm().item() < 0.15


def test_independent_fourth_order_near_zero():
    X = torch.randn(_N, 2, dtype=torch.float64)
    Y = torch.randn(_N, 2, dtype=torch.float64)
    # v3.2: true 4th cumulant; cross-block → 0 under independence (cumulant additivity)
    assert cross_cumulant_residual_perK(X, Y, 4).norm().item() < 0.15


def test_correlated_clearly_positive_both_orders():
    X = torch.randn(_N, 2, dtype=torch.float64)
    Y = X ** 2 + 0.1 * torch.randn(_N, 2, dtype=torch.float64)
    for k in (3, 4):
        assert cross_cumulant_residual_perK(X, Y, k).norm().item() > 0.5


def test_dependence_raises_both_orders():
    Xi = torch.randn(_N, 2, dtype=torch.float64)
    Yi = torch.randn(_N, 2, dtype=torch.float64)
    Xd = torch.randn(_N, 2, dtype=torch.float64)
    Yd = Xd ** 2 + 0.1 * torch.randn(_N, 2, dtype=torch.float64)
    for k in (3, 4):
        indep = cross_cumulant_residual_perK(Xi, Yi, k).norm().item()
        dep = cross_cumulant_residual_perK(Xd, Yd, k).norm().item()
        assert dep > indep, f"k={k}: dependence should raise the cross residual"
