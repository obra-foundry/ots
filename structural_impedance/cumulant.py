"""OTS 3.0 — cumulant kernels (blueprint §1.6–§1.7; axioms §0.5, §0.5.1).

Inner numerical kernels only. Per-component κ admission; aggregate norms forbidden.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def third_central_moment(X: torch.Tensor) -> torch.Tensor:
    """K_3[i,j,k] = mean_t Xc[t,i] Xc[t,j] Xc[t,k]. X:[n,d] → [d,d,d].
    Single einsum, no point loops. Conformance: standard cumulant definition."""
    Xc = X - X.mean(dim=0, keepdim=True)
    return torch.einsum('ti,tj,tk->ijk', Xc, Xc, Xc) / Xc.shape[0]


def fourth_central_moment(X: torch.Tensor) -> torch.Tensor:
    """True 4th cumulant tensor (v3.2 patch: Edgeworth-corrected from raw 4th moment).
    κ_4[i,j,k,l] = E_4[i,j,k,l] - Σ_ij·Σ_kl - Σ_ik·Σ_jl - Σ_il·Σ_jk.
    Cumulant additivity (Sturmfels & Zwiernik arXiv:1011.1722) holds exactly:
    independent X,Y ⇒ cross-block κ_4 = 0. X:[n,d] → [d,d,d,d].
    Conformance: Sturmfels & Zwiernik arXiv:1011.1722; standard cumulant definition."""
    n = X.shape[0]
    Xc = X - X.mean(dim=0, keepdim=True)
    E4 = torch.einsum('ti,tj,tk,tl->ijkl', Xc, Xc, Xc, Xc) / n
    Sigma = torch.einsum('ti,tj->ij', Xc, Xc) / n
    return (E4
            - torch.einsum('ij,kl->ijkl', Sigma, Sigma)
            - torch.einsum('ik,jl->ijkl', Sigma, Sigma)
            - torch.einsum('il,jk->ijkl', Sigma, Sigma))


def cross_cumulant_residual_perK(X: torch.Tensor, Y: torch.Tensor,
                                 k_order: int) -> torch.Tensor:
    """κ_k ∈ ℝ^{d_x + d_y} for k ∈ {3, 4}. Zeroing the within-α and within-β
    blocks leaves the cross-block dependence signature.
    Conformance: axioms §0.5; Sturmfels & Zwiernik arXiv:1011.1722 (cumulants of
    independent vars are additive ⇒ cross-block IS the dependence signature)."""
    Z = torch.cat([X, Y], dim=1)
    if k_order == 3:
        K = third_central_moment(Z)
    elif k_order == 4:
        K = fourth_central_moment(Z)
    else:
        raise ValueError(f"k_order must be 3 or 4, got {k_order}")

    d_x = X.shape[1]
    mask = torch.ones_like(K)
    if k_order == 3:
        mask[:d_x, :d_x, :d_x] = 0.0
        mask[d_x:, d_x:, d_x:] = 0.0
    else:  # k_order == 4
        mask[:d_x, :d_x, :d_x, :d_x] = 0.0
        mask[d_x:, d_x:, d_x:, d_x:] = 0.0
    cross = K * mask
    return cross.reshape(cross.shape[0], -1).norm(dim=1)   # [d_x + d_y]


def admit_per_component(X: torch.Tensor, Y: torch.Tensor,
                        tau_3: float = 1e-3, tau_4: float = 1e-3) -> "tuple[bool, dict]":
    """ADMIT iff min_k ||κ_k|| > τ_k for EACH k ∈ {3, 4}, strict per-component.
    Aggregate composition (e.g. n3+n4>τ) is FORBIDDEN: a high κ_3 may not mask a zero κ_4.
    Conformance: axioms §0.5.1; OTS2 §VI feedback_compositional_scoring_goodhart_check."""
    k3 = cross_cumulant_residual_perK(X, Y, 3)
    k4 = cross_cumulant_residual_perK(X, Y, 4)
    n3, n4 = float(k3.norm()), float(k4.norm())
    admit = (n3 > tau_3) and (n4 > tau_4)         # AND of independent per-k tests
    return admit, {"||k3||": n3, "||k4||": n4,
                   "tau_3": tau_3, "tau_4": tau_4,
                   "blocking_component": (
                       None if admit else
                       ("k3" if n3 <= tau_3 else "k4"))}


def admit_per_component_standardized(X: torch.Tensor, Y: torch.Tensor,
                                     tau_3: float = 0.2, tau_4: float = 0.2
                                     ) -> "tuple[bool, dict]":
    """Admission gate with standardization and permutation-null-calibrated thresholds.

    Standardizes X and Y to unit variance before computing cross-cumulant residuals.
    Default τ values calibrated for z-scored data (κ in σ units, not dollar units).
    τ=0.2 means "20% of a standard deviation's worth of cross-cumulant."
    """
    X_std = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True).clamp(min=1e-8))
    Y_std = (Y - Y.mean(dim=0, keepdim=True)) / (Y.std(dim=0, keepdim=True).clamp(min=1e-8))
    return admit_per_component(X_std, Y_std, tau_3=tau_3, tau_4=tau_4)


def cumulant_difference(X: torch.Tensor, Y: torch.Tensor,
                        k_order: int) -> torch.Tensor:
    """Per-order cumulant DIFFERENCE between two independent populations.

    Computes the cumulant of X, the cumulant of Y, and returns their difference.
    This is the correct primitive for "do these two distributions differ in their
    higher-order structure?" — unlike cross_cumulant_residual_perK, which detects
    within-unit dependence between co-observed feature blocks.

    Args:
        X: [n, d] — sample from population A
        Y: [m, d] — sample from population B (n and m may differ)
        k_order: 3 or 4

    Returns:
        diff_norm: scalar — Frobenius norm of the cumulant difference tensor
    """
    if k_order == 3:
        k_X = third_central_moment(X)
        k_Y = third_central_moment(Y)
    elif k_order == 4:
        k_X = fourth_central_moment(X)
        k_Y = fourth_central_moment(Y)
    else:
        raise ValueError(f"k_order must be 3 or 4, got {k_order}")

    diff = k_X - k_Y
    return diff.norm()
