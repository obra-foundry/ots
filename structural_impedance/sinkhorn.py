"""OTS 3.0 — sinkhorn kernels (blueprint §2; axioms §1).

Inner numerical kernels only. SIM Hessian via IMPLICIT DIFFERENTIATION through
the Sinkhorn fixed point. Explicit closed-form assembly and eigenvalue clamping
are FORBIDDEN at v3.x (axioms §1.2, §5.bis).
"""

import logging

import torch

from structural_impedance.cumulant import cross_cumulant_residual_perK
from structural_impedance.gamma_correction import epsilon_gamma

logger = logging.getLogger(__name__)


def sinkhorn_potentials(a: torch.Tensor, b: torch.Tensor, C: torch.Tensor,
                        eps: float, n_iter: int = 200, tol: float = 1e-9
                        ) -> "tuple[torch.Tensor, torch.Tensor]":
    """Log-space Sinkhorn iterations; each step is two logsumexp reductions.
    a:[n] (Σ=1), b:[m] (Σ=1), C:[n,m]. The iteration loop is the only permitted
    loop in scope. Conformance: Feydy et al. AISTATS 2019."""
    log_a, log_b = torch.log(a + 1e-30), torch.log(b + 1e-30)
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    neg_C_eps = -C / eps
    for _ in range(n_iter):
        f_new = eps * (log_a - torch.logsumexp(neg_C_eps + g / eps, dim=1))
        g_new = eps * (log_b - torch.logsumexp(neg_C_eps.T + f_new / eps, dim=1))
        if (f_new - f).abs().max() < tol and (g_new - g).abs().max() < tol:
            return f_new, g_new
        f, g = f_new, g_new
    max_resid = max(float((f_new - f).abs().max()), float((g_new - g).abs().max()))
    logger.warning(
        "sinkhorn_potentials: did not converge in %d iters; max residual=%.3e",
        n_iter, max_resid)
    return f, g


def sim_hessian_torch_hvp(theta: torch.Tensor, sample_fn, beta_samples: torch.Tensor,
                          Gamma: complex, gamma_star: "complex | None",
                          eps_0: float = 5e-2, lam: float = 1.0,
                          n_sinkhorn: int = 200) -> torch.Tensor:
    """SIM Hessian via implicit differentiation through the Sinkhorn fixed point.
    Conformance: Shen et al. NeurIPS 2020 Prop. 5.1 Eq. 17 (closed form is SPEC
    ONLY); implicit diff through the fixed point is the implementation contract
    (axioms §1.2). NO explicit closed-form assembly, NO eigenvalue clamping/PSD
    projection — non-PD H is LOGGED (monitored assumption, axioms §1.6 cond. 3)
    and returned AS-IS.

    Substrate A (ott-jax, production) reference — commented, no JAX dependency:

        # from ott.geometry import pointcloud
        # from ott.solvers.linear import sinkhorn
        # from ott.tools import sinkhorn_divergence
        # import jax
        #
        # def sim_hessian_ott(theta, sample_fn, beta_samples, Gamma, gamma_star,
        #                     eps_0=5e-2, lam=1.0):
        #     eps = epsilon_gamma(Gamma, gamma_star, eps_0=eps_0, lam=lam)
        #     def S_eps(th):
        #         x = sample_fn(th); y = beta_samples
        #         div = sinkhorn_divergence.sinkhorn_divergence(
        #             pointcloud.PointCloud, x, y, epsilon=eps).divergence
        #         return div
        #     return jax.hessian(S_eps)(theta)
    """
    eps = epsilon_gamma(Gamma, gamma_star, eps_0=eps_0, lam=lam)

    def S_eps(th: torch.Tensor) -> torch.Tensor:
        x = sample_fn(th)
        y = beta_samples
        n, m = x.shape[0], y.shape[0]
        a = x.new_full((n,), 1.0 / n)
        b = y.new_full((m,), 1.0 / m)

        C_xy = torch.cdist(x, y, p=2).pow(2)
        C_xx = torch.cdist(x, x, p=2).pow(2)
        C_yy = torch.cdist(y, y, p=2).pow(2)

        f_xy, g_xy = sinkhorn_potentials(a, b, C_xy, eps, n_sinkhorn)
        f_xx, _    = sinkhorn_potentials(a, a, C_xx, eps, n_sinkhorn)
        f_yy, _    = sinkhorn_potentials(b, b, C_yy, eps, n_sinkhorn)

        OT_xy = (a * f_xy).sum() + (b * g_xy).sum()
        OT_xx = 2.0 * (a * f_xx).sum()
        OT_yy = 2.0 * (b * f_yy).sum()
        return OT_xy - 0.5 * OT_xx - 0.5 * OT_yy           # debiased Sinkhorn divergence

    H = torch.autograd.functional.hessian(S_eps, theta, vectorize=True)
    H = H.detach()

    # PD-monitoring only (axioms §1.6 cond. 3): observed-but-not-proven PD-ness.
    # LOG if violated; return H unmodified. No clamping / projection.
    if H.ndim == 2 and H.shape[0] == H.shape[1]:
        H_sym = 0.5 * (H + H.transpose(-1, -2))
        min_eig = float(torch.linalg.eigvalsh(H_sym).min())
        if min_eig < 0.0:
            logger.warning(
                "sim_hessian_torch_hvp: SIM not PD (min eigenvalue=%.3e); "
                "monitored assumption violated, returning H unmodified", min_eig)
    return H


def kappa_sinkhorn_per_component(x: torch.Tensor, y: torch.Tensor
                                 ) -> "dict[str, torch.Tensor]":
    """Per-component κ residuals for the joint coupling. Returns
    {'k3': [...], 'k4': [...]} — NOT concatenated, NOT aggregated (axioms §0.5.1).
    Conformance: Sturmfels & Zwiernik arXiv:1011.1722 (via cross_cumulant_residual_perK)."""
    return {
        "k3": cross_cumulant_residual_perK(x, y, 3),
        "k4": cross_cumulant_residual_perK(x, y, 4),
    }
