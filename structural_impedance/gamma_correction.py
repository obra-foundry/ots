"""OTS 3.0 — gamma_correction kernels (blueprint §1.1–§1.5; axioms §0.2, §0.6, §1.1).

Inner numerical kernels only. No class wrapping, no I/O, no exception raising,
no validation_status evaluation. Honest-flag logging via stdlib logging only.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def reflection_coefficient(Z_emit: torch.Tensor, Z_recv: torch.Tensor) -> torch.Tensor:
    """Γ = (Z_recv - Z_emit) / (Z_recv + Z_emit). Batched over any shape.
    Conformance: Pozar 4e Ch. 2."""
    return (Z_recv - Z_emit) / (Z_recv + Z_emit)


def epsilon_gamma(Gamma: complex, gamma_star: "complex | None",
                  eps_0: float = 5e-2, lam: float = 1.0, tau: float = 1e-6) -> float:
    """ε(Γ,γ*) = ε₀·exp(λ·|Γ-γ*|²); natural ε₀ when γ* is None.
    Conformance: Feydy et al. AISTATS 2019 (ε > 0 sufficient for all properties)."""
    if gamma_star is None:
        return max(eps_0, tau)
    eps = eps_0 * float(np.exp(lam * abs(Gamma - gamma_star) ** 2))
    return max(eps, tau)


def metric_warp_factor(Gamma: complex, gamma_star: "complex | None"
                       ) -> "tuple[float, str | None]":
    """Returns (warp_factor, obstruction_tag).
    Conformance: scalar rescaling of metric, preserves bound (Chazal et al. arXiv:1207.3885);
    (Γ+γ*) singularity guard per axioms §0.2."""
    if gamma_star is None:
        return 1.0, None
    denom = abs(Gamma + gamma_star)
    if denom <= 1e-12:
        tag = "metric_warp:gamma_plus_gamma_star_singular"
        logger.warning("metric_warp_factor: |Γ+γ*|<=1e-12; natural metric, obstruction=%s", tag)
        return 1.0, tag
    return 1.0 + float(abs((Gamma - gamma_star) / (Gamma + gamma_star))), None


def phi_modulation(Gamma: complex, gamma_star: "complex | None") -> float:
    """Φ = 1 - exp(-|Γ - γ*|² / (1 - |Γ|²)). Real-valued in [0,1) for |Γ| < 1.
    Conformance: Pozar 4e Ch. 2 (1-|Γ|² is the power-transmission coefficient)."""
    if gamma_star is None:
        return 0.0
    denom = 1.0 - float(abs(Gamma)) ** 2          # ALWAYS |Γ|², never Γ²
    if denom <= 1e-12:
        logger.warning(
            "phi_modulation: total reflection regime (1-|Γ|²<=1e-12); returning Φ=1.0")
        return 1.0
    return 1.0 - float(np.exp(-abs(Gamma - gamma_star) ** 2 / denom))


def gamma_eff_from_kurtosis(kappa_2: float, kappa_4: float, Gamma_0: complex,
                            R_crit: float = 4.458, R_scale: float = 2.094) -> complex:
    """Γ_eff(κ_local) = Γ₀·tanh((R - R_crit)/R_scale), R = κ₄/max(κ₂², 1e-12).
    Conformance: axioms §0.6; FAF-W3 OTS2 canonical anchors
    (R_crit_A=4.458, R_crit_B=6.100, R_scale=2.094, LOCKED CANONICAL)."""
    R = kappa_4 / max(kappa_2 ** 2, 1e-12)
    scale = float(np.tanh((R - R_crit) / R_scale))
    return Gamma_0 * scale


def alpha_modulation(kappa_2: float, kappa_4: float, Gamma_0: complex,
                     gamma_star: "complex | None",
                     R_crit: float = 4.458, R_scale: float = 2.094) -> float:
    """α(κ_local) = Φ(Γ_eff(κ_local), γ*). Composes gamma_eff_from_kurtosis then phi_modulation.
    Conformance: axioms §0.6 Φ↔α coupling (CncKernelMonitorAgent surface)."""
    Gamma_eff = gamma_eff_from_kurtosis(kappa_2, kappa_4, Gamma_0, R_crit, R_scale)
    return phi_modulation(Gamma_eff, gamma_star)
