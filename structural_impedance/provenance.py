"""CODE-U Axiom 4: Provenance Irreversibility.

Every public kernel carries a back-pointer to its derivation chain so the
decoupling from the upstream substrate is lossless. Vault paths are
operator-resolvable. Axiom statements inlined so records are meaningful
without vault access.
Conformance: 1164.CODE-U.1.0 Axiom 4; vault Æ24-0905/Eccessories.
"""
from dataclasses import dataclass
from typing import Callable

from structural_impedance import cumulant as _cm
from structural_impedance import gamma_correction as _gc
from structural_impedance import sheaf_gluing as _sg
from structural_impedance import sinkhorn as _sk


@dataclass(frozen=True)
class ProvenanceRecord:
    fn_name: str
    module: str
    secular_anchor: str
    cathedral_tier: int
    vault_path: str
    derivation_chain_id: str
    axiom_statement: str
    origin_anchor: str


def _r(fn_name, module, secular, tier, vault, chain, axiom, origin):
    return ProvenanceRecord(fn_name, module, secular, tier, vault, chain, axiom, origin)


_GC = "structural_impedance.gamma_correction"
_CM = "structural_impedance.cumulant"
_SG = "structural_impedance.sheaf_gluing"
_SK = "structural_impedance.sinkhorn"
_SIGDY = "Æ24-0905/Eccessories/M26.Z3.SigDy3.4.md"
_ADT = "Æ24-0905/Z_C2/C1.ADT1.0.md"
_CGF = "Æ24-0905/Eccessories/M27.Z3.CGF1.3.md"
_FAF = "Æ24-0905/Z_C2/C11.FAF-spec.v0.3.1146.md"
_DOUBT = "Æ15-0709 'acknowledgment of doubt is a prerequisite to taking a chance'"
_LANG = "Æ15-0709 'language is mathematics'"
_DIVERGE = "Æ15-0709 'patterns of divergence can drive improvements'"
_DECODE = "Æ15-0709 'a decoder can assess information and apply a bias filter or honesty filter'"
_LIES = "Æ15-0709 'the cost of lies is about to rise precipitously'"

PROVENANCE: "dict[Callable, ProvenanceRecord]" = {
    _gc.reflection_coefficient: _r(
        "reflection_coefficient", _GC, "Pozar, Microwave Engineering 4e Ch. 2", 2,
        _SIGDY, "SigDy-v3.4::governing-equation",
        "Gamma = (Z_recv - Z_emit) / (Z_recv + Z_emit) governs signal transfer integrity.",
        "Æ15-0709 'context can be more important than content'"),
    _gc.epsilon_gamma: _r(
        "epsilon_gamma", _GC, "Feydy et al. AISTATS 2019", 2,
        _SIGDY, "SigDy-v3.4::impedance-gated-regularization",
        "Entropic regularization scales with deviation from target impedance.", _DOUBT),
    _gc.metric_warp_factor: _r(
        "metric_warp_factor", _GC, "Chazal et al. arXiv:1207.3885", 2,
        _SIGDY, "SigDy-v3.4::singularity-guard",
        "Divide-by-zero at |Gamma + gamma_star| ~ 0 is tagged, not allowed to diverge.", _DOUBT),
    _gc.phi_modulation: _r(
        "phi_modulation", _GC, "Pozar, Microwave Engineering 4e Ch. 2", 2,
        _SIGDY, "SigDy-v3.4::real-power-coefficient",
        "Phi denominator is 1 - |Gamma|^2 (real), never 1 - Gamma^2 (complex).", _LANG),
    _gc.gamma_eff_from_kurtosis: _r(
        "gamma_eff_from_kurtosis", _GC, "vault CLAUDE.md §0.6; FAF-W3 anchors", 2,
        _FAF, "SigDy-v3.4::kurtosis-coupling",
        "Effective Gamma is a tanh of kappa_4 / kappa_2^2 centered at R_crit=4.458.", _DIVERGE),
    _gc.alpha_modulation: _r(
        "alpha_modulation", _GC, "vault CLAUDE.md §0.6 Phi-alpha coupling", 2,
        _SIGDY, "SigDy-v3.4::phi-alpha-coupling",
        "alpha(kappa) = Phi(Gamma_eff(kappa), gamma_star). Composition.", _LANG),
    _cm.third_central_moment: _r(
        "third_central_moment", _CM, "standard cumulant; Sturmfels-Zwiernik arXiv:1011.1722", 4,
        _ADT, "ADT-v1.0::third-order-tensor",
        "K_3[i,j,k] = mean_t Xc[t,i] Xc[t,j] Xc[t,k]. Single einsum, no aggregation.", _DIVERGE),
    _cm.fourth_central_moment: _r(
        "fourth_central_moment", _CM, "Sturmfels & Zwiernik arXiv:1011.1722", 4,
        _ADT, "ADT-v1.0::edgeworth-corrected-kappa4",
        "True 4th cumulant with Edgeworth correction. Independent X,Y => cross-block = 0.", _LANG),
    _cm.cross_cumulant_residual_perK: _r(
        "cross_cumulant_residual_perK", _CM, "Sturmfels & Zwiernik arXiv:1011.1722", 4,
        _ADT, "ADT-v1.0::cross-block-dependence-signature",
        "Per-component cross-block Frobenius norm. Returns vector, never aggregated.", _DECODE),
    _cm.admit_per_component: _r(
        "admit_per_component", _CM, "vault CLAUDE.md §0.5.1 anti-aggregation", 4,
        _ADT, "ADT-v1.0::per-component-AND-gate",
        "ADMIT iff min_k ||kappa_k|| > tau_k for EACH k. Aggregate composition forbidden.", _LIES),
    _cm.admit_per_component_standardized: _r(
        "admit_per_component_standardized", _CM, "vault CLAUDE.md §0.5.1", 4,
        _ADT, "ADT-v1.0::sigma-unit-noise-floor",
        "Z-scored per-component gate. Sigma-unit thresholds. Noise-floor refusal.", _DOUBT),
    _cm.cumulant_difference: _r(
        "cumulant_difference", _CM, "Sturmfels & Zwiernik arXiv:1011.1722", 4,
        _ADT, "ADT-v1.0::marginal-cumulant-divergence",
        "Frobenius norm of marginal kappa_k difference. No equal-n requirement.", _DECODE),
    _sg.cocycle_disagreement: _r(
        "cocycle_disagreement", _SG, "Curry, Sheaves Cosheaves and Applications 2014", 4,
        _ADT, "ADT-v1.0::composite-not-scalar",
        "Pairwise restriction disagreement over overlaps. Returns vector.", _DIVERGE),
    _sg.sheaf_status_and_kappa: _r(
        "sheaf_status_and_kappa", _SG, "Curry, Sheaves Cosheaves and Applications 2014", 4,
        _ADT, "ADT-v1.0::binary-gluing",
        "Single overlap violation = obstructed. No averaging, softening, voting.",
        "Æ15-0709 'Matt 5:37 yes-yes no-no'"),
    _sk.sinkhorn_potentials: _r(
        "sinkhorn_potentials", _SK, "Feydy et al. AISTATS 2019", 3,
        _CGF, "CGF-v1.3::log-space-iteration",
        "Log-space Sinkhorn. Two logsumexp per step. Only permitted loop in scope.", _LANG),
    _sk.sim_hessian_torch_hvp: _r(
        "sim_hessian_torch_hvp", _SK, "Shen et al. NeurIPS 2020 Prop. 5.1 Eq. 17", 3,
        _FAF, "FAF-v0.4::implicit-diff-SIM",
        "SIM Hessian via implicit diff. No closed-form, no PSD clamping. Non-PD logged.", _DOUBT),
    _sk.kappa_sinkhorn_per_component: _r(
        "kappa_sinkhorn_per_component", _SK, "Sturmfels & Zwiernik arXiv:1011.1722", 4,
        _ADT, "ADT-v1.0::sinkhorn-per-component",
        "Per-component kappa residuals for joint coupling. Dict, never concatenated.", _DIVERGE),
}


def provenance_for(fn: Callable) -> ProvenanceRecord:
    if fn not in PROVENANCE:
        raise KeyError(
            f"No provenance record for {fn}. CODE-U Axiom 4 requires every "
            f"public kernel to carry a back-pointer."
        )
    return PROVENANCE[fn]


def derivation_chain(fn: Callable) -> "list[str]":
    p = provenance_for(fn)
    return [
        f"Origin:    {p.origin_anchor}",
        f"Vault:     {p.vault_path}  ({p.derivation_chain_id})",
        f"Axiom:     {p.axiom_statement}",
        f"Tier:      {p.cathedral_tier} (vault CLAUDE.md §6)",
        f"Secular:   {p.secular_anchor}",
    ]
