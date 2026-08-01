"""§5.bis: each public kernel docstring carries its conformance citation anchor."""

from structural_impedance import gamma_correction as gc
from structural_impedance import cumulant as cm
from structural_impedance import lmoment as lm
from structural_impedance import sinkhorn as sk
from structural_impedance import sheaf_gluing as sg

EXPECTED = {
    lm.sample_lmoments:               "Hosking",
    lm.aggregation_contrast_test:     "Hosking",
    gc.reflection_coefficient:        "Pozar",
    gc.epsilon_gamma:                 "Feydy",
    gc.metric_warp_factor:            "Chazal",
    gc.phi_modulation:                "Pozar",
    gc.gamma_eff_from_kurtosis:       "§0.6",
    gc.alpha_modulation:              "§0.6",
    cm.third_central_moment:          "cumulant",
    cm.fourth_central_moment:         "cumulant",
    cm.cross_cumulant_residual_perK:  "Zwiernik",
    cm.admit_per_component:           "§0.5.1",
    sk.sinkhorn_potentials:           "Feydy",
    sk.sim_hessian_torch_hvp:         "Shen",
    sk.kappa_sinkhorn_per_component:  "Zwiernik",
    sg.cocycle_disagreement:          "Curry",
    sg.sheaf_status_and_kappa:        "Curry",
}


def test_every_kernel_has_citation_anchor():
    for fn, anchor in EXPECTED.items():
        doc = fn.__doc__ or ""
        assert "Conformance" in doc, f"{fn.__name__} missing Conformance line"
        assert anchor in doc, f"{fn.__name__} docstring missing anchor '{anchor}'"
