"""CODE-U Axiom 4 — every public kernel carries a provenance back-pointer."""

import structural_impedance as si
from structural_impedance.provenance import PROVENANCE, provenance_for


_NUMERICAL_KERNELS = [
    si.admit_per_component,
    si.admit_per_component_standardized,
    si.cross_cumulant_residual_perK,
    si.cumulant_difference,
    si.third_central_moment,
    si.fourth_central_moment,
    si.cocycle_disagreement,
    si.sheaf_status_and_kappa,
]


def test_every_numerical_kernel_has_provenance():
    for fn in _NUMERICAL_KERNELS:
        rec = provenance_for(fn)
        assert rec.fn_name == fn.__name__
        assert rec.secular_anchor
        assert rec.vault_path.startswith("Æ")
        assert rec.derivation_chain_id
        assert rec.axiom_statement
        assert rec.origin_anchor


def test_provenance_chain_emits_five_lines():
    from structural_impedance.provenance import derivation_chain
    lines = derivation_chain(si.cumulant_difference)
    assert len(lines) == 5
    for tag in ("Origin", "Vault", "Axiom", "Tier", "Secular"):
        assert any(tag in ln for ln in lines)


def test_missing_kernel_raises():
    def fake_kernel():
        pass

    try:
        provenance_for(fake_kernel)
    except KeyError as exc:
        assert "Axiom 4" in str(exc)
    else:
        raise AssertionError("provenance_for must raise on unknown kernel")


def test_sinkhorn_kernels_also_registered():
    from structural_impedance import sinkhorn as sk

    for fn in (sk.sinkhorn_potentials, sk.sim_hessian_torch_hvp,
               sk.kappa_sinkhorn_per_component):
        rec = provenance_for(fn)
        assert rec.fn_name == fn.__name__


def test_gamma_kernels_also_registered():
    from structural_impedance import gamma_correction as gc

    for fn in (gc.reflection_coefficient, gc.epsilon_gamma,
               gc.metric_warp_factor, gc.phi_modulation,
               gc.gamma_eff_from_kurtosis, gc.alpha_modulation):
        rec = provenance_for(fn)
        assert rec.fn_name == fn.__name__


def test_registry_has_no_orphans():
    # Every entry in PROVENANCE points to a real callable.
    for fn in PROVENANCE.keys():
        assert callable(fn)
