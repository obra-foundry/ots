"""CODE-U §4 — round-trip reclamation matches original within tolerance."""

import torch

from structural_impedance.cumulant import (
    admit_per_component_standardized,
    cumulant_difference,
)
from structural_impedance.findings import (
    AdmissionFinding,
    CumulantFinding,
    SheafFinding,
)
from structural_impedance.reclamation import reclamation_test
from structural_impedance.sheaf_gluing import cocycle_disagreement


torch.manual_seed(0)


def _populations(n_x=400, n_y=600, shift=0.7):
    X = torch.randn(n_x, 1, dtype=torch.float64) * 1.2
    Y = torch.randn(n_y, 1, dtype=torch.float64) + shift
    return X, Y


def test_cumulant_reclamation_matches_on_same_data():
    X, Y = _populations()
    val = float(cumulant_difference(X, Y, k_order=3))
    finding = CumulantFinding(
        contrast_name="A vs B",
        feature_name="z",
        k_order=3,
        n_x=X.shape[0], n_y=Y.shape[0],
        real_value=val,
        null_value=val / 5.0,  # plausible null reference
        gate_admit=False,
        blocking_component="k3",
    )
    res = reclamation_test(finding, (X, Y))
    assert res.matched, f"reclamation should match on same data: {res}"
    assert res.relative_error < 1e-6


def test_cumulant_reclamation_fails_on_different_data():
    X, Y = _populations()
    val = float(cumulant_difference(X, Y, k_order=3))
    finding = CumulantFinding(
        contrast_name="A vs B",
        feature_name="z",
        k_order=3,
        n_x=X.shape[0], n_y=Y.shape[0],
        real_value=val,
        null_value=val / 5.0,
        gate_admit=False,
        blocking_component="k3",
    )
    X2 = torch.randn(X.shape[0], 1, dtype=torch.float64)
    Y2 = torch.randn(Y.shape[0], 1, dtype=torch.float64)
    res = reclamation_test(finding, (X2, Y2), tol=0.05)
    assert not res.matched, "reclamation should fail on unrelated data"


def test_sheaf_reclamation_matches_on_same_sections():
    sections = torch.tensor(
        [[209.0, 60.4, -6.98, 8.03],
         [184.2, 39.7, -5.95, 10.86],
         [194.7, 49.0, -5.69, 9.37]],
        dtype=torch.float64,
    )
    overlaps = ((0, 2), (1, 2))
    overlaps_t = torch.tensor(list(overlaps), dtype=torch.long)
    se = cocycle_disagreement(sections, overlaps_t).tolist()
    finding = SheafFinding(
        contrast_name="disease",
        feature_name="lung_asym",
        section_labels=("Cardiomegaly", "Normal", "All"),
        overlaps=overlaps,
        per_overlap_se=tuple(se),
        tol_se=8.0,
        null_max_se=1.9,
        status="obstructed",
        obstruction_at="sheaf:overlap_0_2",
    )
    res = reclamation_test(finding, sections)
    assert res.matched, f"sheaf reclamation should match: {res}"


def test_sheaf_reclamation_fails_on_perturbed_sections():
    sections = torch.zeros(3, 4, dtype=torch.float64)
    overlaps = ((0, 2), (1, 2))
    overlaps_t = torch.tensor(list(overlaps), dtype=torch.long)
    se = cocycle_disagreement(sections, overlaps_t).tolist()
    finding = SheafFinding(
        contrast_name="x",
        feature_name="y",
        section_labels=("A", "B", "All"),
        overlaps=overlaps,
        per_overlap_se=tuple(se),
        tol_se=8.0, null_max_se=1.0,
        status="valid",
        obstruction_at=None,
    )
    # Perturb section 0 by a large amount
    sections_pert = sections.clone()
    sections_pert[0] = torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float64)
    res = reclamation_test(finding, sections_pert, tol=0.02)
    assert not res.matched, "sheaf reclamation should fail on perturbed sections"


def test_admission_reclamation_matches():
    # Admission gate composes cross_cumulant_residual_perK which requires
    # equal sample sizes (cat along dim=1). Use equalized populations.
    X, Y = _populations(n_x=500, n_y=500)
    admit, meta = admit_per_component_standardized(X, Y)
    finding = AdmissionFinding(
        contrast_name="A vs B",
        feature_name="z",
        n_x=X.shape[0], n_y=Y.shape[0],
        k3_norm=float(meta["||k3||"]),
        k4_norm=float(meta["||k4||"]),
        tau_3=0.2, tau_4=0.2,
        admit=admit,
        blocking_component=meta.get("blocking_component"),
    )
    res = reclamation_test(finding, (X, Y))
    assert res.matched, f"admission reclamation should match: {res}"


def test_dispatch_raises_on_unknown_finding_type():
    class FakeFinding:
        pass
    try:
        reclamation_test(FakeFinding(), None)
    except TypeError as exc:
        assert "Reclamation not defined" in str(exc)
    else:
        raise AssertionError("must raise TypeError on unknown finding")
