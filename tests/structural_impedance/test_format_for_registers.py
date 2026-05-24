"""CODE-U Step 3 — every Finding emits in overt + stripped registers."""

from structural_impedance.findings import (
    AdmissionFinding,
    CumulantFinding,
    ICRFinding,
    SheafFinding,
)


def _cf():
    return CumulantFinding(
        contrast_name="Subgroup A vs Subgroup B",
        feature_name="heart-to-thorax width ratio",
        k_order=3, n_x=4500, n_y=4500,
        real_value=2.89e-4, null_value=3.61e-5,
        gate_admit=False, blocking_component="k3")


def _sf():
    return SheafFinding(
        contrast_name="disease subgroups",
        feature_name="lung area asymmetry",
        section_labels=("Cardiomegaly", "Normal", "All"),
        overlaps=((0, 2), (1, 2)),
        per_overlap_se=(18.46, 14.16),
        tol_se=8.0, null_max_se=1.90,
        status="obstructed", obstruction_at="sheaf:overlap_0_2")


def _af():
    return AdmissionFinding(
        contrast_name="HighDice vs LowDice", feature_name="CTR",
        n_x=4500, n_y=4500, k3_norm=0.31, k4_norm=2.10,
        tau_3=0.2, tau_4=0.2, admit=True, blocking_component=None)


def _icr():
    return ICRFinding(
        contrast_name="quality strata",
        interface_name="mask uncertainty (1 - Dice)",
        subgroup_a_label="LowDice", subgroup_b_label="HighDice",
        value_a=0.1820, value_b=0.1184)


def test_cumulant_two_registers():
    f = _cf()
    overt, stripped = f.format_for("overt"), f.format_for("stripped")
    assert "kappa" in overt.lower() or "delta" in overt.lower()
    for token in ("kappa", "cumulant", "sheaf", "cocycle"):
        assert token.lower() not in stripped.lower(), \
            f"stripped leaked '{token}'"


def test_sheaf_two_registers():
    f = _sf()
    overt, stripped = f.format_for("overt"), f.format_for("stripped")
    assert "obstruct" in overt.lower()
    assert "sheaf" not in stripped.lower()
    assert "cocycle" not in stripped.lower()


def test_admission_two_registers():
    f = _af()
    for audience in ("overt", "stripped"):
        assert len(f.format_for(audience)) > 20
    assert "admit" in f.format_for("overt").lower()


def test_icr_two_registers():
    f = _icr()
    for audience in ("overt", "stripped"):
        assert len(f.format_for(audience)) > 20
    assert "%" in f.format_for("overt") or "x" in f.format_for("overt")


def test_chain_id_in_overt():
    assert "ADT-v1.0" in _cf().format_for("overt")
