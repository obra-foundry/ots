"""Blueprint §7 item 1: NO `1 - Γ²` anywhere in structural_impedance/; every Φ denominator uses |Γ|²."""

import pathlib

OTS3 = pathlib.Path(__file__).resolve().parents[2] / "structural_impedance"

FORBIDDEN = [
    "1 - Gamma**2",
    "1 - Gamma ** 2",
    "1 - Gamma.pow(2)",
    "1.0 - Gamma**2",
    "1.0 - Gamma ** 2",
    "1.0 - Gamma.pow(2)",
    "1 - Γ²",   # 1 - Γ²
]


def test_no_complex_squared_denominator():
    for path in OTS3.glob("*.py"):
        src = path.read_text()
        for bad in FORBIDDEN:
            assert bad not in src, f"forbidden complex-squared denom '{bad}' in {path.name}"


def test_phi_denominator_uses_abs_gamma():
    src = (OTS3 / "gamma_correction.py").read_text()
    # the phi_modulation denominator line must compute 1 - |Γ|²
    assert "1.0 - float(abs(Gamma)) ** 2" in src
    assert "ALWAYS |Γ|², never Γ²" in src
