"""CODE-U §3 — the three automated checks flag bad prose, pass good prose."""

from structural_impedance.transmission_diagnostic import (
    audit_transmission,
    detect_moral_contamination,
    detect_over_smoothing,
    detect_signal_bleed,
)


CLEAN_STRIPPED = (
    "On the heart-to-thorax width ratio, two image-quality strata "
    "(n=4,500 each) report the same mean of 0.49 while the patient-by-"
    "patient spread is 73% wider on the difficult-to-segment images. "
    "Standard mean and variance comparison cannot detect this. The "
    "discriminative signal lives at higher distributional moments. The "
    "instrument names the failing subgroup and reports a noise-floor ratio."
)

THEOLOGY = "The covenant between God and the people preserves signal integrity; grace ensures lossless transfer."
MORAL = "The model exhibits morally wrong behavior; it is evil to deploy something this sinful."
SMOOTHED = (
    "Our revolutionary, world-class, paradigm-shifting, unprecedented, "
    "cutting-edge breakthrough delivers amazing, incredible, game-changing "
    "insights with next-generation power."
)


def test_clean_stripped_passes():
    assert audit_transmission(CLEAN_STRIPPED, encoding="stripped") == []


def test_theology_triggers_signal_bleed():
    modes = {f.mode for f in detect_signal_bleed(THEOLOGY)}
    assert "signal_bleed" in modes


def test_moral_prose_triggers_contamination():
    modes = {f.mode for f in detect_moral_contamination(MORAL)}
    assert "moral_contamination" in modes


def test_marketing_prose_triggers_over_smoothing():
    modes = {f.mode for f in detect_over_smoothing(SMOOTHED)}
    assert "over_smoothing" in modes


def test_overt_encoding_allows_theology():
    findings = audit_transmission(
        "The covenant preserves Z_0 binding integrity through galvanization.",
        encoding="overt")
    assert not any(f.mode == "signal_bleed" for f in findings)


def test_corrections_are_present():
    for f in detect_moral_contamination(MORAL):
        assert len(f.correction) > 10
