"""Binary gluing: one bad overlap among many ⇒ obstructed with correct tag."""

import torch

from structural_impedance.sheaf_gluing import sheaf_status_and_kappa


def test_single_violation_obstructs():
    sections = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [5.0, 5.0],   # section 3 disagrees badly
    ], dtype=torch.float64)
    overlaps = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)

    status, residuals, tag = sheaf_status_and_kappa(sections, overlaps, tol=1e-6)
    assert status == "obstructed"
    assert tag == "sheaf:overlap_2_3"
    assert len(residuals) == overlaps.shape[0]


def test_all_agree_is_valid():
    sections = torch.zeros(4, 2, dtype=torch.float64)
    overlaps = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    status, residuals, tag = sheaf_status_and_kappa(sections, overlaps, tol=1e-6)
    assert status == "valid" and residuals == [] and tag is None
