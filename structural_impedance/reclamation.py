"""CODE-U §4 Reclamation Test (round-trip integrity).

Given an emitted finding and a candidate substrate, recompute the finding on
the candidate and verify it reproduces the original within tolerance. Strictly
stronger than the null-permutation check.

Conformance: 1164.CODE-U.1.0 §4; CLAUDE.md §0.5.1.
"""
from dataclasses import dataclass

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
from structural_impedance.sheaf_gluing import cocycle_disagreement


@dataclass(frozen=True)
class ReclamationResult:
    matched: bool
    original_value: float
    reclaimed_value: float
    relative_error: float
    tol: float
    notes: str = ""


def _err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(1) if t.ndim == 1 else t


def _reclaim_cumulant(f: CumulantFinding, X, Y, tol):
    if X.shape[0] != f.n_x or Y.shape[0] != f.n_y:
        return ReclamationResult(
            False, f.real_value, float("nan"), float("nan"), tol,
            f"sample sizes ({X.shape[0]},{Y.shape[0]}) != ({f.n_x},{f.n_y})")
    re = float(cumulant_difference(_as_2d(X), _as_2d(Y), k_order=f.k_order))
    err = _err(re, f.real_value)
    return ReclamationResult(
        err <= tol, f.real_value, re, err, tol,
        f"k={f.k_order} contrast='{f.contrast_name}' feature='{f.feature_name}'")


def _reclaim_sheaf(f: SheafFinding, sections, tol):
    if sections.shape[0] != len(f.section_labels):
        return ReclamationResult(
            False, sum(f.per_overlap_se), float("nan"), float("nan"), tol,
            f"sections have {sections.shape[0]} rows, expected {len(f.section_labels)}")
    overlaps_t = torch.tensor(list(f.overlaps), dtype=torch.long)
    re_vec = cocycle_disagreement(sections, overlaps_t).tolist()
    errs = [_err(a, b) for a, b in zip(re_vec, f.per_overlap_se)]
    worst = max(errs)
    return ReclamationResult(
        worst <= tol, sum(f.per_overlap_se), sum(re_vec), worst, tol,
        f"per-overlap errs={[round(e,4) for e in errs]} status='{f.status}'")


def _reclaim_admission(f: AdmissionFinding, X, Y, tol):
    admit, meta = admit_per_component_standardized(
        _as_2d(X), _as_2d(Y), tau_3=f.tau_3, tau_4=f.tau_4)
    k3_err = _err(meta["||k3||"], f.k3_norm)
    k4_err = _err(meta["||k4||"], f.k4_norm)
    matched = (admit == f.admit
               and meta.get("blocking_component") == f.blocking_component
               and max(k3_err, k4_err) <= tol)
    return ReclamationResult(
        matched, f.k3_norm + f.k4_norm,
        float(meta["||k3||"]) + float(meta["||k4||"]),
        max(k3_err, k4_err), tol,
        f"verdict match={admit == f.admit}, block match="
        f"{meta.get('blocking_component') == f.blocking_component}")


_DISPATCH = {
    CumulantFinding: (_reclaim_cumulant, 0.05, True),    # candidate=(X,Y)
    SheafFinding:    (_reclaim_sheaf,    0.02, False),   # candidate=sections
    AdmissionFinding:(_reclaim_admission,0.05, True),    # candidate=(X,Y)
}


def reclamation_test(finding, candidate, *, tol: float = None) -> ReclamationResult:
    """Dispatch reclamation by finding type. candidate is (X,Y) for
    Cumulant/Admission findings, a sections tensor for Sheaf findings."""
    if type(finding) not in _DISPATCH:
        raise TypeError(
            f"Reclamation not defined for {type(finding).__name__}. "
            f"CODE-U §4 requires round-trip for every emitted finding type.")
    fn, default_tol, is_pair = _DISPATCH[type(finding)]
    t = tol if tol is not None else default_tol
    if is_pair:
        return fn(finding, candidate[0], candidate[1], t)
    return fn(finding, candidate, t)


# Per-type wrappers retained for direct invocation
def reclaim_cumulant(f, X, Y, tol: float = 0.05):
    return _reclaim_cumulant(f, X, Y, tol)


def reclaim_sheaf(f, sections, tol: float = 0.02):
    return _reclaim_sheaf(f, sections, tol)


def reclaim_admission(f, X, Y, tol: float = 0.05):
    return _reclaim_admission(f, X, Y, tol)
