"""OTS 3.0 — sheaf_gluing kernels (blueprint §5; axioms §4).

Gluing is BINARY by construction: a single overlap violation marks obstructed.
"""

import torch


def cocycle_disagreement(sections: torch.Tensor,   # [P, k]
                         overlaps: torch.Tensor     # [O, 2]
                         ) -> torch.Tensor:
    """Pairwise restriction disagreement on overlap pairs. Returns [O].
    Conformance: Curry 2014 cellular sheaf cocycle equality."""
    i, j = overlaps[:, 0], overlaps[:, 1]
    return (sections[i] - sections[j]).norm(dim=1)


def sheaf_status_and_kappa(sections: torch.Tensor, overlaps: torch.Tensor,
                           tol: float) -> "tuple[str, list[float], str | None]":
    """Returns (status, kappa_residual_list, obstruction_at). A SINGLE overlap
    violation suffices to mark obstructed; do not average/soften/vote.
    Conformance: Curry 2014 cellular sheaf cocycle equality (axioms §4)."""
    disagree = cocycle_disagreement(sections, overlaps)
    if disagree.max().item() < tol:
        return "valid", [], None
    worst = int(disagree.argmax().item())
    return "obstructed", disagree.tolist(), f"sheaf:overlap_{overlaps[worst, 0]}_{overlaps[worst, 1]}"
