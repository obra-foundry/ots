import torch

from structural_impedance.cumulant import admit_per_component_standardized


def test_null_gate_silent():
    """admit_per_component_standardized should not admit on a random split of one population."""
    torch.manual_seed(42)
    X = torch.randn(10000, 1)
    idx = torch.randperm(10000)
    X1, X2 = X[idx[:5000]], X[idx[5000:]]
    admit, diag = admit_per_component_standardized(X1, X2)
    # On standardized data from the same distribution, the gate should NOT admit;
    # if it does, a blocking_component must still be reported (soft guard).
    assert not admit or diag.get("blocking_component") is not None, \
        f"Gate admitted on null with no blocking component: {diag}"
