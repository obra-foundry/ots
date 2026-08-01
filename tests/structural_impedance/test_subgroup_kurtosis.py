"""A24 subgroup higher-order cumulant audit — verdict self-test.

Promoted from the module's original ``__main__`` block (the 8-case / 9-check self-test).
Each case constructs a distribution with a known ground truth and asserts the five-way
verdict of ``subgroup_kurtosis_audit``:

- location mixing of Gaussians              -> SPURIOUS_POOLED (pooled kurtosis is an artifact)
- genuine within-group heavy tails t(5)     -> WITHIN_STRUCTURE
- matched Gaussians                          -> R_LOSSLESS (aggregation loses nothing)
- over-fine partition (~50/cell)             -> UNDERPOWERED
- heavy tails + location mixing              -> MIXED
- scale mixture at a common mean             -> SPURIOUS_POOLED
- a constant (degenerate) group              -> SPURIOUS_POOLED (no NaN poisoning)
- mixing artifact + an irrelevant tiny cell  -> SPURIOUS_POOLED (tiny cell must not veto)

Data construction mirrors the original self-test exactly (same seed, sizes, and order)
so the verdicts are the validated ground truth, not a re-derivation.
"""

import numpy as np
from scipy import stats as st

from structural_impedance.subgroup_kurtosis import subgroup_kurtosis_audit


def _verdict(x, groups):
    return subgroup_kurtosis_audit(x, groups)["verdict"]


def test_a24_verdict_self_test():
    rng = np.random.default_rng(42)

    a = np.concatenate([rng.normal(-3, 1, 5000), rng.normal(+3, 1, 5000)])
    assert _verdict(a, np.array([0] * 5000 + [1] * 5000)) == "SPURIOUS_POOLED"

    b = np.concatenate([st.t(5).rvs(5000, random_state=rng),
                        st.t(5).rvs(5000, random_state=rng)])
    assert _verdict(b, np.array([0] * 5000 + [1] * 5000)) == "WITHIN_STRUCTURE"

    c = np.concatenate([rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)])
    assert _verdict(c, np.array([0] * 5000 + [1] * 5000)) == "R_LOSSLESS"

    d = subgroup_kurtosis_audit(st.t(5).rvs(300, random_state=rng),
                                rng.integers(0, 6, 300))
    assert d["verdict"] == "UNDERPOWERED"

    e = np.concatenate([st.t(5).rvs(5000, random_state=rng) - 3,
                        st.t(5).rvs(5000, random_state=rng) + 3])
    assert _verdict(e, np.array([0] * 5000 + [1] * 5000)) == "MIXED"

    f = np.concatenate([rng.normal(0, 1, 5000), rng.normal(0, 3, 5000)])
    assert _verdict(f, np.array([0] * 5000 + [1] * 5000)) == "SPURIOUS_POOLED"

    g = np.concatenate([np.zeros(500), rng.normal(0, 1, 500)])
    assert _verdict(g, np.array([0] * 500 + [1] * 500)) == "SPURIOUS_POOLED"

    h = np.concatenate([rng.normal(-3, 1, 5000), rng.normal(+3, 1, 5000),
                        rng.normal(0, 1, 20)])
    assert _verdict(h, np.array([0] * 5000 + [1] * 5000 + [2] * 20)) == "SPURIOUS_POOLED"
