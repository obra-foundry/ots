"""Aggregation contrast channels: the verified counterexamples and the
cluster-restricted null, trimmed to library runtime.
"""

import numpy as np
import pytest
from scipy import stats as st

from structural_impedance.lmoment import (aggregation_contrast_test,
                                          _permuted_assignment,
                                          _validate_cluster_purity)


def two_groups(a, b):
    return np.concatenate([a, b]), np.array([0] * a.size + [1] * b.size)


def test_order_attribution_counterexample():
    """Equal population tau4, well-separated tau3 (Johnson SU at fixed b has
    tau4 nearly independent of a): the signed order-4 channel fires on a
    pure 3rd-order difference, which is exactly why moment order is never
    attributed from one channel alone."""
    rng = np.random.default_rng(42)
    g1 = st.johnsonsu.rvs(-0.3, 1.0, size=12_000, random_state=rng)
    g2 = st.johnsonsu.rvs(-2.0, 1.0, size=12_000, random_state=rng)
    x, g = two_groups(g1, g2)
    res = aggregation_contrast_test(x, g, n_perm=299, seed=1)
    assert res["channels"]["dispersion_tau3"]["p"] < 0.05
    assert res["channels"]["signed_tau4"]["p"] < 0.05


def test_mirrored_skew_cancellation():
    """Mirror-image skewness pools to zero skewness: the signed order-3
    channel is ~0 by cancellation; the dispersion channel carries it."""
    sk = st.skewnorm(4)
    mu, sd = sk.mean(), sk.std()
    rng = np.random.default_rng(7)
    g1 = (sk.rvs(size=1500, random_state=rng) - mu) / sd
    g2 = -(sk.rvs(size=1500, random_state=rng) - mu) / sd
    x, g = two_groups(g1, g2)
    res = aggregation_contrast_test(x, g, n_perm=299, seed=2)
    assert res["channels"]["dispersion_tau3"]["p"] < 0.05
    assert abs(res["channels"]["signed_tau3"]["stat"]) < 0.05


def test_location_mixture_signed_negative():
    rng = np.random.default_rng(11)
    x, g = two_groups(rng.normal(-3, 1, 3000), rng.normal(+3, 1, 3000))
    res = aggregation_contrast_test(x, g, n_perm=299, seed=3)
    ch = res["channels"]["signed_tau4"]
    assert ch["p"] < 0.05 and ch["stat"] < 0     # bimodal flattening deflates


def test_cluster_null_calibrates_where_naive_inflates():
    """Shared per-cluster heavy-tailed location effects, identical DGP both
    groups (null true): the naive shuffle is anticonservative, the
    cluster-restricted shuffle is not. Trimmed replication count; the full
    measured rates (naive up to 0.91, cluster 0.03-0.08) come from the
    upstream validation suite and are quoted in the module docstring."""
    reps, naive_fp, clust_fp = 25, 0, 0
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        rows_x, rows_g, rows_c = [], [], []
        cid = 0
        for grp in range(2):
            for _ in range(40):
                eff = st.t(4).rvs(random_state=rng) * 1.5
                rows_x.append(eff + rng.normal(0, 1, 25))
                rows_g.append(np.full(25, grp))
                rows_c.append(np.full(25, cid))
                cid += 1
        x = np.concatenate(rows_x)
        g = np.concatenate(rows_g)
        c = np.concatenate(rows_c)
        rn = aggregation_contrast_test(x, g, n_perm=99, seed=r)
        rc = aggregation_contrast_test(x, g, cluster=c, n_perm=99, seed=r)
        naive_fp += rn["channels"]["signed_tau4"]["p"] < 0.05
        clust_fp += rc["channels"]["signed_tau4"]["p"] < 0.05
    assert naive_fp / reps > 0.3        # naive shuffle badly anticonservative
    assert clust_fp / reps <= 0.16      # cluster shuffle calibrated (MC tolerance)


def test_cluster_purity_enforced():
    g = np.array([0, 0, 0, 1, 1, 1])
    c = np.array([0, 0, 1, 1, 2, 2])    # cluster 1 spans both labels
    with pytest.raises(ValueError, match="cluster purity"):
        aggregation_contrast_test(np.arange(6.0), g, cluster=c, n_perm=9)


def test_strata_compose_with_clusters():
    rng = np.random.default_rng(2)
    g = np.repeat([0, 1, 0, 1], 5)
    c = np.repeat([0, 1, 2, 3], 5)
    s = np.repeat([0, 0, 1, 1], 5)
    _validate_cluster_purity(g, c)
    for _ in range(30):
        perm = _permuted_assignment(g, c, s, rng)
        for cl in np.unique(c):
            assert np.unique(perm[c == cl]).size == 1            # clusters intact
        for stratum in (0, 1):
            assert sorted(perm[s == stratum].tolist()) == \
                sorted(g[s == stratum].tolist())                 # strata respected


def test_gating_and_exchangeability_statement():
    rng = np.random.default_rng(19)
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(0, 1, 300),
                        rng.normal(0, 1, 10), np.zeros(40)])
    g = np.array(["a"] * 300 + ["b"] * 300 + ["small"] * 10 + ["const"] * 40)
    note = "units are independent draws from the same survey design"
    res = aggregation_contrast_test(x, g, n_perm=49, seed=4,
                                    exchangeability_note=note)
    assert res["underpowered"] == ["small"]
    assert res["degenerate"] == ["const"]
    assert res["permutation"]["exchangeability"] == note
    res_default = aggregation_contrast_test(x, g, n_perm=49, seed=4)
    assert "UNVERIFIED" in res_default["permutation"]["exchangeability"]
