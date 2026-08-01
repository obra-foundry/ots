"""L-moment estimators against closed forms and the naive PWM definition.

Closed forms (Hosking & Wallis 1997, Regional Frequency Analysis, App. A):
exponential tau3 = 1/3, tau4 = 1/6; uniform tau3 = tau4 = 0; normal
tau3 = 0, tau4 = 30/pi * arctan(sqrt 2) - 9; GPD (Hosking shape k = -xi)
tau3 = (1-k)/(3+k), tau4 = (1-k)(2-k)/((3+k)(4+k)).
"""

import numpy as np
import pytest
from scipy import stats as st

from structural_impedance.lmoment import sample_lmoments, _pwm_b


@pytest.mark.parametrize("dist,tau3,tau4", [
    (st.expon, 1.0 / 3.0, 1.0 / 6.0),
    (st.uniform, 0.0, 0.0),
    (st.norm, 0.0, 30.0 / np.pi * np.arctan(np.sqrt(2.0)) - 9.0),
    (st.genpareto(0.25), (1 - (-0.25)) / (3 + (-0.25)),
     (1 - (-0.25)) * (2 - (-0.25)) / ((3 + (-0.25)) * (4 + (-0.25)))),
])
def test_sample_ratios_converge_to_closed_forms(dist, tau3, tau4):
    rng = np.random.default_rng(7)
    t3 = np.mean([sample_lmoments(dist.rvs(size=20_000, random_state=rng))["tau3"]
                  for _ in range(5)])
    t4 = np.mean([sample_lmoments(dist.rvs(size=20_000, random_state=rng))["tau4"]
                  for _ in range(5)])
    assert abs(t3 - tau3) < 0.02
    assert abs(t4 - tau4) < 0.02


def naive_b(xs, r):
    n = xs.size
    tot = 0.0
    for i in range(1, n + 1):
        num = 1.0
        for j in range(r):
            num *= (i - 1 - j) / (n - 1 - j)
        tot += num * xs[i - 1]
    return tot / n


@pytest.mark.parametrize("n", [4, 17, 100])
def test_vectorized_pwm_equals_naive_definition(n):
    rng = np.random.default_rng(n)
    xs = np.sort(rng.standard_t(5, n))
    for r in range(4):
        assert np.isclose(_pwm_b(xs, r), naive_b(xs, r), rtol=0, atol=1e-12)


def test_heavy_tail_stability_versus_kurtosis():
    """t(3): population kurtosis undefined, population tau4 exists. The
    sample tau4 must be far more stable than sample excess kurtosis."""
    rng = np.random.default_rng(17)
    tau4s, g2s = [], []
    for _ in range(20):
        x = st.t(3).rvs(size=4000, random_state=rng)
        tau4s.append(sample_lmoments(x)["tau4"])
        g2s.append(st.kurtosis(x))
    rel_tau4 = np.std(tau4s) / abs(np.median(tau4s))
    rel_g2 = np.std(g2s) / abs(np.median(g2s))
    assert rel_tau4 < 0.10
    assert rel_g2 / rel_tau4 > 5.0


def test_degenerate_inputs_are_nan_not_error():
    assert np.isnan(sample_lmoments([1.0, 2.0, 3.0])["tau4"])      # n < 4
    lm = sample_lmoments(np.full(10, 2.5))                          # constant
    assert lm["l2"] == 0.0 and np.isnan(lm["tau3"])
