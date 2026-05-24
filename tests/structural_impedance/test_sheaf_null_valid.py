import numpy as np
import scipy.stats
import torch

from structural_impedance.sheaf_gluing import sheaf_status_and_kappa

TOL_SE = 8.0  # disagreement tolerance in bootstrap-SE units


def _stat(a):
    return np.array([float(np.mean(a)), float(np.var(a)),
                     float(scipy.stats.skew(a)), float(scipy.stats.kurtosis(a))])


def _boot_se(pop, n, n_boot=200, seed=0):
    r = np.random.default_rng(seed)
    samples = np.array([_stat(r.choice(pop, n, replace=False)) for _ in range(n_boot)])
    return samples.std(axis=0).clip(min=1e-8)


def _se_sections(groups, pooled, seed):
    n = min(len(g) for g in groups)
    se = _boot_se(pooled, n, seed=seed)
    raw = np.array([_stat(g) for g in groups])
    return torch.tensor(raw / se, dtype=torch.float64)


def test_sheaf_null_valid():
    """SE-normalized sheaf reads 'valid' on random splits of the same population."""
    rng = np.random.default_rng(0)
    pop = rng.lognormal(mean=12, sigma=0.8, size=9000)
    groups = [pop[:3000], pop[3000:6000], pop[6000:9000]]
    sections = _se_sections(groups, pop, seed=1000)
    overlaps = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    status, _, _ = sheaf_status_and_kappa(sections, overlaps, tol=TOL_SE)
    assert status == "valid", f"Sheaf obstructed on null at tol={TOL_SE} SE: {status}"


def test_sheaf_signal_obstructed():
    """SE-normalized sheaf reads 'obstructed' when two populations genuinely differ."""
    rng = np.random.default_rng(0)
    a = rng.lognormal(mean=11.6, sigma=0.8, size=3000)
    b = rng.lognormal(mean=12.4, sigma=0.8, size=3000)
    pooled = np.concatenate([a, b])
    sections = _se_sections([a, b, pooled], pooled, seed=2000)
    overlaps = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    status, _, tag = sheaf_status_and_kappa(sections, overlaps, tol=TOL_SE)
    assert status == "obstructed", f"Sheaf failed to detect real divergence at tol={TOL_SE} SE"
    assert tag == "sheaf:overlap_0_1", f"Unexpected obstruction overlap: {tag}"
