"""
subgroup_kurtosis_audit.py  (v2)

A distributional-heterogeneity diagnostic for subgrouped data.

Given a numeric variable measured across pre-registered subgroups, this module
answers: does any subgroup carry higher-order (4th-cumulant) distributional
structure that a pooled summary erases -- and is apparent structure in the
pooled data real, or just an artifact of mixing subgroups with different
locations/scales?

The reported quantity R = kappa_4 / kappa_2^2 is excess kurtosis (the 4th
standardized cumulant). Precision points governing the implementation:

  1. R = 0 is NECESSARY but NOT SUFFICIENT for normality. Zero excess kurtosis
     certifies only the 4th cumulant. A skewness (G1) companion is computed and
     reported alongside, but the verdict is scoped to the 4th cumulant.

  2. Two distinct operations both get called "aggregation":
       - Averaging subgroup summaries drives R -> 0 by the CLT (washout).
       - Pooling raw observations across heterogeneous subgroups DISTORTS
         pooled kurtosis in either direction: location separation typically
         manufactures platykurtosis (bimodal flattening); scale mixing at a
         common mean manufactures leptokurtosis (Jensen gap between
         sum w*sigma^4 and (sum w*sigma^2)^2).
     Rule: always compute on raw observations, never on pre-aggregated
     summaries.

  3. The within/between decomposition of m4 below is EXACT but its between
     term captures location deviations only: pure scale heterogeneity at
     matched means contributes entirely to the WITHIN sum and leaves
     between == 0. The decomposition is therefore reported as descriptive
     output; the mixing-artifact verdict binds on a permutation test of
     D = R_pooled - sum_g w_g R_g, which is sensitive to both location- and
     scale-mixing.

  4. Inference layer: the sampling distribution of sample kurtosis has finite
     variance only if the 8th moment is finite; in the heavy-tail regime this
     tool targets, the classic n-out-of-n percentile bootstrap is severely
     anticonservative (measured ~33% coverage of a nominal 95% CI on t(5)).
     The operational default is therefore an m-out-of-n WITHOUT-replacement
     subsampling percentile interval (m = max(64, n^(2/3))), Holm-corrected
     across subgroups. Intervals are wide; that width is the honest
     uncertainty of a 4th-moment statistic.

Dependencies: numpy only (scipy used solely in the self-test).
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
# Central moments and cumulant estimators
# ----------------------------------------------------------------------

def _central_moments(x):
    """Plug-in central moments m2, m3, m4 (divide by n). Used for the exact
    additive decomposition identity, not for reporting."""
    x = np.asarray(x, float)
    n = x.size
    d = x - x.mean()
    return n, x.mean(), np.mean(d**2), np.mean(d**3), np.mean(d**4)


def excess_kurtosis_biased(x):
    """g2 = m4/m2^2 - 3 (biased plug-in)."""
    _, _, m2, _, m4 = _central_moments(x)
    return np.nan if m2 == 0 else m4 / m2**2 - 3.0


def excess_kurtosis_G2(x):
    """Bias-corrected sample excess kurtosis (Joanes & Gill 1998; SAS/SPSS G2).
    The correction is exact under normality only; it does not remove the
    (large, downward) heavy-tail bias of sample kurtosis."""
    x = np.asarray(x, float)
    n = x.size
    if n < 4:
        return np.nan
    g2 = excess_kurtosis_biased(x)
    if not np.isfinite(g2):
        return np.nan
    return (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)


def skewness_G1(x):
    """Bias-corrected sample skewness (Joanes & Gill 1998 G1). The 3rd-cumulant
    companion: probes the structure that R = 0 cannot certify."""
    x = np.asarray(x, float)
    n = x.size
    if n < 3:
        return np.nan
    _, _, m2, m3, _ = _central_moments(x)
    if m2 == 0:
        return np.nan
    g1 = m3 / m2**1.5
    return g1 * np.sqrt(n * (n - 1.0)) / (n - 2.0)


def kurtosis_se(n):
    """SE of sample excess kurtosis under normality (finite-sample SEK).
    Normal-theory only; do not use for decisions in heavy-tail regimes."""
    if n < 4:
        return np.nan
    ses = np.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    return 2.0 * ses * np.sqrt((n**2 - 1) / ((n - 3) * (n + 5)))


def moors_kurtosis(x):
    """Robust, quantile-based kurtosis (Moors 1988), octile measure, EXCESS form
    (normal reference 1.2331 subtracted, so 0 ~ mesokurtic). Immune to
    single-outlier domination. Measures the shoulders, not the extreme tails;
    classical-vs-Moors divergence is surfaced per group as a flag for human
    inspection (outlier artifact vs genuine heavy tails), never auto-resolved."""
    x = np.asarray(x, float)
    if x.size < 8:
        return np.nan
    e = np.quantile(x, [i / 8 for i in range(1, 8)])
    e1, e2, e3, _, e5, e6, e7 = e
    denom = e6 - e2
    return np.nan if denom == 0 else ((e7 - e5) + (e3 - e1)) / denom - 1.2331


# ----------------------------------------------------------------------
# Resampling inference: m-out-of-n percentile interval + bootstrap p
# ----------------------------------------------------------------------

def default_subsample_size(n):
    """m-out-of-n subsample size: m = max(64, n^(2/3)), capped at n."""
    return int(min(n, max(64, round(n ** (2.0 / 3.0)))))


def _resample_stats(x, fn, B, m, rng):
    n = x.size
    if m >= n:
        return np.array([fn(x[rng.integers(0, n, n)]) for _ in range(B)])
    return np.array([fn(x[rng.choice(n, m, replace=False)]) for _ in range(B)])


def bootstrap_regret(m, n):
    """The STATED regret of a resampling-size choice (A6: the choice is never
    silent). Returns the mode, the m actually used, m/n, and the cost incurred.

      m-out-of-n (m<n, without replacement): consistent for kurtosis under heavy
        tails WITHOUT the finite-8th-moment requirement -- at the cost of wider
        intervals and reduced efficiency (effective rate ~m^{1/2}, not n^{1/2}).
        Regret = power, paid for validity. The operational default.
      full n-out-of-n (m>=n, with replacement): efficient but ANTICONSERVATIVE
        for kurtosis under heavy tails -- its sampling variance needs a finite
        8th moment, which the heavy-tailed target regime violates, so intervals
        under-cover (measured ~0.33 at nominal 0.95; see --calibrate, the A6
        falsifier). Regret = false confidence. Explicit opt-in only.
    """
    if m >= n:
        return dict(mode="full_n_out_of_n", m=int(n), m_over_n=1.0,
                    regret="anticonservative under heavy tails (needs finite 8th moment)")
    return dict(mode="m_out_of_n", m=int(m), m_over_n=float(m) / float(n),
                regret="wider intervals + reduced efficiency (effective rate ~m^1/2)")


def bootstrap_ci(x, fn=excess_kurtosis_G2, B=2000, alpha=0.05, seed=0, m="auto"):
    """Percentile interval over B resampled statistics, plus a two-sided
    resampling p-value for H0: stat = 0.

    m="auto" (DEFAULT) -> m-out-of-n WITHOUT-replacement subsampling at
             default_subsample_size(n); consistent for kurtosis under heavy
             tails. THE operational default. (A6: replaces the prior SILENT
             n-out-of-n default, which was anticonservative for this audit's
             own heavy-tailed target regime -- a silent wrong default, now gone.)
    m="full" or m=None -> classic n-out-of-n with-replacement bootstrap;
             ANTICONSERVATIVE for kurtosis under heavy tails (variance needs a
             finite 8th moment). EXPLICIT opt-in only; retained for comparison.
    m=int  -> that subsample size.

    The resampling regret is never silent: see bootstrap_regret(m_used, n).
    Returns (lo, hi, p). All-nan resamples (degenerate data) return nans.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n = x.size
    if isinstance(m, str):
        if m == "auto":
            m_eff = default_subsample_size(n)
        elif m == "full":
            m_eff = n
        else:
            raise ValueError(f"m must be 'auto', 'full', None, or int; got {m!r}")
    else:
        m_eff = n if m is None else int(m)
    stats = _resample_stats(x, fn, B, m_eff, rng)
    stats = stats[np.isfinite(stats)]
    if stats.size < max(50, B // 4):
        return np.nan, np.nan, np.nan
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    p = 2.0 * min(np.mean(stats <= 0.0), np.mean(stats >= 0.0))
    p = min(1.0, max(p, 1.0 / stats.size))
    return float(lo), float(hi), float(p)


def holm_adjust(pvals):
    """Holm step-down adjusted p-values, order-preserving. NaNs stay NaN."""
    p = np.asarray(pvals, float)
    adj = np.full(p.shape, np.nan)
    k = np.where(np.isfinite(p))[0]
    if k.size:
        ps = p[k]
        order = np.argsort(ps)
        G = ps.size
        stepdown = (G - np.arange(G)) * ps[order]
        monotone = np.minimum(1.0, np.maximum.accumulate(stepdown))
        out = np.empty(G)
        out[order] = monotone
        adj[k] = out
    return adj


# ----------------------------------------------------------------------
# THE single null-calibrated threshold regime (both A24 instruments).
# Reconciliation: A24_instruments_reconciliation.md §T2 / §7. The whole
# tree thresholds on ONE quantity -- a percentile rank against a
# permutation/resampling null -- not on two fixed cuts (the library's
# fixed R-magnitude R_crit and this module's fixed min_n power floor).
# ----------------------------------------------------------------------

# Per-n point where the m-out-of-n excess-kurtosis null attains nominal
# alpha=0.05. CALIBRATED, not hand-set: the floor's VALUE is already a
# threshold-from-the-null (evidence: the __main__ --calibrate block and the
# always-on calibration self-test). Naming it makes the module's power gate
# an expression of the single regime, not a second ad-hoc threshold.
NULL_CALIBRATED_MIN_N = 200


def null_percentile_rank(stat_obs, null_stats):
    """Percentile rank of an observed statistic within its permutation/
    resampling null, in [0, 1] -- THE null-calibrated quantity the tree
    thresholds on.

        rank = (#{null < obs} + 0.5*#{null == obs} + 0.5) / (N + 1)

    (mid-rank, add-one regularized.) rank=0.5 => null-typical; rank->1 / ->0
    => upper / lower null tail. Feeds gamma_correction.gamma_eff_from_rank(
    rank, Gamma_0) directly (0.5->0, ->1->+Gamma_0, ->0->-Gamma_0), so the
    library's modulation and this module's verdicts run off the SAME regime.
    Detector-not-magnitude by construction: a calibrated comparison against a
    null, never a raw magnitude. NaN-safe (non-finite nulls dropped).

    Conformance: permutation-rank calibration, Lehmann & Romano, Testing
    Statistical Hypotheses, Ch. 15.
    """
    s = np.asarray(null_stats, float)
    s = s[np.isfinite(s)]
    if s.size == 0 or not np.isfinite(stat_obs):
        return np.nan
    below = float(np.count_nonzero(s < stat_obs))
    ties = float(np.count_nonzero(s == stat_obs))
    return (below + 0.5 * ties + 0.5) / (s.size + 1.0)


# ----------------------------------------------------------------------
# Mixing-artifact permutation test (binds the SPURIOUS/MIXED verdicts)
# ----------------------------------------------------------------------

def mixing_permutation_test(x, groups, n_perm=999, seed=0, min_group=4,
                            fn=excess_kurtosis_G2):
    """
    Permutation test of H0: group labels are exchangeable (no between-group
    distributional heterogeneity).

    Statistic (signed contrast): D = fn(pooled) - sum_g w_g fn(group), with
    weights renormalized over groups with n >= min_group and nonzero variance.
    With the default fn (excess kurtosis G2): D < 0 means mixing deflates
    pooled kurtosis (location separation, bimodal flattening); D > 0 means
    mixing inflates it (scale mixing at common mean). With fn=skewness_G1 the
    same contrast (call it D3) detects NET 3rd-order pooling distortion.

    KNOWN LIMITATION of the signed contrast: it vanishes identically when
    per-group values are heterogeneous but average to the pooled value --
    e.g. mirrored skew (+g, -g) pools to 0 with weighted mean 0, so D3 = 0
    even though pooling annihilated the skewness. Use
    dispersion_permutation_test for that failure mode.

    This is the artifact criterion, NOT the m4 between_share: the m4 between
    term is blind to scale mixing, and D is an exact-identity-free contrast
    that needs no per-group power floor (small cells are calibrated by the
    permutation null itself).

    Returns dict(D, p, n_perm). p is the standard add-one permutation p-value.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    groups = np.asarray(groups)
    labels = np.unique(groups)
    stat_pooled = fn(x)
    if not np.isfinite(stat_pooled):
        return dict(D=np.nan, p=np.nan, n_perm=0, rank=np.nan)

    def D_of(assign):
        num = den = 0.0
        for lab in labels:
            xs = x[assign == lab]
            if xs.size < min_group or np.var(xs) == 0:
                continue
            s = fn(xs)
            if np.isfinite(s):
                num += xs.size * s
                den += xs.size
        return np.nan if den == 0 else stat_pooled - num / den

    D_obs = D_of(groups)
    if not np.isfinite(D_obs):
        return dict(D=np.nan, p=np.nan, n_perm=0, rank=np.nan)
    exceed = valid = 0
    null = []
    for _ in range(n_perm):
        Dp = D_of(groups[rng.permutation(x.size)])
        if np.isfinite(Dp):
            valid += 1
            exceed += abs(Dp) >= abs(D_obs)
            null.append(Dp)
    p = (1.0 + exceed) / (1.0 + valid) if valid else np.nan
    rank = null_percentile_rank(D_obs, null) if valid else np.nan
    return dict(D=float(D_obs), p=float(p), n_perm=int(valid), rank=float(rank))


def dispersion_permutation_test(x, groups, n_perm=999, seed=0, min_group=4,
                                fn=skewness_G1):
    """
    Permutation test for shape-heterogeneity that pooling AVERAGES AWAY.

    Statistic: S = sum_g w_g (fn(group) - fn(pooled))^2. Zero iff every group
    matches the pooled shape; large when group shapes differ from the pooled
    summary even if they cancel in the weighted mean. This is the instrument
    for the mirrored-skew blind spot: groups at skew +g and -g pool to skew 0,
    the signed contrast D3 is identically 0, but S = g^2 > 0 fires.

    Default fn is skewness (the 3rd-cumulant annihilation case); pass
    fn=excess_kurtosis_G2 for the kurtosis analog (per-group R heterogeneity
    that the weighted mean hides).

    Returns dict(S, p, n_perm). p is the standard add-one permutation p-value.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    groups = np.asarray(groups)
    labels = np.unique(groups)
    stat_pooled = fn(x)
    if not np.isfinite(stat_pooled):
        return dict(S=np.nan, p=np.nan, n_perm=0, rank=np.nan)

    def S_of(assign):
        num = den = 0.0
        for lab in labels:
            xs = x[assign == lab]
            if xs.size < min_group or np.var(xs) == 0:
                continue
            s = fn(xs)
            if np.isfinite(s):
                num += xs.size * (s - stat_pooled) ** 2
                den += xs.size
        return np.nan if den == 0 else num / den

    S_obs = S_of(groups)
    if not np.isfinite(S_obs):
        return dict(S=np.nan, p=np.nan, n_perm=0, rank=np.nan)
    exceed = valid = 0
    null = []
    for _ in range(n_perm):
        Sp = S_of(groups[rng.permutation(x.size)])
        if np.isfinite(Sp):
            valid += 1
            exceed += Sp >= S_obs       # one-sided: dispersion is nonnegative
            null.append(Sp)
    p = (1.0 + exceed) / (1.0 + valid) if valid else np.nan
    rank = null_percentile_rank(S_obs, null) if valid else np.nan
    return dict(S=float(S_obs), p=float(p), n_perm=int(valid), rank=float(rank))


# ----------------------------------------------------------------------
# Dependence-aware null (cluster-restricted permutation).
#
# The plain mixing_permutation_test above permutes OBSERVATIONS, which assumes
# the observations are exchangeable under H0. Real subgroup data are clustered
# (aftershock sequences; batch instrument discovery; repeated measures), and
# observation-shuffling breaks those clusters, narrowing the null and inflating
# the false-positive rate (boundary-hunt close 1186: ~0.65-0.73 FPR on true
# nulls under clustering). The dependence-correct null permutes whole CLUSTERS
# across groups, keeping each cluster intact, which restores calibration
# (~alpha). This is the encoded backward-flow annotation to A24 mandated by the
# functor-fidelity discipline (1186.functor-fidelity.tier-monotonicity.md):
# A24's permutation null REQUIRES dependence-aware calibration. Relocated to
# this canonical layer from reference_witnesses/boundary.py so the calibration
# engine (NullGym, Ring-2) and the witness suite (Ring-3) share one primitive.
# ----------------------------------------------------------------------

def mixing_contrast_D(x, groups, fn=excess_kurtosis_G2, min_group=4):
    """The signed mixing contrast D = fn(pooled) - sum_g w_g fn(group), with
    weights renormalized over groups with n >= min_group and nonzero variance.
    Same statistic mixing_permutation_test computes internally; exposed as a
    standalone so the cluster-restricted null and the observation null share one
    definition. Returns nan if no group qualifies."""
    x = np.asarray(x, float)
    groups = np.asarray(groups)
    stat_pooled = fn(x)
    if not np.isfinite(stat_pooled):
        return np.nan
    num = den = 0.0
    for g in np.unique(groups):
        xs = x[groups == g]
        if xs.size < min_group or np.var(xs) == 0:
            continue
        s = fn(xs)
        if np.isfinite(s):
            num += xs.size * s
            den += xs.size
    return np.nan if den == 0 else stat_pooled - num / den


def cluster_restricted_mixing_test(x, groups, clusters, fn=excess_kurtosis_G2,
                                   n_perm=999, seed=0, min_group=4):
    """Cluster-restricted permutation test of the mixing contrast: the group
    label is taken as constant within a cluster (modal), and the null permutes
    whole CLUSTERS across groups (members stay intact), so the test respects
    within-cluster dependence. Use to re-check a headline result under a
    dependence-correct null where the data cluster. Returns dict(D, p, n_units).
    Two-sided add-one permutation p-value."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    groups = np.asarray(groups)
    clusters = np.asarray(clusters)
    cl = np.unique(clusters)
    cl_group = {}
    for c in cl:
        g = groups[clusters == c]
        vals, cnts = np.unique(g, return_counts=True)
        cl_group[c] = vals[cnts.argmax()]
    cl_labels = np.array([cl_group[c] for c in cl])
    D_obs = mixing_contrast_D(x, groups, fn=fn, min_group=min_group)
    if not np.isfinite(D_obs):
        return dict(D=np.nan, p=np.nan, n_units=int(cl.size))
    exceed = valid = 0
    for _ in range(n_perm):
        perm = dict(zip(cl, rng.permutation(cl_labels)))
        og = np.array([perm[c] for c in clusters])
        Dp = mixing_contrast_D(x, og, fn=fn, min_group=min_group)
        if np.isfinite(Dp):
            valid += 1
            exceed += abs(Dp) >= abs(D_obs)
    return dict(D=float(D_obs), p=float((1.0 + exceed) / (1.0 + valid))
                if valid else np.nan, n_units=int(cl.size))


def joint_a24_verdict(between_group_divergence, pooling_distortion):
    """
    Joint 2x2 cell for the two distinct aggregation-audit constructs:
    between-group shape divergence (e.g. a per-order cumulant-difference gate)
    x pooling distortion of the pooled summary (this module's permutation
    contrasts). The two statistics have orthogonal null sets -- each can fire
    where the other is silent -- so a joint report is the honest presentation
    when both instruments are run on the same data.

    Returns (cell, label) where cell is (between, pooling) as booleans.
    """
    cell = (bool(between_group_divergence), bool(pooling_distortion))
    labels = {
        (False, False): "NEITHER: no between-group shape divergence detected, "
                        "no pooling distortion detected",
        (True, False):  "DIVERGENCE-ONLY: groups differ in shape, but the pooled "
                        "summary is not distorted (e.g. equal-mean/variance groups "
                        "whose higher moments average out)",
        (False, True):  "DISTORTION-ONLY: groups individually unremarkable, but "
                        "pooling manufactures or destroys shape (e.g. location "
                        "mixing manufacturing platykurtosis)",
        (True, True):   "BOTH: groups differ in shape AND the pooled summary "
                        "misrepresents them",
    }
    return cell, labels[cell]


# ----------------------------------------------------------------------
# Exact within/between decomposition of the pooled 4th central moment
# ----------------------------------------------------------------------

def decompose_m4(x, groups):
    """
    Exact decomposition of the pooled 4th central moment:

        m4_pooled = SUM_g w_g * m4_g                                  (WITHIN)
                  + SUM_g w_g * (4 m3_g d_g + 6 s2_g d_g^2 + d_g^4)   (BETWEEN)

    where w_g = n_g/n and d_g = mu_g - mu_pooled. CAVEATS (v2):
      - The BETWEEN term captures location deviations only. Pure scale
        heterogeneity at matched means gives between == 0 while still
        manufacturing pooled leptokurtosis; use mixing_permutation_test for
        artifact adjudication.
      - between_share is SIGNED and unbounded (the 4*m3_g*d_g cross term can
        be negative); it is descriptive output, not a decision quantity.
      - Per-group R here is the PLUG-IN ratio (R_g_plugin), kept for
        decomposition consistency; the audit reports bias-corrected G2.
    """
    x = np.asarray(x, float)
    groups = np.asarray(groups)
    n = x.size
    mu = x.mean()
    within = between = 0.0
    per_group = {}
    for g in np.unique(groups):
        xs = x[groups == g]
        w = xs.size / n
        _, mu_g, s2_g, m3_g, m4_g = _central_moments(xs)
        d = mu_g - mu
        within += w * m4_g
        between += w * (4 * m3_g * d + 6 * s2_g * d**2 + d**4)
        per_group[str(g)] = dict(
            n=int(xs.size), w=float(w), mean=float(mu_g), var=float(s2_g),
            R_g_plugin=float((m4_g - 3 * s2_g**2) / s2_g**2) if s2_g > 0 else np.nan)
    m4_pooled = within + between
    return dict(m4_pooled=float(m4_pooled), within=float(within),
                between=float(between),
                between_share=float(between / m4_pooled) if m4_pooled != 0 else np.nan,
                per_group=per_group)


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------

def _validate(x, groups):
    x = np.asarray(x, float)
    groups = np.asarray(groups)
    if x.size != groups.size:
        raise ValueError(f"x ({x.size}) and groups ({groups.size}) length mismatch")
    if x.size == 0:
        raise ValueError("empty input")
    bad = ~np.isfinite(x)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} non-finite values in x; clean before auditing")
    try:
        raw = np.unique(groups)
    except TypeError as e:
        raise ValueError(f"group labels are not comparable/sortable (mixed types?): {e}") from e
    if len({str(r) for r in raw}) < raw.size:
        raise ValueError("group labels collide after str() conversion (e.g., 1 vs '1')")
    return x, groups


# ----------------------------------------------------------------------
# Full audit with decision logic
# ----------------------------------------------------------------------

MOORS_DIVERGENCE_FLAG = 1.5   # heuristic |G2 - Moors| threshold for human inspection


def subgroup_kurtosis_audit(x, groups, min_n=NULL_CALIBRATED_MIN_N, B=2000, n_perm=999,
                            alpha=0.05, seed=0, subsample="auto"):
    """
    Per-subgroup R (G2) with m-out-of-n subsampling CIs and Holm-corrected
    significance, a permutation test for mixing distortion of pooled R, a
    3rd-cumulant mixing channel (signed contrast D3 + dispersion contrast S3),
    the exact m4 decomposition (descriptive), a Moors robust cross-check with
    a wired divergence flag, a skewness (G1) companion, and a five-way verdict
    SCOPED TO THE 4TH CUMULANT:

      MIXED            real within-group structure AND mixing distortion
      WITHIN_STRUCTURE real within-group structure, no mixing distortion
      SPURIOUS_POOLED  no within-group structure; pooled R distorted by mixing
                       (decided by the permutation test; needs NO per-group
                       power, so it is never vetoed by an underpowered cell)
      UNDERPOWERED     no structure detected, no mixing detected, but cells
                       below min_n mean R_LOSSLESS cannot be certified.
                       min_n defaults to NULL_CALIBRATED_MIN_N -- the per-n
                       point at which the m-out-of-n kurtosis null attains
                       nominal alpha (a threshold FROM the null, not a hand-set
                       floor), so this gate is the module's expression of the
                       SAME null-calibrated regime the library applies via
                       gamma_eff_from_rank. One regime, two instruments
                       (A24_instruments_reconciliation.md §T2/§7).
      R_LOSSLESS       all cells powered, no 4th-cumulant structure, no mixing
                       distortion of R. Deliberately NOT named "lossless":
                       the verdict certifies the 4th cumulant only. The
                       3rd-cumulant channels (skew CIs, D3, S3) are reported
                       alongside and can fire under any verdict; when they
                       fire under R_LOSSLESS the message says explicitly that
                       the data is NOT distributionally lossless.

    Verdict inputs:
      structured: any powered, non-degenerate group with Holm-adjusted
                  resampling p < alpha for R_g != 0.
      mixing:     mixing_permutation_test p < alpha.
    Degenerate groups (zero variance or n < 4) are excluded from inference and
    listed; they never count as structure (v1 NaN-poisoning fix).
    The result also carries `mixing_rank` / `skew_mixing_rank` /
    `skew_dispersion_rank`: the null-calibrated percentile rank (one regime)
    of each permutation statistic against its own null, ready to feed
    gamma_correction.gamma_eff_from_rank. Each rank is of THIS module's own
    pooling-distortion statistic -- it is NOT the library's between-group
    divergence (the two instruments measure distinct constructs; T1), so the
    shared object is the regime, never a piped magnitude.
    CIs are reported per group at unadjusted level alpha for effect-size
    reading; the structure decision uses p_holm.
    subsample: "auto" (default) -> m = default_subsample_size(n_g);
               None -> classic bootstrap (anticonservative under heavy tails);
               int -> explicit m.
    """
    x, groups = _validate(x, groups)
    dec = decompose_m4(x, groups)
    R_pooled = excess_kurtosis_G2(x)
    mix = mixing_permutation_test(x, groups, n_perm=n_perm, seed=seed + 10_007)
    skew_mix = mixing_permutation_test(x, groups, n_perm=n_perm,
                                       seed=seed + 20_011, fn=skewness_G1)
    skew_disp = dispersion_permutation_test(x, groups, n_perm=n_perm,
                                            seed=seed + 30_013, fn=skewness_G1)

    labels = list(dec["per_group"].keys())
    children = np.random.SeedSequence(seed).spawn(len(labels))
    gstr = groups.astype(str)

    pg, underpowered, degenerate = {}, [], []
    kurt_p, skew_p, tested = [], [], []
    for i, g in enumerate(labels):
        xs = x[gstr == g]
        n_g = xs.size
        entry = dict(n=int(n_g), w=dec["per_group"][g]["w"])
        if n_g < 4 or np.var(xs) == 0:
            degenerate.append(g)
            entry.update(R_g=np.nan, R_robust=np.nan, moors_divergence=np.nan,
                         divergence_flag=None, ci_lo=np.nan, ci_hi=np.nan,
                         p_raw=np.nan, p_holm=np.nan, excludes_zero=None,
                         skew_G1=np.nan, skew_p_raw=np.nan, skew_p_holm=np.nan,
                         skew_structured=None,
                         note="degenerate (zero variance or n<4); excluded from inference")
            pg[g] = entry
            continue
        Rg = excess_kurtosis_G2(xs)
        Rrob = moors_kurtosis(xs)
        div = Rg - Rrob if (np.isfinite(Rg) and np.isfinite(Rrob)) else np.nan
        entry.update(R_g=float(Rg), R_robust=float(Rrob) if np.isfinite(Rrob) else np.nan,
                     moors_divergence=float(div) if np.isfinite(div) else np.nan,
                     divergence_flag=(bool(abs(div) > MOORS_DIVERGENCE_FLAG)
                                      if np.isfinite(div) else None),
                     skew_G1=float(skewness_G1(xs)))
        if n_g < min_n:
            underpowered.append(g)
            entry.update(ci_lo=np.nan, ci_hi=np.nan, p_raw=np.nan, p_holm=np.nan,
                         excludes_zero=None, skew_p_raw=np.nan, skew_p_holm=np.nan,
                         skew_structured=None)
        else:
            gseed = int(children[i].generate_state(1)[0] % np.iinfo(np.int32).max)
            m = (default_subsample_size(n_g) if subsample == "auto"
                 else (None if subsample is None else int(subsample)))
            lo, hi, pk = bootstrap_ci(xs, excess_kurtosis_G2, B=B, alpha=alpha,
                                      seed=gseed, m=m)
            _, _, ps = bootstrap_ci(xs, skewness_G1, B=B, alpha=alpha,
                                    seed=gseed + 1, m=m)
            entry.update(ci_lo=lo, ci_hi=hi, p_raw=pk,
                         skew_p_raw=ps)
            kurt_p.append(pk)
            skew_p.append(ps)
            tested.append(g)
        pg[g] = entry

    for g, pa in zip(tested, holm_adjust(kurt_p)):
        pg[g]["p_holm"] = float(pa) if np.isfinite(pa) else np.nan
        pg[g]["excludes_zero"] = bool(np.isfinite(pa) and pa < alpha)
    for g, pa in zip(tested, holm_adjust(skew_p)):
        pg[g]["skew_p_holm"] = float(pa) if np.isfinite(pa) else np.nan
        pg[g]["skew_structured"] = bool(np.isfinite(pa) and pa < alpha)

    structured = any(v["excludes_zero"] for v in pg.values()
                     if v.get("excludes_zero") is not None)
    skew_any = any(v["skew_structured"] for v in pg.values()
                   if v.get("skew_structured") is not None)
    mixing = bool(np.isfinite(mix["p"]) and mix["p"] < alpha)
    skew_mixing = bool(np.isfinite(skew_mix["p"]) and skew_mix["p"] < alpha)
    skew_dispersed = bool(np.isfinite(skew_disp["p"]) and skew_disp["p"] < alpha)
    third_order_fired = skew_any or skew_mixing or skew_dispersed

    wr = [(v["w"], v["R_g"]) for v in pg.values() if np.isfinite(v.get("R_g", np.nan))]
    wsum = sum(w for w, _ in wr)
    R_within_avg = float(sum(w * r for w, r in wr) / wsum) if wsum > 0 else np.nan

    if structured and mixing:
        verdict = "MIXED"
        msg = (f"Exchangeability rejected on both channels: real within-subgroup 4th-cumulant "
               f"structure (per-subgroup CIs exclude 0) AND a directed pooling contrast "
               f"(permutation p={mix['p']:.4f}, D={mix['D']:+.2f}). Report per-subgroup R, not "
               f"pooled R={R_pooled:.2f}. Licensed: presence + direction, not magnitude.")
    elif structured:
        verdict = "WITHIN_STRUCTURE"
        msg = ("Real within-subgroup 4th-cumulant structure present (per-subgroup CIs exclude 0); "
               "no pooling contrast detected. Pooled or mean-only summaries erase it. Report "
               "per-subgroup R. Licensed: presence + direction, not magnitude.")
    elif mixing:
        verdict = "SPURIOUS_POOLED"
        direction = "deflates" if mix["D"] < 0 else "inflates"
        msg = (f"No within-subgroup structure detected, but exchangeability is rejected via the "
               f"directed kurtosis contrast, which {direction} the pooled statistic: pooled "
               f"R={R_pooled:.2f} vs within-average {R_within_avg:.2f} (D={mix['D']:+.2f}, "
               f"permutation p={mix['p']:.4f}). The pooled kurtosis is not licensed as "
               f"within-group structure; consistent with a pooling artifact (candidate "
               f"mechanisms -- location separation deflates, scale mixing at a common mean "
               f"inflates -- are NOT identified by this test). Licensed: presence + direction, "
               f"not magnitude.")
    elif underpowered:
        verdict = "UNDERPOWERED"
        msg = (f"No structure or mixing distortion detected, but subgroup(s) {underpowered} "
               f"are below n={min_n}: an R_LOSSLESS certificate cannot be issued. Coarsen "
               "the partition or gather more data.")
    else:
        verdict = "R_LOSSLESS"
        if third_order_fired:
            msg = ("4th-cumulant channel clean (per-subgroup R indistinguishable from 0; "
                   "no mixing distortion of pooled R), BUT 3rd-cumulant structure or "
                   "distortion detected -- this data is NOT distributionally lossless. "
                   "See the skewness channel results.")
        else:
            msg = ("No recoverable 4th-cumulant structure: per-subgroup R indistinguishable "
                   "from 0 and no mixing distortion of the pooled statistic. The 3rd-cumulant "
                   "channels (skew CIs, D3 contrast, S3 dispersion) are also quiet. Scoped "
                   "to cumulants 3-4; higher orders not probed.")

    caveats = []
    if underpowered and verdict not in ("UNDERPOWERED",):
        caveats.append(f"underpowered cells {underpowered} excluded from the structure test")
    if degenerate:
        caveats.append(f"degenerate cells {degenerate} excluded from inference")
    flagged = [g for g, v in pg.items() if v.get("divergence_flag")]
    if flagged:
        caveats.append(f"classical-vs-Moors divergence flagged in {flagged}: inspect whether "
                       "outlier artifact or genuine heavy tails (not auto-resolved)")
    if skew_any:
        skg = [g for g, v in pg.items() if v.get("skew_structured")]
        caveats.append(f"skewness companion: 3rd-cumulant structure in {skg}")
    if skew_mixing:
        caveats.append(f"3rd-cumulant mixing: pooling distorts skewness "
                       f"(D3={skew_mix['D']:+.3f}, p={skew_mix['p']:.4f})")
    if skew_dispersed:
        caveats.append(f"3rd-cumulant dispersion: per-group skews differ from the pooled "
                       f"skew -- pooling averages real shape heterogeneity away "
                       f"(S3={skew_disp['S']:.3f}, p={skew_disp['p']:.4f})")
    if caveats:
        msg += " [" + "; ".join(caveats) + "]"

    return dict(verdict=verdict, message=msg,
                R_pooled=float(R_pooled), R_within_avg=R_within_avg,
                mixing_D=mix["D"], mixing_p=mix["p"], mixing_rank=mix["rank"],
                skew_mixing_D=skew_mix["D"], skew_mixing_p=skew_mix["p"],
                skew_mixing_rank=skew_mix["rank"],
                skew_dispersion_S=skew_disp["S"], skew_dispersion_p=skew_disp["p"],
                skew_dispersion_rank=skew_disp["rank"],
                third_order_fired=third_order_fired,
                between_share=float(dec["between_share"]) if np.isfinite(dec["between_share"]) else None,
                m4_within=dec["within"], m4_between=dec["between"],
                underpowered=underpowered, degenerate=degenerate,
                skew_structure=skew_any, per_group=pg)


# ----------------------------------------------------------------------
# Self-test on synthetic ground truth
# ----------------------------------------------------------------------
