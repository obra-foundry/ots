# Subgroup Higher-Order Cumulant Audit: Methods Note (v2.1)

*A distributional-heterogeneity diagnostic for subgrouped measurements. Standalone statistics; reference implementation `subgroup_kurtosis.py` (v2.1), verified on synthetic ground truth including four adversarial cases that misclassified under v1 and the mirrored-skew blind spot identified at independent review (closed in v2.1).*

## Purpose

When a numeric variable is measured across subgroups, pooled summaries (mean, SD, a single AUC, pooled calibration error) describe only the first two moments. This audit asks two questions the pooled summary cannot answer:

1. Does any subgroup carry **higher-order (4th-cumulant) distributional structure** — heavier or lighter tails than Gaussian — that pooling erases?
2. When the *pooled* data looks non-Gaussian, is that **real within-group structure**, or an **artifact of mixing** heterogeneous subgroups?

The reported statistic is **R = κ₄/κ₂² (excess kurtosis)**, the 4th standardized cumulant.

## Precision points

**R = 0 is necessary but not sufficient for normality.** Zero excess kurtosis constrains only the 4th cumulant. v2 computes a **skewness (G1) companion** per subgroup so the 3rd cumulant is probed alongside; the verdict remains scoped to the 4th cumulant and says so.

**"Aggregation" is two operations with opposite effects.**
- *Averaging* subgroup summaries pushes the result toward Gaussian by the CLT — real within-group tail structure is washed out.
- *Pooling* raw observations across heterogeneous subgroups **distorts pooled kurtosis in either direction**: location separation typically *deflates* it (bimodal flattening, platykurtosis); scale mixing at a common mean *inflates* it (Jensen gap between Σw·σ⁴ and (Σw·σ²)²). Both directions are artifacts.

Implementation rule: **compute on raw observations, never on pre-aggregated summaries.**

## The decomposition (exact, descriptive)

With subgroup weights wₘ = nₘ/n, within-group central moments (σ²ₘ, m₃ₘ, m₄ₘ), and location deviations δₘ = μₘ − μ_pooled, the pooled 4th central moment splits exactly:

```
m4_pooled = Σ wₘ · m4ₘ                              ← WITHIN
          + Σ wₘ · (4 m3ₘ δₘ + 6 σ²ₘ δₘ² + δₘ⁴)      ← BETWEEN (location-driven)
```

Verified to machine precision (worst relative error 6e-16 over 200 random configurations). **Two caveats demoted this from decision quantity to descriptive output in v2:**

- The BETWEEN term captures **location deviations only**. Pure scale heterogeneity at matched means gives between ≡ 0 while still manufacturing pooled leptokurtosis (e.g., N(0,1)+N(0,9) → pooled R ≈ +1.9, between_share ≈ 0).
- `between_share` is **signed and unbounded** (the 4·m₃ₘ·δₘ cross term can be negative), so a fixed threshold on it has no calibrated meaning.

## The mixing criterion (permutation null)

The artifact verdict binds on the contrast

```
D = R_pooled − Σ wₘ R_g    (G2 per group; weights over non-degenerate cells)
```

tested against a **permutation null** (group labels shuffled, n_perm = 999, add-one p-value). D < 0: mixing deflates pooled kurtosis (location separation). D > 0: mixing inflates it (scale mixing). Either direction means pooled R misrepresents within-group shape. Because D is a population contrast and the permutation calibrates its noise directly, **this test needs no per-group power floor** — small cells cannot veto it.

## The 3rd-cumulant channels (v2.1; closes the mirrored-skew blind spot)

Independent review identified a case both 4th-cumulant channels miss: mirrored skew at equal mean/variance/kurtosis (groups at skew +γ and −γ). Pooling annihilates the skewness — pooled G1 ≈ 0 against real per-group structure — yet R is undistorted. Two contrasts close it:

- **D₃ = G1_pooled − Σ wₘ·G1_g** (signed, permutation null): net 3rd-order pooling distortion, the skew analog of D. **Insufficient alone**: for mirrored skew both terms are 0, so D₃ ≡ 0 by construction (the reviewer's proposed fix stopped here; the gap was caught at implementation).
- **S₃ = Σ wₘ·(G1_g − G1_pooled)²** (dispersion, one-sided permutation null): fires exactly when per-group skews differ from the pooled skew even if they cancel in the weighted mean. This is the channel that catches annihilation. Verified: skewnorm(±4) groups at n=2000/cell → S₃ p < 0.05 with D₃ ≈ 0, as predicted.

Both are reported in the output (`skew_mixing_D/p`, `skew_dispersion_S/p`) and feed `third_order_fired` together with the per-group skew CIs. They append caveats under any verdict; they do not change the (4th-cumulant-scoped) verdict itself.

## Estimation and uncertainty

Excess kurtosis is high-variance, slow-converging, and dominated by extreme observations. Its sampling distribution has finite variance **only if μ₈ < ∞** — so in exactly the heavy-tail regime this tool targets, the classic n-out-of-n percentile bootstrap is not asymptotically calibrated and measured badly anticonservative.

- **Point estimate:** bias-corrected G2 (Joanes & Gill 1998). Note the correction is exact under normality only; it does not remove the large downward heavy-tail bias.
- **Uncertainty:** **m-out-of-n without-replacement subsampling percentile interval**, m = max(64, n^⅔), B ≥ 2000, with a two-sided resampling p-value for R = 0.
- **Multiplicity:** per-group p-values are **Holm-corrected** across subgroups before the structure decision; bootstrap seeds are independent per group (SeedSequence spawn). CIs are reported at unadjusted α for effect-size reading; decisions use p_holm.
- **Measured calibration (200 reps, B=1000):**

| Setting | v1 (percentile, uncorrected) | v2 (m-out-of-n + Holm) |
|---|---|---|
| Normal null, n=200, false-positive rate (nominal 5%) | 7.7–9.2% | **1.5%** (conservative) |
| t(5), n=500, CI coverage of true R=6 (nominal 95%) | 33% | **63%** |
| t(5), n=2000, coverage | 41.5% | **84%** |

The residual undercoverage is the irreducible downward bias of sample kurtosis under heavy tails; intervals are wide and that width is the honest uncertainty of a 4th-moment statistic. **Decisions bind on the conservative null behavior (detection), not on point coverage.**

- **Robust cross-check:** Moors' octile kurtosis (excess form, normal reference 1.2331). v2 **wires the divergence flag**: |G2 − Moors| > 1.5 sets `divergence_flag` per group and surfaces in the message. The divergence is diagnostic, not noise — high classical with low robust means *either* outlier artifact *or* genuine extreme-tail mass the octiles cannot see. The tool flags for human inspection and never auto-resolves.

## Degenerate cells

Cells with zero variance or n < 4 are listed in `degenerate`, excluded from all inference, and can never count as structure (v1 NaN-poisoned them into `excludes_zero=True`). Note a constant cell pooled with a varying cell is itself a scale-mixing artifact; the permutation test catches it (verified).

## Verdicts (five-way, exhaustive)

| Verdict | Condition | Action |
|---|---|---|
| **MIXED** | structured AND mixing | Pooled R misrepresents both effects; report per-subgroup R only. |
| **WITHIN_STRUCTURE** | structured, no mixing | Real subgroup tail structure pooled metrics erase. Report per-subgroup R. |
| **SPURIOUS_POOLED** | no structure, mixing | Pooled kurtosis is a mixing artifact (direction stated). Never vetoed by small cells. |
| **UNDERPOWERED** | no structure, no mixing, any cell n < min_n | R_LOSSLESS cannot be certified; coarsen the partition or collect more data. |
| **R_LOSSLESS** | no structure, no mixing, all cells powered | 4th-cumulant aggregation lossless. Renamed from "LOSSLESS" in v2.1: the verdict certifies the 4th cumulant only. When any 3rd-cumulant channel fires under this verdict, the message states explicitly that the data is NOT distributionally lossless. |

where **structured** = any powered cell with Holm-adjusted p < α for R ≠ 0, and **mixing** = permutation p < α. Underpowered and degenerate cells, Moors divergences, and 3rd-cumulant findings are appended to every verdict message as caveats — UNDERPOWERED only blocks the LOSSLESS certificate, nothing else.

Verified against synthetic ground truth: the four v1 cases (location mixture → SPURIOUS_POOLED; matched t₅ → WITHIN_STRUCTURE; matched Gaussians → LOSSLESS; ~50/cell → UNDERPOWERED) plus four v1 misclassifications now correct (t₅ ± 3 → **MIXED**, was LOSSLESS; scale mixture → **SPURIOUS_POOLED**, was LOSSLESS; constant cell → **SPURIOUS_POOLED**, was NaN-driven WITHIN_STRUCTURE; mixing + tiny irrelevant cell → **SPURIOUS_POOLED**, was UNDERPOWERED). 8/8 in the self-test.

## Subgroup granularity (the power floor)

Excess kurtosis needs substantially larger n than a mean or variance. Fine partitions shrink per-cell n geometrically; below the floor, R_g is noise (the verified underpowered case ranged R_g = −0.23 to +8.6 across cells of ~50). Rules:
- `min_n = NULL_CALIBRATED_MIN_N` (= 200) is the per-cell floor gating only the per-cell CI/structure test (the mixing test self-calibrates at any n). It is **not** a hand-set placeholder: 200 is the per-n point at which the m-out-of-n kurtosis null attains nominal α=0.05 (a threshold FROM the null — evidence: the `--calibrate` block and the always-on regime self-test), i.e. the module's expression of the single null-calibrated regime shared with the library (`gamma_eff_from_rank`).
- **Pre-register the partition.** Holm correction controls the familywise rate *within* the pre-registered partition; it does not license slicing many ways and keeping the interesting one.
- When cells fall below the floor, use a coarser pre-registered partition.

## Application to model validation

The natural use is a distributional check upstream of summary validation metrics, on case-level model outputs (per-case risk scores, calibration residuals, per-case loss, by pre-registered subgroup):
- **WITHIN_STRUCTURE / MIXED**: a subgroup's error distribution has tail structure pooled AUC or pooled calibration error does not see — escalate to subgroup-level error analysis. Under MIXED, do not quote the pooled number at all.
- **SPURIOUS_POOLED**: warns against attributing heavy-tailed (or suspiciously light-tailed) *pooled* error to the model when it is case-mix.

**Scope:** this diagnoses distributional shape, not outcome. A flag is a reason to look, not evidence of harm; linking tail structure to a clinical miss is a separate step. It establishes no causation and replaces no existing metric.

## Relationship to the cumulant-divergence library (one instrument ≠ the other)

This module and the published cumulant library measure **distinct constructs with orthogonal null sets** (established at independent review; verified by execution):
- **This module**: pooling distortion of the pooled summary (D, D₃, S₃ + per-group CIs). Fires on mean-shifted Gaussians (the canonical mixing case) where the library is silent by construction — the library's kernels center each sample about its own mean, so location effects are invisible to it.
- **The library**: its gated test (`admit_per_component`) detects **cross-block dependence** between co-observed feature blocks, not population divergence — for two independent samples it is silent regardless of shape differences. Its divergence primitive (`cumulant_difference`) is a raw statistic with no built-in null; interpret only against a caller-computed permutation null.
- Verified disagreement cells: equal-mean/equal-variance groups with divergent higher cumulants → library divergence statistic fires (κ₃ 26× null, κ₄ 4.9× null) while this module's D-channel is exactly 0 (p = 0.94); shifted Gaussians → this module fires (p = 0.001) while the library reads ≈ 0 at both orders.
- Present both with `joint_a24_verdict` (a 2×2: between-group divergence × pooling distortion); never present either as "the" instrument for the other's question.

## Status

v2.1 verified: estimators (G2/G1/Moors against scipy and theory), exact decomposition (6e-16), permutation mixing test, 3rd-cumulant channels (D₃ + S₃; mirrored-skew case closed), m-out-of-n calibration (measured, table above), Holm correction, degenerate handling, five-verdict logic (9/9 self-test including the blind-spot closure). Remaining deliberate placeholders: the Moors divergence flag threshold 1.5 (heuristic) and α = 0.05 throughout (convention). (`min_n` is no longer a placeholder — it is `NULL_CALIBRATED_MIN_N`, the per-n null-calibration threshold; the single-threshold-regime unification with the library closed 2026-06-12.) Known limitation: CI point coverage under extreme tails remains below nominal (63–84% measured on t₅); detection decisions are conservative by construction, but treat reported CI endpoints as indicative, not exact. Scope: cumulants 3–4; higher orders not probed.
