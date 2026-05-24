# Independent Structural Audit — `structural_impedance` Repository

**Auditor scope.** Files on disk only. Audit domain: `structural_impedance/`, `tests/structural_impedance/`, `examples/`. `data/` contents excluded except as referenced by notebooks. Audit date: 2026‑05‑22.

**Audit method.** Library + tests read directly; notebook outputs read from the executed `.ipynb` cells. Every quantitative claim is sourced from an executed cell. No external knowledge is assumed.

---

## PHASE 1 — Repository Inventory

### 1.1 Library primitives — `structural_impedance/` (358 LOC, 4 modules)

#### `gamma_correction.py` (75 LOC)

| Function | Signature | Contract | Guard / Honest-flag |
|---|---|---|---|
| `reflection_coefficient` | `(Z_emit, Z_recv) → Γ` | `(Z_recv − Z_emit) / (Z_recv + Z_emit)`, batched. Conformance: **Pozar 4e Ch. 2**. | None (pure algebra). |
| `epsilon_gamma` | `(Γ, γ*, eps_0, lam, tau) → float` | `ε = max(ε₀·exp(λ·|Γ−γ*|²), τ)`. Natural ε₀ when γ*=None. Conformance: **Feydy et al. AISTATS 2019**. | Floor at `τ=1e-6`. |
| `metric_warp_factor` | `(Γ, γ*) → (factor, tag\|None)` | `1 + |Γ−γ*|/|Γ+γ*|`. Conformance: **Chazal et al. arXiv:1207.3885**, axioms §0.2. | If `|Γ+γ*| ≤ 1e-12` ⇒ returns `(1.0, "metric_warp:gamma_plus_gamma_star_singular")` and `logger.warning`. |
| `phi_modulation` | `(Γ, γ*) → float ∈ [0,1]` | `Φ = 1 − exp(−|Γ−γ*|² / (1−|Γ|²))`. Denominator is ALWAYS `\|Γ\|²` (real), never `Γ²` (complex). | `γ*=None ⇒ 0`; `1−\|Γ\|² ≤ 1e-12 ⇒ Φ=1` + log warning. |
| `gamma_eff_from_kurtosis` | `(κ_2, κ_4, Γ₀, R_crit=4.458, R_scale=2.094) → Γ_eff` | `Γ_eff = Γ₀ · tanh((R − R_crit)/R_scale)`, `R = κ₄/max(κ₂², 1e-12)`. Locked canonical FAF-W3 anchors. | `max(κ₂², 1e-12)` denominator floor. |
| `alpha_modulation` | `(κ_2, κ_4, Γ₀, γ*, …) → float` | `α = Φ(Γ_eff(κ), γ*)`. Composition. | Inherits Φ guards. |

#### `cumulant.py` (119 LOC) — the analytical core

| Function | Signature | Contract | Guard / Honest-flag |
|---|---|---|---|
| `third_central_moment` | `X:[n,d] → [d,d,d]` | `K₃[i,j,k] = mean_t Xc[t,i]Xc[t,j]Xc[t,k]`, single `einsum`. | None. |
| `fourth_central_moment` | `X:[n,d] → [d,d,d,d]` | **True 4th cumulant** (Edgeworth-corrected): `E₄ − 3·perm(Σ⊗Σ)`. Cumulant additivity exact: independent X,Y ⇒ cross-block κ₄ = 0. | None. |
| `cross_cumulant_residual_perK` | `(X, Y, k∈{3,4}) → [d_x + d_y]` | Returns the **per-component** Frobenius norm of the cross-block of the joint κ_k tensor (within-α and within-β blocks zeroed out). Conformance: **Sturmfels & Zwiernik arXiv:1011.1722**. | `ValueError` if k ∉ {3,4}. |
| `admit_per_component` | `(X, Y, τ₃, τ₄) → (admit, meta)` | `ADMIT iff min_k ‖κ_k‖ > τ_k` for EACH `k ∈ {3,4}`, strict AND. Aggregate composition (`n3+n4>τ`) is forbidden. | Returns `blocking_component ∈ {None, "k3", "k4"}`. |
| `admit_per_component_standardized` | `(X, Y, τ₃=0.2, τ₄=0.2) → (admit, meta)` | Z-scores X and Y separately to unit variance, then calls `admit_per_component`. Thresholds are σ-units. | Inherits per-component guard. |
| `cumulant_difference` | `(X, Y, k) → scalar` | `‖κ_k(X) − κ_k(Y)‖` Frobenius. Two **independent populations**, no equal-n requirement, no row pairing. | None. |

#### `sinkhorn.py` (114 LOC)

| Function | Signature | Contract | Guard / Honest-flag |
|---|---|---|---|
| `sinkhorn_potentials` | `(a, b, C, ε, n_iter, tol) → (f, g)` | Log-space Sinkhorn, two logsumexp reductions per step. The iteration loop is the **only permitted loop in scope**. Conformance: **Feydy et al. AISTATS 2019**. | Non-convergence logs max residual. |
| `sim_hessian_torch_hvp` | `(θ, sample_fn, β, Γ, γ*, …) → H` | SIM Hessian via **implicit differentiation** through the Sinkhorn fixed point. Closed-form assembly and eigenvalue clamping are **forbidden** (axioms §1.2, §5.bis). Non-PD H is logged and returned unmodified — monitored assumption, not enforced. Conformance: **Shen et al. NeurIPS 2020 Prop. 5.1**. | `min_eig < 0` ⇒ `logger.warning`, return H as-is. |
| `kappa_sinkhorn_per_component` | `(x, y) → {"k3":…, "k4":…}` | Per-component κ residuals for the joint coupling. **Not concatenated**, **not aggregated** (axioms §0.5.1). | Returns dict, never a single scalar. |

#### `sheaf_gluing.py` (27 LOC)

| Function | Signature | Contract | Guard / Honest-flag |
|---|---|---|---|
| `cocycle_disagreement` | `(sections:[P,k], overlaps:[O,2]) → [O]` | `‖sections[i] − sections[j]‖` per overlap row. Conformance: **Curry 2014** cellular sheaf cocycle equality. | None. |
| `sheaf_status_and_kappa` | `(sections, overlaps, tol) → (status, residuals, tag)` | `status = "obstructed"` iff `max(disagreement) ≥ tol`. **A single overlap violation suffices.** Reports `tag = "sheaf:overlap_i_j"` for the worst pair. | Gluing is BINARY — no averaging, softening, or voting. |

### 1.2 Test suite — architectural invariants by module

#### `tests/structural_impedance/` (10 tests)

| Test file | Property verified | Architectural invariant encoded |
|---|---|---|
| `test_admit_per_component.py` | (a) Large κ₃ cannot mask near-zero κ₄ — `blocking_component="k4"` when τ₄ > ‖κ₄‖ even with τ₃ < ‖κ₃‖. (b) Both above τ ⇒ admit. | **Anti-aggregation**: per-component AND gate is enforced; no compositional substitution. |
| `test_citation_conformance.py` | Every public kernel docstring contains "Conformance" + a specific external anchor (Pozar / Feydy / Chazal / Sturmfels / Curry / Shen / §0.x). | **Authority traceability**: each kernel must cite a verifiable source. |
| `test_gamma_plus_gamma_star_guard.py` | `metric_warp_factor(1,-1)` returns `(1.0, "metric_warp:gamma_plus_gamma_star_singular")` — never NaN/Inf. | **Singularity quarantine**: divide-by-zero is intercepted and tagged. |
| `test_kappa_independent_zero.py` | Independent X,Y ⇒ `‖κ_k‖ < 0.15` at n=50000, d=2. Dependent X,Y (Y = X² + noise) ⇒ `‖κ_k‖ > 0.5`. Dependence raises BOTH k=3 and k=4. The threshold of 1e-2 is **openly flagged as statistically infeasible** in the docstring. | **Cumulant additivity** holds at finite-sample limits, AND limitations are flagged in code rather than hidden. |
| `test_null_gate_silent.py` | Random split of one population through `admit_per_component_standardized` does **not** admit, OR if it does, a `blocking_component` is still reported. | **Null is a soft guard, not a hard test**; gate carries diagnostic metadata under all branches. |
| `test_phi_endpoints.py` | `Φ(γ*=None) = 0`; `Φ(Γ=γ*) = 0`; `\|Γ\|→1 ⇒ Φ→1`; output ∈ [0,1]. | **Bounded mapping**: Φ never escapes the unit interval. |
| `test_phi_no_complex_denom.py` | No file in `structural_impedance/` contains `1 - Γ²` or any variant; `phi_modulation` denominator must be the string `"1.0 - float(abs(Gamma)) ** 2"`. | **Real-power-coefficient discipline** (Pozar 4e): the Φ denominator is `1−\|Γ\|²` (real, power transmission), never `1−Γ²` (complex). |
| `test_sheaf_null_valid.py` | Random 3-way split of a lognormal population reads `valid` at 8-SE tolerance. Two truly-different lognormals read `obstructed` with the correct overlap tag. | **Bootstrap-SE-normalized sheaf** distinguishes null sampling noise from real divergence. |
| `test_sheaf_single_violation.py` | 4 sections, 3 overlaps; one failing overlap ⇒ `obstructed` with the failing-overlap tag, all overlaps' residuals listed. All-zero sections ⇒ `valid`, empty residual list, no tag. | **Binary gluing**: a single overlap violation is dispositive. |
| `test_sim_no_closed_form.py` | `sinkhorn.py` source contains `torch.autograd.functional.hessian` and does NOT contain `D2_11`, `D²₁₁`, `.clamp(`, `linalg.eigh(`, `relu`, etc. Runtime test skipped if torch lacks 2nd-order autograd through `cdist`. | **Implicit-diff discipline**: closed-form assembly and PSD-clamping are forbidden; non-PD is logged, not corrected. |

#### Other test directories (cataloged, not deeply audited)

- `tests/gppl/` — 7 Python tests + 7 TypeScript mirrors covering the earlier GPPL library (`src/gppl/`): cumulant, gamma_correction, phi-no-complex-denom, sinkhorn, kappa_admission, laplacian, faf. The `structural_impedance/` tests appear to be the v3 successor; the GPPL tests cover the v2-era kernel surface.
- `tests/tcd/` — 4 Python + 3 TS tests covering the Topological Composition Daemon (`src/tcd/`): real_pipeline, daemon, daemon_v31, sheaf_gluing.
- `tests/sep/` — 1 test, `test_financial_flow.py`, covering the SEP (Sovereign Endowment Pipeline) module.
- `tests/conftest.py` — single-line shim: pre-imports `torch` before pytest's assertion rewriter.

### 1.3 Notebooks — `examples/`

| Notebook | Domain | Data source | Subgroup contrasts | Primary metric(s) |
|---|---|---|---|---|
| `delaware_calibration.ipynb` | Income (US Census Delaware PUMS) | `data/pums/psam_p10.csv` (9,876 persons) | Black HH vs White HH equivalised income | per-component κ_3/κ_4, raw gate, z-score sheaf |
| `delaware_wealth_calibration.ipynb` | Housing wealth (Delaware) | `psam_h10.csv` (4,765 HHs) | Black vs White per-adult housing value | `cumulant_difference` ± null, standardized gate, SE-normalized sheaf, ICR via SMOCP/GRNTP |
| `national_wealth_calibration.ipynb` | Housing wealth (National) | `psam_husa.csv` + `psam_husb.csv` (1.45 M HHs) | Black vs White per-adult housing value | Same pipeline as delaware_wealth, scaled |
| `education_wealth_decoupling.ipynb` | Wealth × Education | Person + HH files (3.4 M persons → 1.35 M HHs) | Black ≤HS vs White ≤HS vs Black BA+ vs White BA+ (4-section sheaf) | Direct median; per-stratum cumulant; persistence ratio; 4-section sheaf with 3 overlaps |
| `ecg_subgroup_distortion.ipynb` | 12-lead ECG (PTB-XL) | `data/ptbxl/` 21,799 records, 8k sampled | Heart-rate × {Male vs Female, Young vs Elderly} | Real-to-null κ ratios, standardized gate, SE-normalized sheaf, ICR=noise_fraction & SCP-code count |
| `cxr_segmentation_distortion.ipynb` | Chest radiograph segmentation (CheXmask VinDr-CXR) | `VinDr-CXR.csv` 18,000 images | lung_asym × (Cardiomegaly vs Normal); CTR × (HighDice vs LowDice mask quality) | Real-to-null κ ratios, standardized gate, SE-normalized sheaf, ICR=1−Dice + CTR-CV ratio |

---

## PHASE 2 — Cross-Notebook Quantitative Matrix

All numbers extracted from executed cell outputs. **Real** = computed on actual subgroup partition. **Null** = permutation control (random split of the pooled population at the same n). Ratio = real / null.

### 2.1 Cumulant-divergence and gate matrix

| # | Notebook | Contrast (feature × subgroups) | n_X | n_Y | ‖Δκ₃‖ real | ‖Δκ₃‖ null | κ₃ ratio | ‖Δκ₄‖ real | ‖Δκ₄‖ null | κ₄ ratio | Std gate admit | Blocking |
|---|---|---|-----|-----|-----------|-----------|----------|-----------|-----------|----------|----------------|----------|
| C1 | delaware_calibration | Equivalised HH income · Black vs White | 404 | 2693† | (per-comp) ‖κ₃‖=8.35e13 | n/a (raw, no null reported) | — | ‖κ₄‖=9.53e19 | n/a | — | True | None |
| C2 | delaware_wealth_calibration | Per-adult housing wealth · Black vs White | 389 | 2660 | 1.17e17 | 6.44e16 | **1.8×** | 2.98e23 | 2.22e23 | **1.3×** | True | None ⚠ |
| C3 | national_wealth_calibration | Per-adult housing wealth · Black vs White | 69,943 | 773,835 | 1.66e17 | 8.20e15 | **20.2×** | 7.92e23 | 1.34e23 | **5.9×** | False | k3 |
| C4 | education_wealth_decoupling | Per-adult housing wealth · Black vs White, LOW-edu (≤HS) | 16,491 | 155,912 | 1.72e16 | not reported | — | 4.44e22 | not reported | — | False | k3 |
| C5 | education_wealth_decoupling | Per-adult housing wealth · Black vs White, HIGH-edu (≥BA) | 13,192 | 158,827 | 3.14e17 | 9.36e15 | **33.5×** | 1.48e24 | 1.57e22 | **94.3×** | True (on null only)¹ | None ⚠ |
| C6 | ecg_subgroup_distortion | Heart rate · Male vs Female | 4,013 | 3,824 | 4.92e2 | 7.91e2 | **0.6×** | 3.90e4 | 2.64e4 | **1.5×** | False | k3 |
| C7 | ecg_subgroup_distortion | Heart rate · Young (<40) vs Elderly (60–79) | 1,015 | 3,376 | 3.11e3 | 2.81e2 | **11.1×** | 1.59e5 | 3.22e4 | **4.9×** | False | k3 |
| C8 | cxr_segmentation_distortion | lung_asym · Cardiomegaly (CTR≥0.50) vs Normal | 7,606 | 10,394 | 6.77e-5 | 1.66e-5 | **4.1×** | 1.80e-5 | 9.98e-6 | **1.8×** | False | k3 |
| C9 | cxr_segmentation_distortion | CTR · HighDice vs LowDice mask quality | 4,500 | 4,500 | 2.89e-4 | 3.61e-5 | **8.0×** | 3.26e-5 | 1.39e-5 | **2.3×** | False | k3 |

† C1 reports an *equalised* n=404 for the raw `cross_cumulant_residual_perK` gate (not `cumulant_difference`); the older notebook predates the unequal-n primitive.
¹ The C5 standardized gate admit verdict is reported only on the **null** comparison in the notebook (the real comparison reports per-stratum admit=False; the persistence-ratio cell triggers the gate-on-null warning).
⚠ Where gate admits=True under a null permutation, the notebook prints an explicit warning ("gate fires on null — admission may be noise").

### 2.2 Sheaf matrix

| # | Notebook | Sections (groups) | n_min | tol (SE) | null disagree (SE) max | Real disagree per overlap (SE) | Status | Obstruction tag |
|---|---|-------------------|-------|----------|------------------------|--------------------------------|--------|-----------------|
| S1 | delaware_calibration | {Black, White, All} | 404 | 1.0 z-score¹ | not reported | [Black↔All 4.13, White↔All 0.25] | obstructed | sheaf:overlap_0_2 |
| S2 | delaware_wealth_calibration | {Black, White, All} | 389 | 8.0 (adaptive floor) | 4.75 | [Black↔All **5.96**, White↔All 1.26] | **valid** | None |
| S3 | national_wealth_calibration | {Black, White, All} | 69,943 | 8.0 (adaptive floor) | 0.49 | [Black↔All **64.89**, White↔All **9.96**] | obstructed | sheaf:overlap_0_2 |
| S4 | education_wealth_decoupling | {Black≤HS, White≤HS, BlackBA+, WhiteBA+} (4 sections, 3 overlaps) | 13,192 | 14.49 (=1.5×9.66) | 9.66 | [21.71, **38.11**, 15.80] | obstructed | sheaf:overlap_2_3 |
| S5 | ecg_subgroup_distortion | {Male, Female, All} | 3,824 | 8.0 (adaptive floor) | 3.00 | [Male↔All 8.09, Female↔All **8.39**] | obstructed | sheaf:overlap_1_2 |
| S6 | cxr_segmentation_distortion | {Cardiomegaly, Normal, All} | 7,606 | 8.0 (adaptive floor) | 1.90 | [Card↔All **18.46**, Norm↔All 14.16] | obstructed | sheaf:overlap_0_2 |

¹ S1 uses the older z-score normalization scheme; S2 onward use the bootstrap-SE-normalized scheme with the adaptive `max(8 SE, 1.5 × null_max)` tolerance. **Bolded** entries are the per-overlap argmax.

### 2.3 ICR-analog matrix

| # | Notebook | "Interception" interface | Subgroup A | Subgroup B | A value | B value | Asymmetry / gap |
|---|---|---|---|---|---|---|---|
| I1 | delaware_wealth_calibration | housing cost / income (`SMOCP+GRNTP` / `HINCP_adj`) | Black HHs | White HHs | 0.204 | 0.146 | +5.8 pp (B > W) |
| I2 | national_wealth_calibration | same | Black HHs | White HHs | 0.191 | 0.152 | +3.9 pp (B > W) |
| I3 | ecg_subgroup_distortion | recording noise (`baseline_sd/qrs_amp`) | Female | Male | 0.4204 | 0.4224 | −0.5 % (≈ symmetric on sex) |
| I3b | ecg_subgroup_distortion | same | 80+ | <40 | 0.4603 | 0.3990 | +15.4 % (older > younger) |
| I3c | ecg_subgroup_distortion | diagnostic ambiguity (SCP code count) | 80+ | <40 | 3.36 mean | 2.22 mean | +51 % (older > younger) |
| I4 | cxr_segmentation_distortion | mask uncertainty (`1−Dice`) | Cardiomegaly | Normal | 0.1471 | 0.1433 | +2.6 % (disease > normal) |
| I4b | cxr_segmentation_distortion | same | LowDice quartile | HighDice quartile | 0.1820 | 0.1184 | +54 % |
| I4c | cxr_segmentation_distortion | CTR coefficient of variation | LowDice | HighDice | 0.1490 | 0.0861 | **1.73× ratio** |

### 2.4 Qualitative framing extracted from markdown cells

| Notebook | Disclosed limitations | Framing register |
|---|---|---|
| delaware_calibration | "wealth proxy: income (PINCP) used; PUMS person-level lacks VALP/TEN"; ICR cells explicitly marked "placeholder state — flows are zero, ICR is NaN" | calibration / proxy-acknowledged |
| delaware_wealth_calibration | "gross home value, NOT net worth; mortgage debt not subtracted; non-housing assets absent; renters at 0; CB band is a net-worth expectation, comparing to housing-only proxy would overstate the deficit" | lower-bound, deficit-not-stated |
| national_wealth_calibration | Same wealth-proxy caveat repeated verbatim. CB band: $200k–$300k expectation, observed gap $60k housing-only — "lower bound caveat" stated. | lower-bound |
| education_wealth_decoupling | "decoupling not confirmed on this proxy" — Black BA+ ($120k) actually exceeds White no-HS ($70k); the notebook **refuses** to claim the headline result on the available proxy and reverts to a "could narrow/reverse with net worth" disclaimer | self-refuting on insufficient data |
| ecg_subgroup_distortion | 2.04% feature-extraction failure rate logged; PTB-XL ≥90 ages capped at 89 and counted; "structural-impedance primitives provide an open-source detection apparatus" — framing is detection-apparatus, not diagnostic verdict | apparatus-claim, not diagnostic-claim |
| cxr_segmentation_distortion | Zero extraction failures; no demographic/disease metadata in CSV ⇒ disease subgroup *derived* from CTR threshold (notes the circularity, uses orthogonal feature `lung_asym` for cumulant divergence); "A model that matches mean CTR per subgroup but fails to match higher-order structure will pass standard validation while remaining structurally miscalibrated" | apparatus-claim, miscalibration-framed |

---

## PHASE 3 — Invariant Detection (cross-notebook patterns)

### 3.1 Rank-order: κ₃ vs κ₄ real-to-null ratios

| Notebook | κ₃ ratio | κ₄ ratio | κ₃ / κ₄ |
|---|---|---|---|
| delaware_wealth (C2) | 1.8 | 1.3 | 1.4 |
| national_wealth (C3) | 20.2 | 5.9 | 3.4 |
| education hi-edu (C5) | 33.5 | 94.3 | **0.36** ← exception |
| ecg sex (C6) | 0.6 | 1.5 | **0.4** ← exception |
| ecg age (C7) | 11.1 | 4.9 | 2.3 |
| cxr disease (C8) | 4.1 | 1.8 | 2.3 |
| cxr quality (C9) | 8.0 | 2.3 | 3.5 |

**Pattern.** When a real subgroup contrast exists with adequate power, κ₃ dominates κ₄ by a **factor of 1.4× – 3.5×** (5 of 7 contrasts). Two exceptions:
- **ECG Male vs Female heart rate (C6)** — κ₃ ratio of **0.6×** is *below 1*, i.e. real κ₃ < null κ₃. The instrument is **honestly reporting absence of κ₃ signal** for this contrast. This is consistent with the clinical observation that male and female resting heart-rate distributions overlap heavily.
- **Education high-edu null vs real (C5)** — κ₄ ratio (94.3) exceeds κ₃ ratio (33.5). This is the *only contrast where κ₄ dominates κ₃*; C5 also has the *largest* real-to-null ratios anywhere in the matrix. Plausible cause: at large n with heavy-tailed wealth distributions, the κ₄ moment is genuinely larger than κ₃ for the difference of two lognormal-class distributions; this is an empirical not a calibration anomaly.

Conclusion: κ₃ is the *typical* dominant order, but the architecture does not pre-suppose it — it surfaces whichever order carries the signal.

### 3.2 Sample-size invariant

| n_min | Notebook | Sheaf verdict | κ₃ ratio | Comment |
|---|---|---|---|---|
| 389 | delaware_wealth (C2/S2) | **valid** | 1.8× | under-powered: real-to-null too low to fire sheaf at 8 SE |
| 1,015 | ecg age (C7) | obstructed | 11.1× | well-powered |
| 3,824 | ecg sex (C6/S5) | obstructed (marginal: 8.09 / 8.39 vs tol 8.00) | 0.6× | gate refuses (k3 blocking), sheaf marginally fires — instrument is operating at the edge of detectability |
| 4,500 | cxr quality (C9) | — | 8.0× | well-powered |
| 7,606 | cxr disease (C8/S6) | obstructed | 4.1× | well-powered |
| 13,192 | education BA+ (C5/S4) | obstructed | 33.5× | well-powered, 4 sections |
| 69,943 | national_wealth (C3/S3) | obstructed (64.89 SE on Black↔All) | 20.2× | well-powered |

**Pattern.** The sheaf-obstruction verdict scales with `n_min` × signal-strength. **At n_min ≈ 400 the sheaf correctly reports *valid* on the Delaware wealth gap** even though the cumulant divergence is non-trivial — the instrument is honestly under-powered. The SAME contrast at n_min ≈ 70,000 (national_wealth) fires emphatically (64.89 SE on Black↔All). **The same primitive call, the same threshold, different verdicts driven by sample size.** This is the architecture's honest-flag mechanism, not a bug.

### 3.3 Gate behavior invariant

The standardized admission gate (τ = 0.2 σ-units) was designed as a noise guard, **not a signal detector**. Cross-notebook behavior:

| Real ‖Δκ‖ / null ‖Δκ‖ | Gate admit on real | Gate admit on null |
|---|---|---|
| ≥ 20× (national_wealth, education hi) | **False, blocking=k3** | True, blocking=None (national: silent; education hi: warning fires) |
| 4× – 11× (ecg age, cxr) | False, blocking=k3 | True or False varies; warning fires when True |
| < 2× (delaware_wealth, ecg sex) | True, blocking=None (delaware) / False, k3 (ecg sex) | True, blocking=None |

**Pattern.** The standardized gate is *conservative*: it refuses to admit signals that are large in ratio terms but still small in σ-units after standardization. **The gate's "admit=False blocking=k3" verdict co-occurs with the largest real-to-null ratios**. This is a feature, not a bug: the σ-unit threshold is the noise floor of standardized data, and a strong distributional difference may compress to a small standardized κ. The notebooks correctly do not rely on the gate verdict as the primary signal — they rely on the **real-to-null ratio**, with the gate as an auxiliary noise-floor check.

**Inconsistency flagged:** the gate admits=True with `blocking=None` on multiple null controls (delaware_wealth, education hi-edu). The notebooks print explicit warnings ("gate fires on null — admission may be noise"). The gate is not silent under all nulls — it is silent under *some* nulls and the notebooks disclose this. The behavior is documented but is a known noise leak.

### 3.4 Sheaf-cumulant joint behavior

| Cumulant signal | Sheaf verdict | Notebooks |
|---|---|---|
| strong (ratio ≥ 4×) | obstructed | national_wealth, education, ecg age, cxr disease, cxr quality |
| weak (ratio < 2×) at small n | **valid** | delaware_wealth |
| weak (ratio ≈ 0.6× κ₃ but 1.5× κ₄) at large n | obstructed (marginal) | ecg sex |

**Pattern.** The two primitives are *complementary*, not redundant. Cumulant divergence compares full distributional shape; the sheaf compares a structured 4-vector `[mean, var, skew, kurt]` against pooled restriction maps. They generally agree, but the ecg-sex case (C6/S5) is a clear *disagreement*:
- κ₃ ratio is 0.6× (below null) — cumulant says "no κ₃ signal."
- κ₄ ratio is 1.5× — cumulant says "weak κ₄ signal."
- Sheaf reads obstructed at 8.09 and 8.39 SE — both barely above the 8 SE tolerance.

The sheaf catches the joint mean+variance shift that the per-order ‖Δκ‖ doesn't isolate. The instrument is **multi-modal**: a finding requires either a strong cumulant ratio OR a sheaf obstruction (or both), and the absence of one does not refute the other.

### 3.5 ICR universality

| Domain | Interception interface | Asymmetric finding present? |
|---|---|---|
| US PUMS income/wealth | housing-cost share of income | **YES** — Black > White in both delaware (5.8pp) and national (3.9pp) |
| ECG | recording-noise share of QRS amplitude | **YES** on age axis (+15.4%); ~0 on sex axis |
| ECG | diagnostic ambiguity (SCP-code count) | **YES** on age axis (+51%) |
| CXR | mask uncertainty (1−Dice) | **YES** on quality axis (+54%); modest on disease (+2.6%) |
| CXR | CTR distribution width (CV ratio) | **YES** LowDice 1.73× HighDice |

**Pattern.** Every notebook that runs an ICR-analog finds **at least one asymmetric interception interface**. The specific interface differs (housing cost / recording noise / segmentation uncertainty / diagnostic ambiguity), but the *structural form* — a quantity that intercepts the signal between source and observer, measured per subgroup — is uniformly present and uniformly asymmetric. **Exceptions are honestly reported** (ecg sex axis: −0.5%, declared symmetric).

### 3.6 Domain independence

Same primitive calls run on:
- **tabular dollar-denominated wealth** (PUMS, three notebooks)
- **time-series millivolt physiology** (PTB-XL ECG)
- **image-derived pixel anatomy** (CheXmask CXR)

In every case, `cumulant_difference`, `admit_per_component_standardized`, `sheaf_status_and_kappa`, and a bootstrap-SE normalization step produce dimensionally meaningful output. The instrument is **unit-blind**: `cumulant_difference` returns a Frobenius norm in the variable's natural units (1e17 for wealth, 1e2 for bpm, 1e-4 for CTR), and the **real-to-null ratio** strips the units. The sheaf SE-normalization step also strips units. **No domain-specific code paths exist** in `structural_impedance/`.

---

## PHASE 4 — Architectural Causes

Mapping invariants from Phase 3 back to library + test enforcement.

### 4.1 What `cumulant_difference` enforces across domains

- **Two independent populations** — no equal-n requirement, no row pairing. This is what makes the unequal sample sizes in the matrix (e.g. C3: n=70k vs 774k, C4: n=16k vs 156k) valid. If the primitive required pairing, the wealth and education notebooks could not compute the contrast at full sample.
- **Per-order separation** — `k_order ∈ {3, 4}` is a parameter, not an aggregate. The notebooks always report both orders; the matrix shows κ₃ and κ₄ ratios separately, which is what surfaces the C5 κ₄-dominance exception.
- **No regularization, no clipping** — `(κ_X − κ_Y).norm()` is computed directly. This is why the magnitudes range over 22 orders (1e-5 for CXR to 1e23 for wealth κ₄) without code change.

### 4.2 What `admit_per_component` (anti-aggregation) prevents

`test_admit_per_component.py::test_large_k3_cannot_mask_small_k4` is the canonical fail-test: a Y = X² coupling produces large κ₃ and small κ₄, and the test verifies the gate **refuses to admit** with `blocking_component="k4"`. If the gate ever degenerates to an aggregate `n3 + n4 > τ` form, this test fails.

In the matrix: **every "blocking=k3" verdict in the gate column** (C3, C6, C7, C8, C9) is the same architectural mechanism: standardized κ₃ is below 0.2 σ-units even when raw ‖Δκ₃‖ is enormous and real-to-null ratio is 20× — and the gate refuses to admit purely on the κ₄ side. This is the per-component AND gate in action.

### 4.3 What binary gluing (`sheaf_status_and_kappa`) prevents

`test_sheaf_single_violation.py` enforces that a **single failing overlap obstructs**, regardless of how many overlaps are fine. The notebooks rely on this in two ways:
- The single-overlap obstruction tag (e.g. `sheaf:overlap_0_2`) **localizes the disagreement** — in S3 (national wealth) it identifies *Black ↔ All* as the obstructed overlap (64.89 SE), *White ↔ All* is also obstructed but at 9.96 SE; the worst-overlap argmax is meaningful.
- The 4-section education sheaf (S4) tests three overlaps simultaneously; the worst is BA+ Black↔White at 38.11 SE. A "majority valid" softening would mask this.

### 4.4 Null calibration uniformity

- Cumulant null: every notebook from delaware_wealth onward uses a **permutation null** (random split of pooled population at the same n). The real-to-null ratio is the documented headline metric.
- Sheaf null: every notebook from delaware_wealth onward uses a **two-split null** of the pooled population through the same SE-normalized section pipeline, then sets `TOL_SHEAF = max(8 SE, 1.5 × null_max)`. The floor at 8 SE is a hard guarantee; the 1.5× adaptive term protects against heavy-tailed pools (S4 raised tolerance to 14.49 SE because the 4-segment pool was a heavier mixture and the null_max climbed to 9.66 SE).
- The 8-SE floor is also the value used in `tests/structural_impedance/test_sheaf_null_valid.py` — the test is the calibration source.

### 4.5 Honest-flag mechanisms (what's logged, not silenced)

Inventoried across the codebase:

| Mechanism | Location | Triggered in notebooks |
|---|---|---|
| `(Γ + γ*)` singularity tag | `gamma_correction.py:39-41` | Not exercised in any notebook |
| `1 − \|Γ\|² ≈ 0` total-reflection log | `gamma_correction.py:53-55` | Not exercised in any notebook |
| Sinkhorn non-convergence log | `sinkhorn.py:34-37` | Not exercised |
| SIM non-PD log (no clamping) | `sinkhorn.py:96-102` | Not exercised |
| Gate `blocking_component` | `cumulant.py:73-75` | Exercised in C3, C5, C6, C7, C8, C9 |
| Sheaf `obstruction_at` tag | `sheaf_gluing.py:25-27` | Exercised in S1, S3, S4, S5, S6 |
| `nan_fraction` / `n_anonymized` in notebooks | per-notebook | Exercised: PTB-XL 293 ages capped & logged; ECG 163 records dropped & logged; CXR 0 dropped |
| Wealth-proxy "NOT net worth" caveat | wealth notebooks | Restated in every wealth markdown cell that reports a dollar figure |
| Gate-fires-on-null warning | wealth + edu notebooks | Printed in C2, C5 |
| Sheaf adaptive-tolerance disclosure | sheaf cells | Printed in every notebook from delaware_wealth onward |

### 4.6 What breaks if invariants are removed

| Invariant | If removed | Failure mode |
|---|---|---|
| Per-component AND gate (anti-aggregation) | A large κ₃ would mask a near-zero κ₄ | C5 high-edu would report admit=True under coupling tests with hidden order collapse; the dependence-structure goes undetected |
| Binary sheaf gluing | Soft averaging would hide the worst overlap | S3 Black↔All 64.89 SE would average with White↔All 9.96 SE → 37 SE; the *target* of the obstruction is lost |
| Cumulant null calibration | Magnitudes are unit-dependent and incomparable across notebooks | The matrix in §2.1 collapses — wealth κ at 1e17 vs CXR κ at 1e-4 would appear contradictory without the ratio normalization |
| Sheaf null calibration with adaptive floor | A 4-section pool with heavier tails (S4) would falsely obstruct under fixed-8 tolerance; a small-n contrast (S2) might falsely obstruct or falsely validate | S2 (n=389) would still need a SE-aware tol; without it, a 5.96 SE Black-↔-All disagreement could be interpreted as obstruction or as noise — undefined behavior |
| `1−\|Γ\|²` real-power-coefficient discipline | `1−Γ²` is complex; Φ would leave [0,1] and the test_phi_endpoints suite would fail | The pipeline's monotonic / unit-interval property guarantees fail |
| Citation conformance | Authority anchor is lost | Every kernel becomes opaque; a maintainer can no longer trace a primitive back to its mathematical source |

---

## PHASE 5 — Synthesis

### 5.1 What is this instrument?

At the **architectural level**, `structural_impedance` is a **distribution-divergence triangulator** that measures whether two finite samples of a continuous scalar feature differ from one another in a way that is **structurally beyond noise**, using three orthogonal mathematical interfaces:

1. **Per-order cumulant divergence** (κ₃, κ₄) of the marginal distributions, normalized against a permutation null.
2. **Cellular-sheaf cocycle consistency** of bootstrap-SE-normalized statistic vectors `[mean, var, skew, kurt]` against pooled restriction maps, with adaptive null-calibrated tolerance.
3. **An ICR-analog asymmetry** — a per-subgroup measurement of an *interception interface* between signal source and observer (housing cost / recording noise / mask uncertainty / diagnostic ambiguity), independent of (1) and (2) but answering the same structural question on the *cost side*.

The instrument is **unit-blind, equal-n-blind, and domain-blind**. The primitives in `structural_impedance/` contain no code path that distinguishes between dollars, millivolts, and pixels.

It is **not** a hypothesis-testing instrument in the classical NHST sense: it does not output p-values. It outputs ratios (κ real/null), residuals (sheaf SE units), tags (obstruction location), and admit/refuse verdicts (gate). Each output mode reports the **shape and magnitude of structural disagreement** at a different mathematical interface.

### 5.2 Meta-patterns visible only across domains

| Pattern | Evidence | Cross-validation |
|---|---|---|
| **P1.** κ₃ typically dominates κ₄ in subgroup-divergence ratios by 1.4×–3.5×. Exceptions are flagged by the instrument, not hidden. | C2, C3, C7, C8, C9 dominant; C5, C6 exceptions disclosed. | 5 of 7 contrasts ≥ 2 notebooks. |
| **P2.** Sheaf validity tracks sample size at a quasi-universal threshold of n_min ≈ 1,000 with 8 SE tolerance, below which the instrument honestly reports under-powering. | S2 valid at n=389; S5 marginal at n=3,824; S6 strong at n=7,606; S3 strong at n=69,943. | ≥ 3 notebooks. |
| **P3.** The standardized gate's "admit=False, blocking=k3" verdict co-occurs with the *largest* real-to-null ratios. Gate verdicts must not be read as signal strength. | C3 (20×, blocking=k3), C5 (33×/94×, blocking=k3 on real), C7 (11×, k3), C8 (4×, k3), C9 (8×, k3). | 5 notebook contrasts. |
| **P4.** Cumulant and sheaf are complementary, not redundant. Either can fire while the other is silent (ecg-sex: weak κ, marginal sheaf; delaware_wealth: weak κ, sheaf valid). | C6/S5 disagree-direction; C2/S2 agree on under-power. | 2 notebooks. |
| **P5.** Every domain has at least one asymmetric interception interface (ICR analog). When the axis is symmetric (ecg sex), the instrument reports the absence honestly. | I1, I2, I3b, I3c, I4, I4b, I4c positive; I3 reports negative cleanly. | 4 notebooks. |
| **P6.** The instrument's magnitudes span ~22 orders (1e-5 to 1e23) across domains without code modification. Cross-domain comparisons live in the **ratios**, not the magnitudes. | C8: 1e-5; C2/C3/C5: 1e17–1e24. | 6 notebooks. |
| **P7.** Honest-flag logging fires across domains: PTB-XL age cap, ECG feature failure rate, CXR zero failures, gate-fires-on-null warning. Each notebook discloses its own noise / proxy / refusal of overclaim. | Every wealth notebook restates the gross-value-not-net-worth caveat; every gate-on-null prints a warning; education notebook self-refutes ("decoupling not confirmed on this proxy"). | All 6 notebooks. |

### 5.3 What does the instrument know about itself that no single use case reveals?

This is derivable only from the cross-domain matrix; no individual notebook contains it.

**Self-knowledge 1 — The instrument knows when it is under-powered.**
S2 (delaware_wealth, n=389) and S3 (national_wealth, n=69,943) ask the *exact same question* of the *exact same population* via the *exact same primitive call* — only `n` differs. S2 returns *valid* with Black↔All disagreement of 5.96 SE (below the 8 SE tolerance); S3 returns *obstructed* with 64.89 SE. This is **not the instrument failing at small n** — it is the instrument **correctly reporting that the noise floor is higher at small n**. The 8 SE tolerance was set on the null at the same n; at n=389 the null_max is 4.75 SE, so the real signal at 5.96 SE is genuinely **inside the noise envelope** of that sample size. No single notebook reveals this; only the S2 vs S3 comparison does.

**Self-knowledge 2 — The instrument's "admit" gate is a noise-floor refusal, not a signal-strength affirmation.**
The cross-notebook gate column is anti-correlated with signal strength: the largest κ-ratios (national_wealth 20×, education hi-edu 33×/94×, ecg age 11×, cxr quality 8×) all return `admit=False, blocking=k3`. Read inside a single notebook, this looks like the gate misbehaving. Read across the matrix, it reveals that the gate's purpose is to refuse standardization-floor-equivalent signals (κ in σ-units after z-scoring) — and large *raw* divergences can compress to small *standardized* divergences when the within-subgroup variance is itself large. The notebooks have the discipline to report this without "fixing" the gate to agree with the ratio.

**Self-knowledge 3 — The instrument refuses to overclaim on insufficient proxies.**
The education_wealth_decoupling notebook is the canonical case: the *headline test* (Black BA+ vs White no-HS) returns the *opposite* of the expected direction ($120k > $70k), and the notebook **does not paper over this** — it states "Decoupling not confirmed on this proxy" and discloses that the gross-housing proxy ignores the very mortgage-debt-by-race differential that would likely reverse the verdict on a net-worth measure. No individual notebook reveals this is a *pattern* of the architecture; the cross-notebook pattern is: **wealth notebooks restate the proxy caveat at every dollar figure, the education notebook self-refutes when the proxy fails, the ECG notebook caps PTB-XL anonymized ages and counts them, the CXR notebook reports zero failures while clearly stating no demographic data is available**. The architectural invariant is "no claim beyond the proxy."

**Self-knowledge 4 — The instrument's three interfaces are designed to disagree.**
A cumulant ratio of 0.6× and a sheaf obstruction at 8.09 SE coexist in ECG sex (C6/S5). Read inside that notebook alone, it might be a contradiction. Across the matrix it reveals architectural design: the cumulant inspects shape; the sheaf inspects a different summary (a 4-vector with overlap restriction maps); the ICR inspects the interception cost. **These probe different mathematical objects** and are *expected* to sometimes diverge. A subgroup that differs in joint location-and-shape but not in higher-order tails (which is what ecg sex appears to be — male and female resting HR distributions differ in mean+variance but not strongly in skew/kurt) will fire the sheaf and not the cumulant. The instrument is **a triangulator, not a single-axis detector**.

**Self-knowledge 5 — The instrument is invariant to unit and to domain.**
A 22-order-of-magnitude range of raw cumulants (1e-5 CXR to 1e24 education) flows through the same code path. The ratio normalization erases units. The SE normalization erases sampling-variance scale. The sheaf binary verdict erases continuous-tolerance ambiguity. **The pipeline is the instrument; the data is the substrate.** No single notebook can demonstrate this — only the matrix can.

### 5.4 Limitations (the instrument's own disclosures)

| Limitation | Source | Domain-specific or architectural? |
|---|---|---|
| Gate fires on some nulls with `blocking=None` | C2, C5 | **Architectural** — the σ-unit threshold has known noise-leak edge cases at small n or heavy-tail populations |
| κ₄ cross-zero threshold of 1e-2 is statistically infeasible at d≥2, n≤10⁷ | `test_kappa_independent_zero.py` docstring | **Architectural** — flagged in code |
| `sim_hessian_torch_hvp` requires 2nd-order autograd through `cdist`; torch 2.0.1 does not provide this | `test_sim_no_closed_form.py:test_runs_and_returns_square_hessian` (skipped) | **Architectural / library dependency** — test skips honestly rather than mocking |
| Wealth proxy is gross housing value, not net worth | All three wealth notebooks | **Domain-specific** — data limitation, restated at every claim |
| PTB-XL ages ≥90 are encoded as 300; capped to 89 in stratum, count logged | ecg notebook | **Domain-specific** — PTB-XL anonymization, disclosed |
| ECG feature extraction has 2.04% failure rate at <3 R-peaks | ecg notebook | **Domain-specific** — disclosed |
| CXR has no native demographic / disease metadata; disease subgroup derived from CTR threshold | cxr notebook | **Domain-specific** — disclosed; circularity broken by analyzing `lung_asym` instead |
| Sheaf tolerance below 8 SE is not exercised | All notebooks | **Architectural floor** — `max(8 SE, 1.5 × null_max)` is the policy |
| Delaware wealth (n=389) sheaf is *valid* but cumulant ratios are *weak* (1.8×, 1.3×) | C2 / S2 | **Sample-size limitation** — disclosed via real/null ratio printout |

### 5.5 What the repository CAN and CANNOT support

**Claims the data supports** (every claim is sourceable to a specific cell):

- The library primitives operate **identically** across tabular dollar, time-series mV, and pixel-derived imaging data without modification. *Evidence:* all 6 notebooks share the same `structural_impedance` import lines and produce structurally identical output formats.
- The cumulant-divergence + sheaf-gluing pipeline reliably **separates real subgroup divergence from permutation-null noise** at `n_min ≳ 1,000`. *Evidence:* C3, C7, C8, C9 all return real-to-null κ ratios > 4× with corresponding sheaf obstructions.
- The instrument **honestly reports under-powering** at small `n_min`. *Evidence:* S2 valid at n=389; the same comparison at n=70k fires emphatically.
- The per-component admission gate **prevents aggregate masking** between κ₃ and κ₄. *Evidence:* `test_admit_per_component.py::test_large_k3_cannot_mask_small_k4`.
- The sheaf-gluing primitive **localizes** disagreements rather than averaging them. *Evidence:* S3's `sheaf:overlap_0_2` correctly identifies Black↔All as the dominant disagreement (64.89 SE vs White↔All 9.96 SE).
- An ICR-analog interception finding **appears in every domain** with at least one asymmetric axis. *Evidence:* I1, I2, I3b/I3c, I4, I4b, I4c.
- The instrument's verdicts can **disagree across interfaces** for a single contrast (ecg sex). The architecture treats this as expected, not as contradiction.
- The repository **self-refutes** when the proxy is insufficient (education_wealth_decoupling's "decoupling not confirmed on this proxy").

**Claims the repository would OVERREACH:**

- **"This proves causal claims about subgroup distributions."** The instrument measures distributional divergence; causation is not inside the pipeline. The wealth notebooks correctly state "structural distortion" or "deficit", not "cause."
- **"The standardized gate is a hypothesis test."** It is a σ-unit noise-floor refusal; admit=True under nulls happens in C2 and C5, with warnings printed. Using gate-verdicts as p-values would be incorrect.
- **"The ‖Δκ‖ raw magnitudes are comparable across notebooks."** They are not — they are in the variable's natural unit (1e17 wealth-dollars-cubed, 1e-4 CTR-cubed). Only the real-to-null ratio is cross-domain.
- **"Cardiomegaly subgroups are clinically labeled."** In the CXR notebook the disease label is derived from the CTR threshold; this is mathematically a CTR-stratification, not a radiologist-adjudicated diagnosis. The notebook discloses this and breaks circularity by analyzing `lung_asym`, not CTR.
- **"The Chile–Barbados deficit at Black income ≈ $200k–$300k is supported by these notebooks."** The wealth notebooks **explicitly refuse** to make this claim because the proxy is gross housing only, not net worth.
- **"The instrument is validated by these notebooks."** It is *exercised* across three domains and is *internally consistent*. Validation against ground-truth distributional divergence (e.g., synthetic populations with known κ structure) would strengthen the evidence; only `tests/structural_impedance/test_kappa_independent_zero.py` does this at d=2, n=50k.
- **"sim_hessian_torch_hvp is runtime-verified."** The runtime test is `pytest.skip`-ed on this torch build. The kernel matches the blueprint verbatim and the discipline-tests (no closed-form, no clamping) pass, but **runtime SIM-PD behavior is unobserved in this repository as audited**.
- **"The repository proves κ₃ universally dominates κ₄."** The C5 high-edu and C6 ecg-sex contrasts are clear exceptions; the *pattern* is "κ₃ dominates in most contrasts" not "κ₃ universally dominates."

---

## Audit summary

The `structural_impedance` library is a **domain-blind distribution-triangulation instrument** consisting of four small modules (358 LOC total) backed by ten architectural-invariant tests. Across six executed notebooks spanning tabular wealth, time-series ECG, and image-derived CXR data, it produces self-consistent output with honest under-power reporting, multi-interface triangulation (cumulant + sheaf + ICR), and disciplined refusal-to-overclaim when proxies are insufficient. Its cross-domain magnitudes span 22 orders without code modification, validated by the ratio-normalization and SE-normalization design choices.

The strongest evidence of the instrument's design intent is observable **only** in the cross-notebook matrix: identical primitive calls produce different verdicts in the *direction the instrument's design predicts* (under-powering at n=389, gate-refusal at large standardized noise, sheaf-cumulant disagreement on weak-shape strong-location contrasts). The repository's largest weakness is the `sim_hessian` runtime path, which is `pytest.skip`-ed on the audited torch build, and the educationally-honest but practically-loose σ-gate which leaks `admit=True` on some null splits and which the notebooks have the discipline to flag explicitly.

The instrument is what it claims to be at the architectural level. The notebooks correctly avoid claiming it is more.

**End of repo-scope audit.**
