# structural_impedance

Aggregate metrics mask distributional divergence. `structural_impedance` detects this masked divergence using per-component triangulation across three independent mathematical interfaces: cumulant decomposition, sheaf-theoretic gluing, and impedance-ratio measurement. Unlike standard aggregate-metric approaches, the tool is domain-blind and unit-blind, demonstrated across ECG, chest radiography, and census wealth.

**358 LOC of math. 544 LOC of audit infrastructure. 14 architectural tests. 6 cross-domain notebooks.**

## Install

```bash
git clone https://github.com/obra-foundry/ots
cd ots
pip install -e .
```

## Use

```python
import structural_impedance as si
import torch

X = torch.randn(500, 1, dtype=torch.float64)
Y = torch.randn(500, 1, dtype=torch.float64) + 0.5

d = si.cumulant_difference(X, Y, k_order=3)
admit, meta = si.admit_per_component_standardized(X, Y)
```

```python
finding = si.CumulantFinding(
    contrast_name="Group A vs Group B",
    feature_name="resting heart rate (bpm)",
    k_order=3, n_x=500, n_y=500,
    real_value=float(d), null_value=float(d) / 5.0,
    gate_admit=admit, blocking_component=meta.get("blocking_component"),
)

print(finding.format_for("stripped"))
# -> "On resting heart rate (bpm), the third-moment difference between
#     Group A vs Group B (n=500 vs 500) is 5.0 times the same-population
#     noise floor. Standard mean and variance comparison cannot detect
#     this; the discriminative structure lives at higher distributional
#     moments."
```

## Five axioms

1. `Γ = (Z_recv − Z_emit) / (Z_recv + Z_emit)` governs signal transfer integrity.
2. A target impedance `γ*` exists for every receiver topology.
3. Cumulant non-additivity prevents Goodhart regression at orders k ≥ 3.
4. Every primitive accepts an optional `gamma_star`; absent returns a natural-Γ estimate.
5. Composite operations return a sheaf-gluing status and a residual κ vector. Never a scalar aggregate.

Test-enforced. See `tests/structural_impedance/`.

## Modules

| Module | LOC | Purpose |
| --- | --- | --- |
| `gamma_correction.py` | 75 | Reflection coefficient, Φ modulation, ε-Γ gating, kurtosis coupling |
| `cumulant.py` | 119 | Per-order cumulant tensors, per-component admission AND-gate, marginal κ difference |
| `sheaf_gluing.py` | 27 | Cocycle disagreement, binary obstruction verdict with worst-overlap tag |
| `sinkhorn.py` | 114 | Log-space potentials, implicit-diff SIM Hessian, per-component κ residuals |
| `provenance.py` | 139 | Axiom 4 back-pointer registry for every public kernel |
| `findings.py` | 173 | Typed result objects with `format_for(overt \| stripped)` adapters |
| `transmission_diagnostic.py` | 116 | Three automated failure-mode checks |
| `reclamation.py` | 116 | Round-trip integrity verifier |

## Cross-domain demonstrations

Same primitives, three substrates, 22 orders of magnitude of cumulant range, no domain-specific code paths.

| Notebook | Domain | n | Headline |
| --- | --- | --- | --- |
| `examples/ecg_subgroup_distortion.ipynb` | PTB-XL ECG | 21,799 records | κ₃ Young vs Elderly **11.1×**; sheaf marginal on sex (under-power honestly reported) |
| `examples/cxr_segmentation_distortion.ipynb` | CheXmask VinDr-CXR | 18,000 images | CTR-CV ratio **1.73×** across mask-quality strata |
| `examples/delaware_wealth_calibration.ipynb` | US Census PUMS wealth | 4,765 HHs | sheaf valid at n=389 (instrument honestly reports its own under-powering) |
| `examples/delaware_calibration.ipynb` | US Census PUMS income | 9,876 | calibration baseline; per-component admission demonstrated |
| `examples/national_wealth_calibration.ipynb` | US Census PUMS wealth | 1.45 M HHs | κ₃ real/null **20.2×**; sheaf obstructed 64.89 SE |
| `examples/education_wealth_decoupling.ipynb` | US Census PUMS × education | 3.4 M persons | κ₃ persistence ratio **18.2×** (high/low edu) |

Every notebook ships executed. Outputs are in the `.ipynb` cells.

## What this is NOT

* Not a hypothesis test. No p-values. Outputs are ratios, residuals, tags, binary verdicts.
* Not a causal-inference instrument. Measures distributional divergence; causation is upstream.
* Not a scalar aggregate metric. Per-component returns are test-enforced; aggregate composition is forbidden.
* Not a clinical diagnosis. The CXR cardiomegaly subgroup is derived from the CTR threshold; this is a CTR-stratification, not a radiologist label. Disclosed in notebook.
* Not the upstream architecture. This is the externalized math kernel plus its audit infrastructure. The upstream substrate (CGF, Dokimics, B-filter, AIR, TCP, Hill ξ, Fisher-Rao) is intact upstream and not in this repo.

## Audit

`AUDIT.md` at repo root. 557 lines, eight phases:

* Phase 1–5: repo-scope audit (library, tests, notebooks, cross-validation, synthesis).
* Phase 6–8: substrate-lineage placement (correspondence to the upstream framework corpus; non-linear trajectory; recursive integration roadmap).

Independent third-party audit reproducible from the executed notebook outputs alone.

## Documentation standard

**CODE-U** (*Content Origin Documentation & Encoding — Unified*): the four-axiom documentation standard this repository follows. All public artifacts are compiled binaries; all generative derivations are provenance-registered.

| Axiom | Status | Evidence |
| --- | --- | --- |
| 1 Source/Binary Distinction | ✓ | Library executes without upstream substrate installation |
| 2 Implementation Precedes Publication | ✓ | Operator metabolized 22 years (2004 origin → 2026 externalization) |
| 3 Minimal Necessary Specification | ✓ | `format_for(overt \| stripped)` two-register output |
| 4 Provenance Irreversibility | ✓ | `provenance.PROVENANCE` registry; back-pointer for every public kernel |

## Tests

```bash
pytest tests/structural_impedance/ -q
# 46 passed, 1 skipped in 2.51s
```

The skipped test is `test_sim_no_closed_form::test_runs_and_returns_square_hessian`, which requires second-order autograd through `torch.cdist` (not available in `torch==2.0.1`). The discipline-tests for `sim_hessian_torch_hvp` (no closed-form assembly, no PSD clamping) pass.

## Citations

Every public kernel carries its anchor in the docstring and in `provenance.PROVENANCE`.

* Sturmfels, B., & Zwiernik, P. *Binary cumulant varieties.* arXiv:1011.1722.
* Curry, J. *Sheaves, Cosheaves and Applications.* PhD thesis, 2014.
* Feydy, J., Séjourné, T., Vialard, F.-X., Amari, S., Trouvé, A., Peyré, G. *Interpolating between Optimal Transport and MMD using Sinkhorn Divergences.* AISTATS 2019.
* Shen, Z., Feydy, J., Liu, P., Curiale, A., San José Estépar, R., San José Estépar, R., Niethammer, M. *Accurate Point Cloud Registration with Robust Optimal Transport.* NeurIPS 2020. Prop. 5.1 Eq. 17.
* Pozar, D. *Microwave Engineering* 4e. Wiley. Chapter 2.
* Chazal, F., de Silva, V., Glisse, M., Oudot, S. *The Structure and Stability of Persistence Modules.* arXiv:1207.3885.

## License

AGPL-3.0-or-later. See `LICENSE`.

AGPL was chosen deliberately. The licensing philosophy is anti-capture: derivatives must remain open, network use counts as distribution (§13), and SaaS extraction without source release is forbidden. If that license is incompatible with your use case, your use case is incompatible with this work.

## Author

Obra Foundry Core Maintainers.

## Provenance

Conceptual lineage: aggregate evaluation metrics (mean, variance, AUC) are blind to distributional distortion at cumulant orders k ≥ 3. When subgroups within a population carry different higher-moment structure, standard reporting masks the divergence. The primitives in this library isolate that divergence through three independent mathematical interfaces: cumulant decomposition detects the distortion, sheaf-theoretic gluing verifies that local subgroup distributions cannot be consistently reconciled with the pooled aggregate, and the impedance metaphor quantifies the fraction of signal intercepted at the transmission interface before it reaches the compounding load. The anti-aggregation discipline (per-component returns, no scalar summaries, binary obstruction verdicts) prevents the tool itself from reproducing the masking it was built to detect.

Every public kernel carries an inspectable back-pointer to its derivation chain:

```python
import structural_impedance as si

for line in si.derivation_chain(si.cumulant_difference):
    print(line)
```
