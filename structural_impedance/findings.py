"""CODE-U Step 3 Gradient Encoding Decision.

Finding dataclasses carry results plus a format_for() adapter that emits in
two registers:

  overt      substrate-class internal vocabulary (framework-native)
  stripped   professional / clinical / external first-contact

Conformance: 1164.CODE-U.1.0 §1 Step 3; CLAUDE.md §4.13.
"""
from dataclasses import dataclass
from typing import Literal, Optional

Audience = Literal["overt", "stripped"]


@dataclass(frozen=True)
class CumulantFinding:
    contrast_name: str
    feature_name: str
    k_order: int
    n_x: int
    n_y: int
    real_value: float
    null_value: float
    gate_admit: bool
    blocking_component: Optional[str]
    provenance_chain_id: str = "ADT-v1.0::marginal-cumulant-divergence"

    @property
    def ratio(self) -> float:
        return self.real_value / max(self.null_value, 1e-12)

    def format_for(self, audience: Audience) -> str:
        if audience == "overt":
            return (
                f"||Delta kappa_{self.k_order}|| {self.contrast_name} on "
                f"{self.feature_name}: real={self.real_value:.3e}, "
                f"null={self.null_value:.3e}, ratio={self.ratio:.1f}x. "
                f"n_x={self.n_x:,} n_y={self.n_y:,}. "
                f"Gate admit={self.gate_admit} blocking={self.blocking_component}. "
                f"Chain: {self.provenance_chain_id}."
            )
        order = "third-moment" if self.k_order == 3 else "fourth-moment"
        return (
            f"On {self.feature_name}, the {order} difference between "
            f"{self.contrast_name} (n={self.n_x:,} vs {self.n_y:,}) is "
            f"{self.ratio:.1f} times the same-population noise floor. Standard "
            f"mean and variance comparison cannot detect this; the discriminative "
            f"structure lives at higher distributional moments."
        )


@dataclass(frozen=True)
class SheafFinding:
    contrast_name: str
    feature_name: str
    section_labels: tuple
    overlaps: tuple
    per_overlap_se: tuple
    tol_se: float
    null_max_se: float
    status: str
    obstruction_at: Optional[str]
    provenance_chain_id: str = "ADT-v1.0::binary-gluing"

    def format_for(self, audience: Audience) -> str:
        if audience == "overt":
            rows = [
                f"{self.section_labels[i]}<->{self.section_labels[j]}: {d:.2f} SE "
                f"[{'OBSTRUCTED' if d >= self.tol_se else 'valid'}]"
                for (i, j), d in zip(self.overlaps, self.per_overlap_se)
            ]
            return (
                f"Sheaf {self.contrast_name} on {self.feature_name}: "
                f"status={self.status}, tol={self.tol_se:.2f} SE "
                f"(null_max={self.null_max_se:.2f}), obstruction_at="
                f"{self.obstruction_at or 'None'}. Per-overlap: [{'; '.join(rows)}]. "
                f"Chain: {self.provenance_chain_id}."
            )
        if self.status == "obstructed":
            worst = max(range(len(self.per_overlap_se)), key=lambda i: self.per_overlap_se[i])
            label = self.section_labels[self.overlaps[worst][0]]
            return (
                f"The {self.contrast_name} distributions for {self.feature_name} "
                f"cannot be reconciled with the pooled reference beyond the "
                f"measured noise ceiling. Largest disagreement localizes to the "
                f"'{label}' subgroup. Tolerance set adaptively from a "
                f"same-population control."
            )
        return (
            f"The {self.contrast_name} distributions for {self.feature_name} are "
            f"consistent with the pooled reference within the measured noise "
            f"ceiling. No structural disagreement at the current sample size."
        )


@dataclass(frozen=True)
class AdmissionFinding:
    contrast_name: str
    feature_name: str
    n_x: int
    n_y: int
    k3_norm: float
    k4_norm: float
    tau_3: float
    tau_4: float
    admit: bool
    blocking_component: Optional[str]
    provenance_chain_id: str = "ADT-v1.0::per-component-AND-gate"

    def format_for(self, audience: Audience) -> str:
        if audience == "overt":
            return (
                f"Per-component admission {self.contrast_name} on "
                f"{self.feature_name} (n={self.n_x:,} vs {self.n_y:,}): "
                f"admit={self.admit}, ||k3||={self.k3_norm:.3f}, "
                f"||k4||={self.k4_norm:.3f}, tau_3={self.tau_3}, "
                f"tau_4={self.tau_4}, blocking={self.blocking_component}. "
                f"AND-gate. Chain: {self.provenance_chain_id}."
            )
        if self.admit:
            return (
                f"For {self.feature_name} between {self.contrast_name}, both "
                f"higher-order signals exceed the noise-unit threshold. "
                f"Finding is admitted."
            )
        return (
            f"For {self.feature_name} between {self.contrast_name}, the finding "
            f"is refused at the noise-unit threshold. Blocking signal at order "
            f"{self.blocking_component}. Noise-floor refusal; the ratio test in "
            f"the divergence report is the primary signal."
        )


@dataclass(frozen=True)
class ICRFinding:
    contrast_name: str
    interface_name: str
    subgroup_a_label: str
    subgroup_b_label: str
    value_a: float
    value_b: float
    units: str = ""
    provenance_chain_id: str = "ADT-v1.0::intercept-to-compound-ratio"

    @property
    def relative_gap_pct(self) -> float:
        b = self.value_b
        return (self.value_a - b) / b * 100.0 if abs(b) > 1e-12 else float("nan")

    @property
    def ratio(self) -> float:
        b = self.value_b
        return self.value_a / b if abs(b) > 1e-12 else float("nan")

    def format_for(self, audience: Audience) -> str:
        u = f" {self.units}" if self.units else ""
        if audience == "overt":
            return (
                f"ICR analog {self.contrast_name} on {self.interface_name}: "
                f"{self.subgroup_a_label}={self.value_a:.4f}{u}, "
                f"{self.subgroup_b_label}={self.value_b:.4f}{u}, "
                f"gap={self.relative_gap_pct:+.1f}%, ratio={self.ratio:.2f}x. "
                f"Chain: {self.provenance_chain_id}."
            )
        direction = "higher" if self.value_a > self.value_b else "lower"
        return (
            f"On the {self.interface_name} interface, {self.subgroup_a_label} "
            f"carries a {abs(self.relative_gap_pct):.1f}% {direction} share than "
            f"{self.subgroup_b_label}. The asymmetry is at the cost-side of the "
            f"signal pathway."
        )
